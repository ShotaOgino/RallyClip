"""Per-point player identity and server detection (roadmap Phase 1).

numpy only (no torch / cv2 / av): this runs on the per-frame near/far player
records the runtime already produces (``DataPreprocessor.assign_players`` ->
``rescale_player_to_reference``), sampled at the model manifest fps, so it can run
in-engine (``RunRequest.save_player_tracks``) or offline on an
``scripts/eval/dump_heatmap_tracks.py`` .npz dump.

Coordinates are reference-frame pixels (``ref_size``: the artifact's screen size,
1280x720 for v1 artifacts, 1920x1080 for v0.6.0's v2 features; dumps without a
``ref_size`` key are read as 1280x720), y grows DOWN. Every geometric threshold is a
fraction of the frame or of a body/box size, so no rule depends on the reference size.
Keypoints are COCO-17 (0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles); NaN where absent.

NEAR-PLAYER-FIRST. On the beta-tennis footage the far player is essentially never
detected (shipped far record present in 0.02-0.1 % of frames; the raw pre-court-filter
detections hold a far-court person in < 0.01 %: the 960-px pose pass misses it). Every
decision therefore works from the near player alone; far-player paths are optional,
best effort, and marked "[far, optional]" below. A far-court crop / higher-res pose
pass is what would make them useful: the v0.6.0 runtime's net-line slots take the far
player from a far-court crop pass, and ``PlayerTracks`` prefers that record.

Heuristics (documented, deliberately simple; appearance features replace them later):

* ``detect_server`` - the server tosses and hits overhead within ~2.5 s of the point
  start: a wrist above the head with the arm extended. The receiver does not. Near
  raises -> near serves; near clearly seen and never reaching overhead -> far serves
  ("elimination"). [far, optional]: a far player's own raise is scored the same way
  when it happens to be detected.
* ``pick_far_player`` - [far, optional] best-effort far player from raw detections.
* ``track_identities`` - ids "A"/"B" (A = near player of the first point). Ends change
  after games 1, 3, 5, ... A changeover is scored from the gap before the point, a
  scale-free body signature of the near player, and serve-side continuity (the server
  side is the SAME across a changeover and FLIPS across the other game boundaries).
* ``serve_consistency`` - label-free sanity metric on the server sequence.
* ``analyze_points`` - all of the above plus the serve side (deuce / ad; ``analysis.serve_side``),
  one dict per point (the ``_points.json`` rows).
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# COCO-17 indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
HEAD_KPS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

SIDES = ("near", "far")
PLAYER_KEYS = ("box", "kp", "kpconf", "present")
POINTS_SCHEMA = "rallyclip.points/v1"

PointLike = Union[Tuple[float, float], Sequence[float], Mapping[str, Any]]


@dataclass(frozen=True)
class PlayerAnalysisConfig:
    """Heuristic knobs (not model-contract values; tuned on the 2026-08-24 matches)."""

    kp_conf_min: float = 0.3          # keypoints below this confidence are treated as missing
    # --- serve detection
    serve_pre_s: float = 1.0          # window starts this long before the decoded point start
    serve_post_s: float = 2.5         # ... and ends this long after it
    arm_extension_min: float = 0.7    # |wrist - shoulder| / torso length for an "extended" arm
    head_margin: float = 0.0          # wrist must be this many torso lengths above the head
    raise_frames_scale: float = 1.5   # side score = 1 - exp(-raise_frames / scale)
    min_side_score: float = 0.3       # below this a side does not "qualify" as server
    baseline_penalty: float = 0.75    # score multiplier when the player is NOT back near the baseline
    elimination: bool = True          # far serves if near is well seen, never raises, far unseen
    elimination_confidence: float = 0.3
    elimination_min_valid: float = 0.7  # near arm keypoints valid in >= this fraction of the window
    elimination_max_far_present: float = 0.3
    # --- identity / side changes
    gap_lo_s: float = 35.0            # gap score 0 at/below this gap
    gap_mid_s: float = 60.0           # gap score 0.6 here ("strong hint")
    gap_hi_s: float = 90.0            # gap score 1 at/above this gap
    candidate_gap_s: float = 25.0     # only boundaries with at least this gap can be a changeover
    signature_k: int = 3              # points pooled on each side of a boundary for the signature
    signature_noise_floor: float = 0.03  # min typical point-to-point log-ratio distance
    w_gap: float = 0.5
    w_signature: float = 0.2
    w_serve: float = 0.3
    change_threshold: float = 0.5     # combined score needed to declare a change of ends
    decision_softness: float = 0.06   # identity confidence: logistic width around the threshold
    smooth_half_window: int = 2       # server-side majority filter half width (points)
    # --- far player from raw (pre-court-filter) detections
    far_near_iou_max: float = 0.3     # a raw box overlapping the near box this much IS the near player
    far_max_height_ratio: float = 0.7  # far box height < this x near player height
    far_min_rise: float = 0.8         # far feet at least this many near-heights above the near feet
    far_max_dx: float = 0.25          # |far centre x - court centre x| <= this fraction of the width
    far_band_halfwidth: float = 0.06  # min half width of the far feet band (fraction of frame height)
    far_band_mads: float = 3.0        # far feet band = median +- this many MADs of pass-1 picks
    far_continuity_weight: float = 2.0  # cost weight of |dx| to the previous pick (fraction of width)
    far_continuity_s: float = 2.0     # previous pick counts for continuity within this many seconds


DEFAULT_CONFIG = PlayerAnalysisConfig()


# ------------------------------------------------------------------ data model


@dataclass
class SideTrack:
    """One side's per-frame player (near or far), NaN / present=False when absent."""

    box: np.ndarray      # (n, 4) x1, y1, x2, y2
    kp: np.ndarray       # (n, 17, 2)
    kpconf: np.ndarray   # (n, 17)
    present: np.ndarray  # (n,) bool

    @classmethod
    def empty(cls, n: int) -> "SideTrack":
        return cls(
            box=np.full((n, 4), np.nan, dtype=np.float32),
            kp=np.full((n, 17, 2), np.nan, dtype=np.float32),
            kpconf=np.full((n, 17), np.nan, dtype=np.float32),
            present=np.zeros(n, dtype=bool),
        )

    def valid_keypoints(self, conf_min: float) -> np.ndarray:
        """(n, 17, 2) keypoints with low-confidence / absent entries set to NaN.
        Frames with no confidence values at all keep their keypoints."""
        kp = np.array(self.kp, dtype=np.float64, copy=True)
        conf = np.asarray(self.kpconf, dtype=np.float64)
        low = np.isfinite(conf) & (conf < conf_min)
        kp[low] = np.nan
        kp[~np.asarray(self.present, dtype=bool)] = np.nan
        return kp


def pack_player_records(players: Sequence[Optional[Mapping[str, Any]]]) -> SideTrack:
    """Pack runtime player dicts ({box, keypoints, conf} or None) into a SideTrack.
    Same packing as scripts/eval/dump_heatmap_tracks.py."""
    n = len(players)
    track = SideTrack.empty(n)
    for i, p in enumerate(players):
        if p is None:
            continue
        track.present[i] = True
        track.box[i] = np.asarray(p["box"], dtype=np.float32).reshape(-1)[:4]
        track.kp[i] = np.asarray(p["keypoints"], dtype=np.float32).reshape(17, 2)
        conf = p.get("conf")
        if conf is not None:
            track.kpconf[i] = np.asarray(conf, dtype=np.float32).reshape(17)
    return track


RAW_MAX = 8
RAW_KEYS = ("raw_boxes", "raw_kp", "raw_kpconf", "raw_box_conf", "raw_count")


@dataclass
class RawDetections:
    """All pose detections per kept frame BEFORE the near-court filter, reference pixels,
    sorted by box confidence, NaN-padded to ``RAW_MAX``. The shipped preprocessor keeps
    only detections inside the near-court mask, which drops the far player in >99% of
    frames; ``pick_far_player`` recovers it from these."""

    boxes: np.ndarray     # (n, K, 4)
    kp: np.ndarray        # (n, K, 17, 2)
    kpconf: np.ndarray    # (n, K, 17)
    box_conf: np.ndarray  # (n, K)
    count: np.ndarray     # (n,)

    @classmethod
    def empty(cls, n: int, k: int = RAW_MAX) -> "RawDetections":
        return cls(
            boxes=np.full((n, k, 4), np.nan, dtype=np.float32),
            kp=np.full((n, k, 17, 2), np.nan, dtype=np.float32),
            kpconf=np.full((n, k, 17), np.nan, dtype=np.float32),
            box_conf=np.full((n, k), np.nan, dtype=np.float32),
            count=np.zeros(n, dtype=np.int16),
        )

    @classmethod
    def from_frames(cls, frames: Sequence[Optional[Mapping[str, Any]]], scale_xy: Tuple[float, float] = (1.0, 1.0),
                    max_det: int = RAW_MAX) -> "RawDetections":
        """Pack pose frame dicts ({boxes, keypoints, conf, box_conf} in source pixels, or
        None) into reference pixels (multiply by ``scale_xy``). Same packing as
        scripts/eval/dump_heatmap_tracks.py."""
        raw = cls.empty(len(frames), max_det)
        sx, sy = float(scale_xy[0]), float(scale_xy[1])
        for i, fd in enumerate(frames):
            if not fd:
                continue
            b = np.asarray(fd.get("boxes", np.empty((0, 4))), dtype=np.float32).reshape(-1, 4)
            k = np.asarray(fd.get("keypoints", np.empty((0, 17, 2))), dtype=np.float32).reshape(-1, 17, 2)
            c = np.asarray(fd.get("conf", np.empty((0, 17))), dtype=np.float32).reshape(-1, 17)
            bc = np.asarray(fd.get("box_conf", np.full((len(b),), -1.0)), dtype=np.float32).reshape(-1)
            order = (np.argsort(-bc, kind="stable") if len(bc) == len(b) else np.arange(len(b)))[:max_det]
            m = len(order)
            raw.count[i] = m
            if not m:
                continue
            raw.boxes[i, :m] = b[order] * np.array([sx, sy, sx, sy], dtype=np.float32)
            if len(k) == len(b):
                raw.kp[i, :m] = k[order] * np.array([sx, sy], dtype=np.float32)
            if len(c) == len(b):
                raw.kpconf[i, :m] = c[order]
            if len(bc) == len(b):
                raw.box_conf[i, :m] = bc[order]
        return raw

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {"raw_boxes": self.boxes, "raw_kp": self.kp, "raw_kpconf": self.kpconf,
                "raw_box_conf": self.box_conf, "raw_count": self.count}


@dataclass
class PlayerTracks:
    """Near/far player tracks at the analysis fps (1:1 with the model's kept frames).

    ``far`` is the effective far player: the shipped preprocessor's far record
    (``"shipped"``; v0.5.0 footage: almost always absent; v0.6.0 net-line slots: the
    far-court crop pass), with frames it misses filled from ``raw`` detections by
    ``pick_far_player`` when those are available (``"raw"`` if the shipped record is
    empty, else ``"shipped+raw"``).

    ``geometry``: the run's court geometry dict (``homography_px_to_m`` in reference
    pixels; npz key ``geometry_json``), used by ``analysis.serve_side``; None if absent."""

    timestamps: np.ndarray  # (n,) seconds in source time
    fps: float
    near: SideTrack
    far: SideTrack
    ref_size: Tuple[int, int] = (1280, 720)
    shipped_far: Optional[SideTrack] = None
    raw: Optional[RawDetections] = None
    far_source: str = "shipped"
    geometry: Optional[Dict[str, Any]] = None

    def with_raw(self, raw: Optional[RawDetections], cfg: "PlayerAnalysisConfig" = None) -> "PlayerTracks":
        """Attach raw detections and fill the far player from them (no-op for None).

        The shipped far record wins on every frame where it is present (v0.6.0+
        net-line slots take it from the far-court crop pass, which sees the far player
        far more often than the full-frame raw detections do); ``pick_far_player``
        fills the remaining frames. ``far_source``: "raw" when the shipped record is
        empty (v0.5.0 footage), "shipped+raw" when both contribute."""
        if raw is None:
            return self
        shipped = self.shipped_far if self.shipped_far is not None else self.far
        picked = pick_far_player(raw, self.near, self.ref_size, cfg or DEFAULT_CONFIG, fps=self.fps)
        keep = np.asarray(shipped.present, dtype=bool)
        if keep.any():
            far = SideTrack(
                box=np.where(keep[:, None], shipped.box, picked.box),
                kp=np.where(keep[:, None, None], shipped.kp, picked.kp),
                kpconf=np.where(keep[:, None], shipped.kpconf, picked.kpconf),
                present=keep | np.asarray(picked.present, dtype=bool),
            )
            source = "shipped+raw"
        else:
            far, source = picked, "raw"
        return PlayerTracks(self.timestamps, self.fps, self.near, far, self.ref_size,
                            shipped_far=shipped, raw=raw, far_source=source, geometry=self.geometry)

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def side(self, name: str) -> SideTrack:
        if name not in SIDES:
            raise ValueError(f"side must be 'near' or 'far', got {name!r}")
        return self.near if name == "near" else self.far

    def window(self, t0: float, t1: float) -> np.ndarray:
        """Frame indices with t0 <= timestamp <= t1."""
        lo = int(np.searchsorted(self.timestamps, t0, side="left"))
        hi = int(np.searchsorted(self.timestamps, t1, side="right"))
        return np.arange(lo, max(lo, hi))

    # -- construction
    @classmethod
    def from_arrays(cls, data: Mapping[str, Any]) -> "PlayerTracks":
        """From the dump/npz key layout: ``{near,far}_{box,kp,kpconf,present}``,
        ``timestamps``, ``fps``, ``ref_size``, optional ``geometry_json``. Missing player
        keys -> absent side."""
        timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
        n = timestamps.shape[0]
        fps = float(np.asarray(data["fps"]).reshape(())) if "fps" in data else (
            1.0 / float(np.median(np.diff(timestamps))) if n > 1 else 1.0)
        sides = {}
        for side in SIDES:
            if f"{side}_present" not in data:
                sides[side] = SideTrack.empty(n)
                continue
            sides[side] = SideTrack(
                box=np.asarray(data[f"{side}_box"], dtype=np.float32).reshape(n, 4),
                kp=np.asarray(data[f"{side}_kp"], dtype=np.float32).reshape(n, 17, 2),
                kpconf=np.asarray(data[f"{side}_kpconf"], dtype=np.float32).reshape(n, 17),
                present=np.asarray(data[f"{side}_present"], dtype=bool).reshape(n),
            )
        ref = tuple(int(v) for v in np.asarray(data["ref_size"]).reshape(-1)[:2]) if "ref_size" in data else (1280, 720)
        geometry = None
        if "geometry_json" in data:
            try:
                geometry = json.loads(str(np.asarray(data["geometry_json"]).reshape(())))
            except ValueError:
                geometry = None
        tracks = cls(timestamps=timestamps, fps=fps, near=sides["near"], far=sides["far"], ref_size=ref,
                     geometry=geometry if isinstance(geometry, dict) else None)
        if all(k in data for k in RAW_KEYS):
            k = np.asarray(data["raw_boxes"]).shape[1]
            raw = RawDetections(
                boxes=np.asarray(data["raw_boxes"], dtype=np.float32).reshape(n, k, 4),
                kp=np.asarray(data["raw_kp"], dtype=np.float32).reshape(n, k, 17, 2),
                kpconf=np.asarray(data["raw_kpconf"], dtype=np.float32).reshape(n, k, 17),
                box_conf=np.asarray(data["raw_box_conf"], dtype=np.float32).reshape(n, k),
                count=np.asarray(data["raw_count"]).reshape(n),
            )
            tracks = tracks.with_raw(raw)
        return tracks

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "PlayerTracks":
        with np.load(str(path), allow_pickle=False) as data:
            return cls.from_arrays({k: data[k] for k in data.files})

    @classmethod
    def from_records(
        cls,
        near_players: Sequence[Optional[Mapping[str, Any]]],
        far_players: Sequence[Optional[Mapping[str, Any]]],
        *,
        fps: float,
        start_time: float = 0.0,
        ref_size: Tuple[int, int] = (1280, 720),
        raw: Optional[RawDetections] = None,
        geometry: Optional[Dict[str, Any]] = None,
    ) -> "PlayerTracks":
        """From in-engine kept-frame records (one near/far dict-or-None per frame);
        ``raw`` (pre-court-filter detections, same frames) enables ``pick_far_player``;
        ``geometry`` = the court geometry dict (serve side)."""
        if len(near_players) != len(far_players):
            raise ValueError("near/far record counts differ")
        n = len(near_players)
        timestamps = float(start_time) + np.arange(n, dtype=np.float64) / float(fps)
        tracks = cls(
            timestamps=timestamps,
            fps=float(fps),
            near=pack_player_records(near_players),
            far=pack_player_records(far_players),
            ref_size=(int(ref_size[0]), int(ref_size[1])),
            geometry=geometry,
        )
        return tracks.with_raw(raw)

    def truncated(self, n: int) -> "PlayerTracks":
        def cut(s: SideTrack) -> SideTrack:
            return SideTrack(s.box[:n], s.kp[:n], s.kpconf[:n], s.present[:n])

        raw = None if self.raw is None else RawDetections(
            self.raw.boxes[:n], self.raw.kp[:n], self.raw.kpconf[:n], self.raw.box_conf[:n], self.raw.count[:n])
        return PlayerTracks(self.timestamps[:n], self.fps, cut(self.near), cut(self.far), self.ref_size,
                            shipped_far=None if self.shipped_far is None else cut(self.shipped_far),
                            raw=raw, far_source=self.far_source, geometry=self.geometry)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "fps": np.asarray(float(self.fps)),
            "ref_size": np.asarray(self.ref_size, dtype=np.int32),
        }
        for side in SIDES:
            # far_* stay the shipped preprocessor's record (dump-compatible); the
            # raw-picked far player is re-derived from raw_* on load.
            s = self.shipped_far if side == "far" and self.shipped_far is not None else self.side(side)
            out[f"{side}_box"] = np.asarray(s.box, dtype=np.float32)
            out[f"{side}_kp"] = np.asarray(s.kp, dtype=np.float32)
            out[f"{side}_kpconf"] = np.asarray(s.kpconf, dtype=np.float32)
            out[f"{side}_present"] = np.asarray(s.present, dtype=bool)
        if self.raw is not None:
            out.update(self.raw.to_arrays())
        if self.geometry is not None:
            out["geometry_json"] = np.asarray(json.dumps(self.geometry))
        return out

    def save_npz(self, path: Union[str, Path], **extra: Any) -> Path:
        """Write the dump-compatible .npz (extra arrays, e.g. pointness, ride along)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.to_arrays()
        arrays.update({k: np.asarray(v) for k, v in extra.items()})
        with open(path, "wb") as fh:
            np.savez(fh, **arrays)
        return path


# ------------------------------------------------------------------ geometry


def _nanmin(a: np.ndarray, axis: int) -> np.ndarray:
    filled = np.where(np.isnan(a), np.inf, a).min(axis=axis)
    return np.where(np.isinf(filled), np.nan, filled)


def _nanmedian(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return np.full(a.shape[1:] if axis == 0 else (), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN, silently
        return np.nanmedian(a, axis=axis)


def _dist(kp: np.ndarray, i: int, j: int) -> np.ndarray:
    d = kp[:, i, :] - kp[:, j, :]
    return np.sqrt((d * d).sum(axis=-1))


def _mid(kp: np.ndarray, i: int, j: int) -> np.ndarray:
    return (kp[:, i, :] + kp[:, j, :]) / 2.0


def _torso_length(kp: np.ndarray) -> np.ndarray:
    """Shoulder-mid to hip-mid, falling back to a same-side shoulder-hip distance."""
    sm, hm = _mid(kp, L_SHOULDER, R_SHOULDER), _mid(kp, L_HIP, R_HIP)
    d = np.sqrt(((sm - hm) ** 2).sum(axis=-1))
    side = np.stack([_dist(kp, L_SHOULDER, L_HIP), _dist(kp, R_SHOULDER, R_HIP)], axis=1)
    fallback = _nanmedian(side.T, axis=0) if side.size else np.full(kp.shape[0], np.nan)
    return np.where(np.isfinite(d), d, fallback)


def _body_scale(kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Per-frame length unit: torso length; else 1.4 x shoulder width; else 0.3 x box height.
    Torso length is used rather than shoulder width because a server stands side-on to a
    rear camera, which collapses the projected shoulder width."""
    torso = _torso_length(kp)
    shoulder = _dist(kp, L_SHOULDER, R_SHOULDER) * 1.4
    box_h = (box[:, 3] - box[:, 1]) * 0.3
    scale = np.where(np.isfinite(torso) & (torso > 1e-3), torso, shoulder)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, box_h)
    return np.where(np.isfinite(scale) & (scale > 1e-3), scale, np.nan)


def arm_raised(kp: np.ndarray, box: np.ndarray, cfg: PlayerAnalysisConfig = DEFAULT_CONFIG) -> Tuple[np.ndarray, np.ndarray]:
    """Per frame: (raised, arm_valid).

    raised = for either arm, wrist above the head (y smaller than the highest head
    keypoint, minus ``head_margin`` torso lengths) AND the arm extended
    (|wrist - shoulder| >= ``arm_extension_min`` torso lengths). Without head keypoints
    the head is assumed 0.35 torso lengths above the shoulder line.
    arm_valid = at least one wrist+shoulder pair and a body scale are observed.
    """
    kp = np.asarray(kp, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    n = kp.shape[0]
    scale = _body_scale(kp, box)
    head_y = _nanmin(kp[:, HEAD_KPS, 1], axis=1)
    shoulder_y = _nanmin(kp[:, (L_SHOULDER, R_SHOULDER), 1], axis=1)
    head_y = np.where(np.isfinite(head_y), head_y, shoulder_y - 0.35 * scale)
    raised = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    for shoulder, wrist in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)):
        wy = kp[:, wrist, 1]
        ext = _dist(kp, wrist, shoulder) / scale
        ok = np.isfinite(wy) & np.isfinite(ext) & np.isfinite(head_y)
        valid |= ok
        with np.errstate(invalid="ignore"):
            raised |= ok & (wy < head_y - cfg.head_margin * scale) & (ext >= cfg.arm_extension_min)
    return raised, valid


# ------------------------------------------------------------------ far player


def _iou_many(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]), 0, None)
    inter = ix * iy
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    with np.errstate(all="ignore"):
        return np.where(area > 0, inter / area, 0.0)


def pick_far_player(
    raw: RawDetections,
    near: SideTrack,
    ref_size: Tuple[int, int] = (1280, 720),
    cfg: PlayerAnalysisConfig = DEFAULT_CONFIG,
    fps: float = 5.0,
) -> SideTrack:
    """Recover the far player per frame from the raw (pre-court-filter) detections.

    Candidates: detections that are not the near player (IoU with the near box <=
    ``far_near_iou_max``), whose feet are at least ``far_min_rise`` near-player heights
    above the near player's feet, whose box is smaller than ``far_max_height_ratio`` x
    the near player's height (this frame, else the match median), and whose centre is
    within ``far_max_dx`` of the court centre x (spectators, coaches and players on the
    neighbouring courts stand at the image sides). Pass 1 picks the candidate closest to the court centre x (the
    near player's median x over the match) to estimate the far player's feet band
    (median +- ``far_band_mads`` MADs, at least ``far_band_halfwidth``); pass 2 keeps
    only candidates in that band (rejects umpire chair, ball kids at the net, people
    behind the far fence) and picks the lowest cost
    ``|cx - centre| / W + far_continuity_weight * |cx - previous pick| / W`` (previous
    pick within ``far_continuity_s``). Isolated single-frame picks that jump away from
    both neighbours are dropped. Returns a SideTrack (absent where nothing qualifies).
    """
    n, k = raw.boxes.shape[:2]
    width, height = float(ref_size[0]), float(ref_size[1])
    out = SideTrack.empty(n)
    if n == 0:
        return out
    boxes = np.asarray(raw.boxes, dtype=np.float64)
    count = np.asarray(raw.count).reshape(n)
    slot_ok = np.arange(k)[None, :] < count[:, None]
    cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
    foot = boxes[..., 3]
    h = boxes[..., 3] - boxes[..., 1]
    valid = slot_ok & np.isfinite(cx) & np.isfinite(foot) & (h > 0)

    nbox = np.asarray(near.box, dtype=np.float64)
    npres = np.asarray(near.present, dtype=bool) & np.isfinite(nbox).all(axis=1)
    if npres.any():
        centre_x = float(np.median((nbox[npres, 0] + nbox[npres, 2]) / 2.0))
        near_h_med = float(np.median(nbox[npres, 3] - nbox[npres, 1]))
        near_foot_med = float(np.median(nbox[npres, 3]))
    else:
        centre_x, near_h_med, near_foot_med = width / 2.0, 0.4 * height, 0.95 * height
    near_h = np.where(npres, nbox[:, 3] - nbox[:, 1], near_h_med)
    near_foot = np.where(npres, nbox[:, 3], near_foot_med)

    for i in np.flatnonzero(npres & valid.any(axis=1)):
        valid[i] &= _iou_many(np.nan_to_num(boxes[i]), nbox[i]) <= cfg.far_near_iou_max
    valid &= foot < (near_foot - cfg.far_min_rise * near_h)[:, None]
    valid &= h < (cfg.far_max_height_ratio * near_h)[:, None]
    dx = np.abs(cx - centre_x) / width
    valid &= dx <= cfg.far_max_dx

    # pass 1: closest to the court centre -> far feet band
    cost1 = np.where(valid, dx, np.inf)
    best1 = cost1.argmin(axis=1)
    has1 = np.isfinite(cost1[np.arange(n), best1])
    if not has1.any():
        return out
    f1 = foot[np.arange(n), best1][has1]
    band_c = float(np.median(f1))
    band_w = max(cfg.far_band_mads * float(np.median(np.abs(f1 - band_c))), cfg.far_band_halfwidth * height)
    valid &= np.abs(foot - band_c) <= band_w

    # pass 2: centre + continuity, sequential
    window = max(1.0, cfg.far_continuity_s * float(fps))  # frames
    pick = np.full(n, -1, dtype=int)
    last_i, last_cx = -10**9, None
    for i in range(n):
        cand = np.flatnonzero(valid[i])
        if cand.size == 0:
            continue
        cost = dx[i, cand].copy()
        if last_cx is not None and (i - last_i) <= window:
            cost += cfg.far_continuity_weight * np.abs(cx[i, cand] - last_cx) / width
        j = int(cand[int(np.argmin(cost))])
        pick[i] = j
        last_i, last_cx = i, cx[i, j]

    # drop isolated jumps: a pick whose both neighbours are picked and agree with each
    # other but not with it
    jump = 0.08 * width
    for i in range(1, n - 1):
        if pick[i] < 0 or pick[i - 1] < 0 or pick[i + 1] < 0:
            continue
        a, b, c = cx[i - 1, pick[i - 1]], cx[i, pick[i]], cx[i + 1, pick[i + 1]]
        if abs(a - c) < jump and abs(b - a) >= jump and abs(b - c) >= jump:
            pick[i] = -1

    rows = np.flatnonzero(pick >= 0)
    cols = pick[rows]
    out.present[rows] = True
    out.box[rows] = raw.boxes[rows, cols]
    out.kp[rows] = raw.kp[rows, cols]
    out.kpconf[rows] = raw.kpconf[rows, cols]
    return out


# ------------------------------------------------------------------ points


def _point_bounds(point: PointLike) -> Tuple[float, float]:
    if isinstance(point, Mapping):
        return float(point["start_s"]), float(point["end_s"])
    s, e = point[0], point[1]
    return float(s), float(e)


def _baseline_ranks(tracks: PlayerTracks, side: str) -> Optional[np.ndarray]:
    """Sorted foot-y (box bottom) of the side over the whole match, or None."""
    s = tracks.side(side)
    foot = np.asarray(s.box[:, 3], dtype=np.float64)[np.asarray(s.present, dtype=bool)]
    foot = foot[np.isfinite(foot)]
    return np.sort(foot) if foot.size >= 10 else None


def _depth_rank(sorted_foot: Optional[np.ndarray], value: float, side: str) -> Optional[float]:
    """0..1, 1 = as far back (towards that side's baseline) as this player ever is.
    Near baseline = feet low in the image (large y); far baseline = feet high (small y)."""
    if sorted_foot is None or not np.isfinite(value):
        return None
    rank = float(np.searchsorted(sorted_foot, value)) / float(sorted_foot.size)
    return rank if side == "near" else 1.0 - rank


# ------------------------------------------------------------------ serve


def _side_serve_evidence(tracks, side, idx, t_start, cfg, foot_sorted) -> Dict[str, Any]:
    s = tracks.side(side)
    n = idx.size
    ev: Dict[str, Any] = {"frames": int(n), "present_frac": 0.0, "arm_valid_frac": 0.0,
                          "raise_frames": 0, "first_raise_s": None, "baseline_rank": None, "score": 0.0}
    if n == 0:
        return ev
    present = np.asarray(s.present, dtype=bool)[idx]
    ev["present_frac"] = round(float(present.mean()), 3)
    kp = s.valid_keypoints(cfg.kp_conf_min)[idx]
    raised, valid = arm_raised(kp, s.box[idx], cfg)
    ev["arm_valid_frac"] = round(float(valid.mean()), 3)
    k = int(raised.sum())
    ev["raise_frames"] = k
    if k:
        first = int(np.flatnonzero(raised)[0])
        ev["first_raise_s"] = round(float(tracks.timestamps[idx[first]] - t_start), 2)
        # baseline check on the frame just before / at the first raise
        foot = s.box[idx[: first + 1], 3].astype(np.float64)
        foot = foot[np.isfinite(foot)]
        rank = _depth_rank(foot_sorted, float(foot[-1]) if foot.size else float("nan"), side)
        ev["baseline_rank"] = None if rank is None else round(rank, 3)
    score = 1.0 - math.exp(-k / cfg.raise_frames_scale) if k else 0.0
    if ev["baseline_rank"] is not None and ev["baseline_rank"] < 0.5:
        score *= cfg.baseline_penalty
    ev["score"] = round(score, 3)
    return ev


def detect_server(
    point: PointLike,
    tracks: PlayerTracks,
    cfg: PlayerAnalysisConfig = DEFAULT_CONFIG,
    *,
    _foot_cache: Optional[Dict[str, Optional[np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Which side serves this point: {"server": "near"|"far"|None, "confidence", "evidence"}.

    In [start - serve_pre_s, start + serve_post_s] each side scores
    ``1 - exp(-raise_frames / raise_frames_scale)`` where a raise frame has a wrist above
    the head with the arm extended (see ``arm_raised``), times ``baseline_penalty`` when
    the player is in the front half of their own depth range at the first raise. The
    higher-scoring side (score >= ``min_side_score``) serves; confidence = its score minus
    the other side's. In practice only the near side ever scores (the far player is not
    detected; its raise path is [far, optional]). When neither side qualifies but the
    near player is well observed (arm keypoints valid in >= ``elimination_min_valid`` of
    frames: standing, then reacting, never reaching overhead = receiving) and the far
    player is not seen serving (present in <= ``elimination_max_far_present`` of the
    window), the far player is returned by elimination with a low fixed confidence
    (``elimination_confidence``). Otherwise server None, confidence 0.
    """
    t_start, t_end = _point_bounds(point)
    t0, t1 = t_start - cfg.serve_pre_s, min(t_end, t_start + cfg.serve_post_s)
    idx = tracks.window(t0, max(t0, t1))
    cache = _foot_cache if _foot_cache is not None else {}
    for side in SIDES:
        if side not in cache:
            cache[side] = _baseline_ranks(tracks, side)
    ev = {side: _side_serve_evidence(tracks, side, idx, t_start, cfg, cache[side]) for side in SIDES}
    evidence: Dict[str, Any] = {"window_s": [round(t0, 2), round(t1, 2)], "method": "none", **ev}
    sn, sf = ev["near"]["score"], ev["far"]["score"]
    best = "near" if sn >= sf else "far"
    hi, lo = max(sn, sf), min(sn, sf)
    if hi >= cfg.min_side_score and hi > lo:
        evidence["method"] = "raise"
        return {"server": best, "confidence": round(hi - lo, 3), "evidence": evidence}
    if (
        cfg.elimination
        and ev["near"]["raise_frames"] == 0
        and ev["near"]["arm_valid_frac"] >= cfg.elimination_min_valid
        and ev["far"]["present_frac"] <= cfg.elimination_max_far_present
    ):
        evidence["method"] = "elimination"
        return {"server": "far", "confidence": cfg.elimination_confidence, "evidence": evidence}
    return {"server": None, "confidence": 0.0, "evidence": evidence}


# ------------------------------------------------------------------ body signature

# name -> ((a, b), (c, d)) segment-length ratio |a-b| / |c-d|; each averaged over left/right
# where the pair is mirrored. Scale-free, so the near player's distance to camera cancels.
_SIGNATURE_RATIOS: Dict[str, Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]] = {
    "shoulder_hip": (((L_SHOULDER, R_SHOULDER),), ((L_HIP, R_HIP),)),
    "torso_leg": (((L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)), ((L_HIP, L_ANKLE), (R_HIP, R_ANKLE))),
    "thigh_shin": (((L_HIP, L_KNEE), (R_HIP, R_KNEE)), ((L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE))),
    "upperarm_forearm": (((L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW)), ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST))),
    "shoulder_torso": (((L_SHOULDER, R_SHOULDER),), ((L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP))),
}
SIGNATURE_KEYS: Tuple[str, ...] = tuple(_SIGNATURE_RATIOS)


def body_signature(kp: np.ndarray) -> Tuple[np.ndarray, int]:
    """Median log limb-length ratios over frames -> (len(SIGNATURE_KEYS),), frames used.

    Frames contribute per ratio wherever both lengths are observed; NaN where a ratio was
    never observed. Logs make ratios symmetric (r vs 1/r) for distance computations."""
    kp = np.asarray(kp, dtype=np.float64)
    out = np.full(len(SIGNATURE_KEYS), np.nan)
    if kp.shape[0] == 0:
        return out, 0
    used = np.zeros(kp.shape[0], dtype=bool)
    for k, name in enumerate(SIGNATURE_KEYS):
        num_pairs, den_pairs = _SIGNATURE_RATIOS[name]
        num = _nanmedian(np.stack([_dist(kp, a, b) for a, b in num_pairs]), axis=0)
        den = _nanmedian(np.stack([_dist(kp, a, b) for a, b in den_pairs]), axis=0)
        with np.errstate(all="ignore"):
            r = np.log(num / den)
        ok = np.isfinite(r)
        if ok.any():
            out[k] = float(np.median(r[ok]))
            used |= ok
    return out, int(used.sum())


# ------------------------------------------------------------------ identity


def _gap_score(gap: float, cfg: PlayerAnalysisConfig) -> float:
    """0 below gap_lo_s, 0.6 at gap_mid_s, 1 at gap_hi_s and above (piecewise linear)."""
    if not np.isfinite(gap) or gap <= cfg.gap_lo_s:
        return 0.0
    if gap <= cfg.gap_mid_s:
        return 0.6 * (gap - cfg.gap_lo_s) / max(1e-6, cfg.gap_mid_s - cfg.gap_lo_s)
    if gap <= cfg.gap_hi_s:
        return 0.6 + 0.4 * (gap - cfg.gap_mid_s) / max(1e-6, cfg.gap_hi_s - cfg.gap_mid_s)
    return 1.0


def smooth_server_sides(
    servers: Sequence[Optional[str]],
    confidences: Optional[Sequence[float]] = None,
    half_window: int = 2,
) -> List[Optional[str]]:
    """Confidence-weighted majority of near/far over +-half_window points (None ignored)."""
    n = len(servers)
    conf = list(confidences) if confidences is not None else [1.0] * n
    out: List[Optional[str]] = []
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        vote = 0.0
        for j in range(lo, hi):
            w = max(1e-3, float(conf[j] or 0.0)) * (2.0 if j == i else 1.0)
            if servers[j] == "near":
                vote += w
            elif servers[j] == "far":
                vote -= w
        out.append("near" if vote > 0 else "far" if vote < 0 else servers[i])
    return out


def _serve_boundary_score(smoothed: Sequence[Optional[str]], i: int, k: int) -> Tuple[float, Optional[str]]:
    """Serve-side continuity across boundary i|i+1 from k smoothed points per side.
    +1: same side before and after (changeover-compatible); -1: side flips (a game
    boundary WITHOUT a change of ends); 0: unknown."""
    before = [s for s in smoothed[max(0, i - k + 1): i + 1] if s is not None]
    after = [s for s in smoothed[i + 1: i + 1 + k] if s is not None]
    if not before or not after:
        return 0.0, None
    b = max(set(before), key=before.count)
    a = max(set(after), key=after.count)
    return (1.0, "same") if a == b else (-1.0, "flip")


def track_identities(
    points: Sequence[PointLike],
    tracks: PlayerTracks,
    servers: Optional[Sequence[Mapping[str, Any]]] = None,
    cfg: PlayerAnalysisConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """Per point {near_id, far_id, side_change_before, confidence, evidence}; ids "A"/"B".

    The near player of the first point is "A". For every boundary between consecutive
    points with gap >= ``candidate_gap_s``:

      score = w_gap * gap_score + w_signature * signature_change + w_serve * serve_term

    * gap_score: 0 at <= gap_lo_s (35 s), 0.6 at gap_mid_s (60 s), 1 at >= gap_hi_s (90 s).
      Change-of-ends breaks are 60-120 s, other game breaks ~30-60 s, points 15-30 s.
    * signature_change: distance between the near player's body signature (median log
      limb ratios, ``body_signature``) pooled over ``signature_k`` points before vs after,
      divided by the match's typical point-to-point distance, squashed to 0..1.
    * serve_term: from detected server sides (smoothed): +1 same side across the boundary
      (a changeover keeps the server side), -1 flip (game boundary without change), 0 n/a.
      Pass ``servers`` (detect_server outputs) or they are computed here.

    score >= ``change_threshold`` flips the ids. Confidence per point is the product of
    per-decision confidences ``sigmoid(|score - threshold| / decision_softness)`` of every
    candidate boundary so far: identity errors accumulate along the match.
    """
    bounds = [_point_bounds(p) for p in points]
    n = len(bounds)
    if n == 0:
        return []
    if servers is None:
        cache: Dict[str, Optional[np.ndarray]] = {}
        servers = [detect_server(b, tracks, cfg, _foot_cache=cache) for b in bounds]
    smoothed = smooth_server_sides(
        [s.get("server") for s in servers], [s.get("confidence", 0.0) for s in servers], cfg.smooth_half_window
    )

    near_kp = tracks.near.valid_keypoints(cfg.kp_conf_min)
    sigs = np.full((n, len(SIGNATURE_KEYS)), np.nan)
    sig_frames = np.zeros(n, dtype=int)
    for i, (s, e) in enumerate(bounds):
        idx = tracks.window(s, e)
        sigs[i], sig_frames[i] = body_signature(near_kp[idx])
    # typical point-to-point signature distance within the match (noise floor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # points without any signature
        step = np.nanmean(np.abs(np.diff(sigs, axis=0)), axis=1) if n > 1 else np.array([])
    step = step[np.isfinite(step)]
    noise = max(float(np.median(step)) if step.size else 0.0, cfg.signature_noise_floor)

    def pooled(lo: int, hi: int) -> np.ndarray:
        block = sigs[max(0, lo): max(0, hi)]
        return _nanmedian(block, axis=0) if block.shape[0] else np.full(len(SIGNATURE_KEYS), np.nan)

    results: List[Dict[str, Any]] = []
    current = ("A", "B")
    conf = 1.0
    for i in range(n):
        boundary: Optional[Dict[str, Any]] = None
        if i > 0:
            gap = bounds[i][0] - bounds[i - 1][1]
            boundary = {"gap_s": round(gap, 1)}
            if gap >= cfg.candidate_gap_s:
                g = _gap_score(gap, cfg)
                before, after = pooled(i - cfg.signature_k, i), pooled(i, i + cfg.signature_k)
                both = np.isfinite(before) & np.isfinite(after)
                if both.any():
                    d = float(np.mean(np.abs(before[both] - after[both])))
                    sig = float(1.0 - math.exp(-max(0.0, d / noise - 1.0)))
                else:
                    d, sig = float("nan"), 0.0
                serve, serve_kind = _serve_boundary_score(smoothed, i - 1, cfg.signature_k)
                score = cfg.w_gap * g + cfg.w_signature * sig + cfg.w_serve * serve
                changed = score >= cfg.change_threshold
                # P(decision right) ~ logistic in the distance from the threshold
                conf *= 1.0 / (1.0 + math.exp(-abs(score - cfg.change_threshold) / cfg.decision_softness))
                boundary.update({
                    "candidate": True, "gap_score": round(g, 3),
                    "signature_dist": None if not np.isfinite(d) else round(d, 4),
                    "signature_score": round(sig, 3), "serve": serve_kind,
                    "score": round(score, 3), "side_change": bool(changed),
                })
                if changed:
                    current = (current[1], current[0])
        results.append({
            "near_id": current[0],
            "far_id": current[1],
            "side_change_before": bool(boundary and boundary.get("side_change")),
            "confidence": round(conf, 3),
            "evidence": {
                "boundary": boundary,
                "signature": [None if not np.isfinite(v) else round(float(v), 4) for v in sigs[i]],
                "signature_frames": int(sig_frames[i]),
                "server_side_smoothed": smoothed[i],
            },
        })
    return results


# ------------------------------------------------------------------ consistency


def _runs(seq: Sequence[Any]) -> List[int]:
    runs: List[int] = []
    prev = object()
    for v in seq:
        if runs and v == prev:
            runs[-1] += 1
        else:
            runs.append(1)
        prev = v
    return runs


def _run_stats(seq: Sequence[Any]) -> Dict[str, Any]:
    runs = _runs(seq)
    hist: Dict[str, int] = {}
    for r in runs:
        key = str(r) if r < 10 else "10+"
        hist[key] = hist.get(key, 0) + 1
    return {
        "n_runs": len(runs),
        "runs": runs,
        "run_length_hist": dict(sorted(hist.items(), key=lambda kv: (len(kv[0]), kv[0]))),
        "median_run": float(np.median(runs)) if runs else None,
        "short_run_frac": round(sum(1 for r in runs if r <= 2) / len(runs), 3) if runs else None,
        "runs_4_to_7_frac": round(sum(1 for r in runs if 4 <= r <= 7) / len(runs), 3) if runs else None,
    }


def serve_consistency(points_with_server: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Label-free sanity metric for a server sequence.

    The same player serves a whole game (4-7 points typically, deuce games longer,
    tiebreaks aside) and the server alternates between games. Points whose server is
    None are dropped from the run computation and counted in ``none_frac``.

    * ``by_player`` (when items carry ``server_id`` "A"/"B"): runs should mostly be 4-7
      long -> good: ``runs_4_to_7_frac`` >= ~0.6, ``short_run_frac`` (runs of 1-2,
      almost always detection or identity errors) <= ~0.15.
    * ``by_side`` (``server`` near/far): the server side only flips every TWO games
      (changeovers keep it), so good runs are ~8-14 points and short runs are errors.
    * ``none_frac``: fraction of points with no server; good is <= ~0.1.
    """
    n = len(points_with_server)
    sides = [p.get("server") for p in points_with_server]
    known = [s for s in sides if s is not None]
    out: Dict[str, Any] = {
        "n_points": n,
        "none_frac": round(1.0 - len(known) / n, 3) if n else None,
        "server_counts": {k: sum(1 for s in sides if s == k) for k in ("near", "far")},
        "by_side": _run_stats(known),
    }
    ids = [p.get("server_id") for p in points_with_server if p.get("server_id") is not None]
    if ids:
        out["by_player"] = _run_stats(ids)
        out["server_id_counts"] = {k: ids.count(k) for k in ("A", "B")}
    return out


# ------------------------------------------------------------------ combined


def analyze_points(
    points: Sequence[PointLike],
    tracks: PlayerTracks,
    cfg: PlayerAnalysisConfig = DEFAULT_CONFIG,
    *,
    geometry: Any = None,
    game_ids: Optional[Sequence[Optional[int]]] = None,
    score_confidence: Optional[float] = None,
    serve_cfg: Any = None,
) -> List[Dict[str, Any]]:
    """One dict per point: start_s, end_s, server, server_confidence, server_id,
    near_id, far_id, identity_confidence, side_change_before, near_present_frac,
    far_present_frac, plus the serve-side fields ``analysis.serve_side.ROW_KEYS``
    (serve_side, serve_side_confidence, serve_side_source, serve_side_score,
    serve_side_agree, serve_side_visual*). JSON-serialisable.

    Serve side: visual from ``geometry`` (default ``tracks.geometry``; None -> no visual
    side); score parity from ``game_ids`` (per-point game index from a winners list /
    scoreboard, confidence ``score_confidence`` or the config's ``score_confidence``),
    else from the detected server_id sequence via ``analysis.score`` segmentation
    (``segmentation_confidence``)."""
    from analysis.serve_side import DEFAULT_SERVE_SIDE_CONFIG, game_ids_from_servers, serve_sides

    bounds = [_point_bounds(p) for p in points]
    cache: Dict[str, Optional[np.ndarray]] = {}
    servers = [detect_server(b, tracks, cfg, _foot_cache=cache) for b in bounds]
    idents = track_identities(bounds, tracks, servers, cfg)
    rows: List[Dict[str, Any]] = []
    for (s, e), srv, ident in zip(bounds, servers, idents):
        idx = tracks.window(s, e)
        near_frac = float(np.asarray(tracks.near.present)[idx].mean()) if idx.size else 0.0
        far_frac = float(np.asarray(tracks.far.present)[idx].mean()) if idx.size else 0.0
        side = srv["server"]
        server_id = None if side is None else (ident["near_id"] if side == "near" else ident["far_id"])
        rows.append({
            "start_s": round(s, 3),
            "end_s": round(e, 3),
            "server": side,
            "server_confidence": srv["confidence"],
            "server_id": server_id,
            "near_id": ident["near_id"],
            "far_id": ident["far_id"],
            "identity_confidence": ident["confidence"],
            "side_change_before": ident["side_change_before"],
            "near_present_frac": round(near_frac, 3),
            "far_present_frac": round(far_frac, 3),
        })
    scfg = serve_cfg or DEFAULT_SERVE_SIDE_CONFIG
    if game_ids is None:
        game_ids = game_ids_from_servers([r["server_id"] for r in rows])
        if score_confidence is None:
            score_confidence = scfg.segmentation_confidence
    sides = serve_sides(bounds, tracks, tracks.geometry if geometry is None else geometry, servers,
                        game_ids, score_confidence=score_confidence, cfg=scfg)
    for row, side in zip(rows, sides):
        row.update(side)
    return rows


def points_document(rows: Sequence[Mapping[str, Any]], cfg: PlayerAnalysisConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """The ``<output_name>_points.json`` payload: rows + consistency + side changes +
    the serve-side cross-check (``analysis.serve_side.cross_check``)."""
    from analysis.serve_side import cross_check

    return {
        "schema": POINTS_SCHEMA,
        "points": list(rows),
        "consistency": serve_consistency(rows),
        "serve_side_check": cross_check(rows),
        "side_changes_s": [r["start_s"] for r in rows if r.get("side_change_before")],
        "config": asdict(cfg),
    }


def iter_point_bounds(points: Iterable[PointLike]) -> List[Tuple[float, float]]:
    return [_point_bounds(p) for p in points]
