"""Net-line player slots, shared by training preprocess and the runtime.

Near slot: the usual near-player pick among full-frame detections whose box
bottom is below the net line. Far slot: the far-crop detection that sits above
the net line and, in court meters, is closest to the center line.

numpy only (no h5py/torch) so the frozen app can import it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Court meters: origin at the net center, u to the camera's right, v away from
# the camera. Far-slot gate: doubles half-width plus room to run wide, and from
# just short of the net to the far baseline plus run-back.
FAR_MAX_ABS_U = 10.97 / 2 + 2.0
FAR_MIN_V = -1.0
FAR_MAX_V = 23.77 / 2 + 6.0
FAR_TIE_U = 1.0
# Preferred far-slot region (inside the gate above): the far half-court, doubles
# sidelines plus FAR_COURT_MARGIN_U, from just short of the net to the baseline
# plus FAR_COURT_RUNBACK_V. Gate-only candidates (out wide / deep behind the
# baseline: line judges, ball kids, bystanders) are a fallback, counted in stats.
FAR_COURT_MARGIN_U = 1.0
FAR_COURT_MIN_V = -0.5
FAR_COURT_RUNBACK_V = 4.0


@dataclass
class CourtGeometry:
    """Per-video court from the line-map court model (static camera)."""

    net_a: np.ndarray  # net-bottom line endpoints, pixels
    net_b: np.ndarray
    px_to_m: np.ndarray  # 3x3 homography, pixels -> court meters
    source_sha256: str = ""
    # Far-baseline corners (pixels) when the court model found them; they drive
    # the adaptive far-court crop (extraction.crop_pass.adaptive_crop).
    far_left: Optional[np.ndarray] = field(default=None, compare=False)
    far_right: Optional[np.ndarray] = field(default=None, compare=False)

    @classmethod
    def from_dict(cls, data: dict, source_sha256: str = "") -> "CourtGeometry":
        if not data.get("net_line_px") or not data.get("homography_px_to_m"):
            raise ValueError("Court geometry has no net line or homography")
        a, b = (np.asarray(p, dtype=np.float64) for p in data["net_line_px"])
        points = data.get("points_px") or {}
        corners = [None if points.get(k) is None else np.asarray(points[k], dtype=np.float64)
                   for k in ("far_left", "far_right")]
        return cls(a, b, np.asarray(data["homography_px_to_m"], dtype=np.float64), source_sha256, *corners)

    @classmethod
    def load(cls, geometry_dir: Optional[str], stem: str) -> "CourtGeometry":
        if not geometry_dir:
            raise ValueError("slot_mode=netline needs preprocess.court_geometry_dir")
        path = Path(geometry_dir) / f"{stem}.json"
        raw = path.read_bytes()
        try:
            return cls.from_dict(json.loads(raw), hashlib.sha256(raw).hexdigest())
        except ValueError as exc:
            raise ValueError(f"{exc}: {path}") from None

    def net_y_at(self, x: float) -> float:
        dx = self.net_b[0] - self.net_a[0]
        t = 0.0 if abs(dx) < 1e-9 else (x - self.net_a[0]) / dx
        return float(self.net_a[1] + t * (self.net_b[1] - self.net_a[1]))

    def below_net(self, box: np.ndarray) -> bool:
        """Box bottom lower on screen than the net line at the box's center x."""
        return float(box[3]) > self.net_y_at((float(box[0]) + float(box[2])) / 2)

    def feet_to_court(self, box: np.ndarray) -> tuple[float, float]:
        x, y = (float(box[0]) + float(box[2])) / 2, float(box[3])
        u, v, w = self.px_to_m @ np.array([x, y, 1.0])
        return float(u / w), float(v / w)


def feet_on_court(box: np.ndarray, court_mask: Optional[np.ndarray]) -> bool:
    """Feet (bottom-centre) inside the playable court (mask: 0 == on court).

    Deliberately feet, not centroid: the court trapezoid narrows with height, so
    a far player near a sideline can have their feet cleanly on court while their
    centroid falls outside the narrower band above them.
    """
    if court_mask is None:
        return True
    fx = int(np.clip((box[0] + box[2]) / 2, 0, court_mask.shape[1] - 1))
    fy = int(np.clip(box[3], 0, court_mask.shape[0] - 1))
    return bool(court_mask[int(fy), int(fx)] == 0)


def below_net_mask(boxes: np.ndarray, geometry: CourtGeometry) -> np.ndarray:
    return np.array([geometry.below_net(b) for b in boxes], dtype=bool)


def in_far_court(u: float, v: float) -> bool:
    """Feet (court meters) inside the far half-court plus the run-around margin."""
    return abs(u) <= 10.97 / 2 + FAR_COURT_MARGIN_U and FAR_COURT_MIN_V <= v <= 23.77 / 2 + FAR_COURT_RUNBACK_V


class StaticDetections:
    """Flags detections whose feet have stayed put for a long stretch of frames.

    A spectator, umpire or coach standing beside the far court yields a crop
    detection at the same spot frame after frame; a player in a point never does.
    Anchors are kept in pixels (court-meter depth is too foreshortened at the far
    baseline for a stable radius): a detection matches an anchor when its feet are
    within max(min_radius_px, radius_frac x box height). An anchor is static once it
    spans >= min_frames calls with hits in >= min_fill of them; anchors unseen for
    forget_frames calls are dropped. One instance per video (stateful).
    """

    def __init__(self, min_frames: int = 100, min_fill: float = 0.6, radius_frac: float = 0.3,
                 min_radius_px: float = 6.0, forget_frames: int = 50) -> None:
        self.min_frames, self.min_fill = int(min_frames), float(min_fill)
        self.radius_frac, self.min_radius_px = float(radius_frac), float(min_radius_px)
        self.forget_frames = int(forget_frames)
        self.anchors: list[list[float]] = []  # [x, y, first, last, hits]
        self.frame = -1

    def update(self, boxes: np.ndarray) -> np.ndarray:
        self.frame += 1
        f = self.frame
        flags = np.zeros(len(boxes), dtype=bool)
        for i, b in enumerate(np.asarray(boxes, dtype=np.float64).reshape(-1, 4)):
            x, y = (b[0] + b[2]) / 2, b[3]
            radius = max(self.min_radius_px, self.radius_frac * (b[3] - b[1]))
            best, best_d = None, radius
            for a in self.anchors:
                d = max(abs(a[0] - x), abs(a[1] - y))
                if d <= best_d:
                    best, best_d = a, d
            if best is None:
                self.anchors.append([x, y, f, f, 1])
                continue
            if best[3] != f:
                best[3], best[4] = f, best[4] + 1
            span = f - best[2] + 1
            flags[i] = span >= self.min_frames and best[4] >= self.min_fill * span
        self.anchors = [a for a in self.anchors if f - a[3] <= self.forget_frames]
        return flags


def new_far_stats() -> dict:
    return {"frames": 0, "picked": 0, "outside_court": 0, "static_picked": 0, "static_rejected": 0}


def pick_far(
    boxes: np.ndarray,
    court_mask: Optional[np.ndarray],
    geometry: CourtGeometry,
    static: Optional[np.ndarray] = None,
    stats: Optional[dict] = None,
) -> Optional[int]:
    """Crop detection most likely to be the far singles player, or None.

    Feet on court and above the net line; then gated to the far half in court
    meters; closest to the center line wins, with candidates within FAR_TIE_U
    of the best broken by closeness to the net. Meters, not pixels: pixel
    offsets shrink with depth, so a spectator behind the fence would look
    closer to center than a player one meter off it.

    Within the gate, candidates are tiered: feet inside the far half-court
    (in_far_court) and moving, then inside but frame-static (``static``, from a
    per-video StaticDetections), then outside the court but moving. A static
    candidate outside the court (a bystander) is never picked. ``stats``
    (new_far_stats) counts frames, picks, picks outside the court and static
    decisions.
    """
    if stats is not None:
        stats["frames"] += 1
    tiers: dict[int, list] = {}
    for i in range(len(boxes)):
        if not feet_on_court(boxes[i], court_mask) or geometry.below_net(boxes[i]):
            continue
        u, v = geometry.feet_to_court(boxes[i])
        if not (abs(u) < FAR_MAX_ABS_U and FAR_MIN_V < v < FAR_MAX_V):
            continue
        is_static = bool(static[i]) if static is not None else False
        inside = in_far_court(u, v)
        if is_static and not inside:
            if stats is not None:
                stats["static_rejected"] += 1
            continue
        tiers.setdefault(0 if inside and not is_static else (1 if inside else 2), []).append((abs(u), v, i))
    if not tiers:
        return None
    tier = min(tiers)
    scored = tiers[tier]
    best_u = min(s[0] for s in scored)
    pick = min((s for s in scored if s[0] - best_u <= FAR_TIE_U), key=lambda s: s[1])[2]
    if stats is not None:
        stats["picked"] += 1
        stats["outside_court"] += int(tier == 2)
        stats["static_picked"] += int(tier == 1)
    return pick
