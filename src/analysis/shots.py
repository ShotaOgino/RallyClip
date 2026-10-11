"""Shot events (swing moments) and a rule-based stroke type from 5 fps pose tracks.

Roadmap Phase 2, first cut, label-free. Input is the per-frame near/far player record
written by ``scripts/eval/dump_heatmap_tracks.py`` (reference pixels of the dump's
``ref_size``: 1280x720 for v1 artifacts, 1920x1080 for v0.6.0; every rule is in body
scales or box heights, so none depends on it; NaN where absent, COCO-17 keypoint
order). numpy only: no torch / cv2 / av, so this can later be called from the engine
without touching the torch-free runtime invariant.

Pipeline per match, then per point and player side
---------------------------------------------------
0. Once per match: ``match_handedness`` (hitting arm per camera side from the serve
   geometry, label-free, with a confidence and an override) and
   ``match_swing_reference`` (each side's median in-point swing peak, the reference of
   the swing-likelihood gate).
1. ``detect_swings``: wrist speed relative to the body (mid-shoulder), normalised by a
   rolling body scale (shoulder width, floored at half the torso length so a side-on
   torso does not inflate speeds), with a per-step left/right label-swap repair; the
   faster wrist by default (see ``detect_swings`` for why not the dominant one only);
   lightly smoothed, then peak-picked with a threshold
   ``max(abs_floor, median + rel_k * 1.4826 * MAD)`` over the player's own speeds in the
   window, a bounded prominence test, non-maximum suppression with a refractory period,
   a parabolic sub-sample peak time, and the swing-likelihood gate (peak >=
   ``swing_gate_rel`` x the match reference).
2. ``classify_stroke``: serve / overhead / forehand / backhand / (volley, off by
   default) / unknown from the pose around the contact time (rules below).
3. ``analyze_point_shots``: both sides, merged in time with the rally alternation prior
   (serve toss + contact pairs collapsed onto the contact, de-duplicate near-simultaneous
   opposite-side events, suppress same-side doubles, fill a missing opponent shot or
   serve as ``source="prior"``).

Left / right: camera geometry vs keypoint labels
------------------------------------------------
COCO "left_wrist" / "right_wrist" are meant to be the person's own sides. On the
beta-tennis rear camera YOLO mostly labels the near player (seen from behind) by image
side instead: the racket wrist is "right" while it is on the camera-right of the body and
becomes "left" when it crosses over (serve and forehand follow-through, two-handed
backhand), and YOLO often puts both wrist labels on one arm. On 2R the shoulders, hips
and ears are all ordered against the back-view convention in ~20 % of near frames (part
real rotation, e.g. the serve's shoulder turn, part whole-skeleton label flips). So by default (``ShotParams.lr_source = "image"``) no rule trusts the
labels for a side decision: sides come from image x and ``CAMERA_RIGHT_SIGN`` (near
player: camera-right = their right; far player, facing the camera: mirrored), and the
hand that matters is chosen by position (highest, or both hands' centroid), not by label.
``lr_source = "labels"`` keeps the earlier label-based rules for comparison.

Known limits (by design of a label-free baseline)
-------------------------------------------------
* 5 fps sees a swing in 2-3 samples: the peak time is good to ~0.1 s at best, contact
  pose is the nearest sample (up to 0.1 s away), short volleys / blocked returns can
  fall under the threshold, and a fast swing can alias.
* The far player is small (shoulder width ~10-20 px) and often missing; its keypoint
  noise is a large fraction of its body scale, so far events are both missed and
  hallucinated more often. The alternation prior compensates for missing ones and
  reports how many it added. Far handedness usually has no serve evidence and falls
  back to "right" with confidence 0.
* Handedness is per camera side; a side holds two different players across
  change-overs. Pass per-player handedness (``override``) once Phase 1 identity exists.
* Volleys need the ball (bounce or not), so the volley rule is off by default.
* A second serve after a fault inside one detected point is not recognised as a serve.
* Thresholds started as hand-set guesses; the 2026-10-12 changes (handedness, image-side
  forehand/backhand, serve pairs, volley off, swing gate) were checked against model
  pseudo-labels of one match (``labels/``), each with the physical rationale in its
  docstring. Not validated on a second match or a human-verified label set yet.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# COCO-17 keypoint indices.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
SHOULDER = {"left": 5, "right": 6}
ELBOW = {"left": 7, "right": 8}
WRIST = {"left": 9, "right": 10}
HIP = {"left": 11, "right": 12}
OTHER_ARM = {"left": "right", "right": "left"}
SIDES = ("near", "far")
# Camera geometry (beta-tennis / broadcast-style rear camera): the near player is seen
# from behind, so the player's own right is at larger image x (+1); the far player faces
# the camera, so the player's own right is at smaller image x (-1). Every image-side rule
# multiplies image x offsets by this sign to get "toward the player's own right".
CAMERA_RIGHT_SIGN = {"near": 1.0, "far": -1.0}
LR_SOURCES = ("image", "labels")
STROKES = ("serve", "forehand", "backhand", "volley", "overhead", "unknown")


@dataclass(frozen=True)
class ShotParams:
    """Every tunable of the baseline in one place (units noted per field).

    Speeds are in body scales per second (bs/s); one body scale ~ shoulder width
    (~0.4 m for a junior), so 10 bs/s ~ 4 m/s of wrist motion relative to the torso.
    """

    kp_conf_min: float = 0.3          # keypoint confidence below this counts as missing
    scale_window_sec: float = 1.0     # half-width of the rolling median body scale
    box_scale_ratio: float = 0.23     # body-scale fallback = ratio * box height
    elbow_gain: float = 1.6           # wrist-missing fallback: elbow speed * gain
    dominant_only: bool = False       # signal = hitting-arm wrist only (else max of both wrists)
    # Where the stroke rules get "left" / "right" from (see ``classify_stroke``):
    # "image"  - image x + CAMERA_RIGHT_SIGN; the COCO left/right keypoint labels are not
    #            trusted for side decisions (default; YOLO labels the near player's wrists
    #            by image side, so a racket arm that crosses the body changes label).
    # "labels" - trust the keypoint labels as the person's own sides (the pre-2026-10-12
    #            rules; kept for comparison and for footage where the labels are reliable).
    lr_source: str = "image"
    handedness_min_confidence: float = 0.3  # below this the estimate falls back to "right"
    max_jump_box_h: float = 1.0       # box centre moved > this * box height in one step -> identity jump
    max_box_h_ratio: float = 1.6      # box height changed by more than this factor -> identity jump
    max_gap_factor: float = 1.5       # samples further apart than this * nominal dt are not consecutive
    rel_k: float = 1.0                # threshold = median + rel_k * 1.4826 * MAD
    abs_floor: float = 6.0            # bs/s, absolute minimum peak speed
    min_samples_for_stats: int = 5    # fewer finite speeds -> threshold = 1.5 * abs_floor
    refractory_sec: float = 0.6       # no two swings by one player closer than this
    prominence_window_sec: float = 1.2  # look this far each side for the valley around a peak
    prominence_frac: float = 0.3      # peak must rise >= frac * peak above the higher valley
    swing_gate_rel: float = 0.65      # drop peaks < this * the side's median in-point swing peak
                                      # (needs a match-level reference; 0 = off)
    swing_ref_min_events: int = 10    # fewer swings in the match -> no reference, no gate
    # classify_stroke
    contact_max_dt: float = 0.25      # use samples within this of the peak time
    overhead_margin: float = 0.8      # wrist above head by > margin * body scale
    serve_window_sec: float = 2.5     # a serve is the first swing within this of point start
    serve_pre_sec: float = 0.5        # serve posture may be seen this long before the peak
    toss_post_sec: float = 0.5        # the tossing hand tops out up to this long after its speed peak
    serve_pair_sec: float = 1.5       # toss -> contact of one serve (2R: 0.8-1.25 s); the server's
                                      # next real shot is >= ~2 s later (2R: >= 2.3 s)
    side_margin: float = 0.15         # |wrist - midline| below margin * body scale -> unknown
    fbh_prep_sec: float = 0.4         # image FH/BH rule: hands' side over [t - this, t]
    volley_enabled: bool = False      # volley rule off until there is ball evidence (see classify_stroke)
    volley_depth_box_h: float = 0.5   # near player's feet this many box heights in front of their baseline
    volley_short_sec: float = 0.4     # swing above half-peak no longer than this
    # analyze_point_shots
    pre_pad_sec: float = 0.4          # frames read before the point start (speed context)
    opp_dedupe_sec: float = 0.4       # opposite-side events closer than this: keep the stronger
    same_side_min_gap_sec: float = 1.5  # same-side consecutive events closer: drop the weaker;
                                        # further apart: insert a prior opponent shot between
    serve_prior_min_lead_sec: float = 0.8   # first event this long after start allows a prior serve
    serve_prior_lead_sec: float = 1.2       # prior serve placed this long before the first event


DEFAULT_PARAMS = ShotParams()


@dataclass
class SwingEvent:
    """One detected (or prior-inferred) swing.

    ``t`` seconds (same clock as the tracks' ``timestamps``); ``peak_speed`` in body
    scales/s (0 for prior events); ``arm`` is the wrist that carried the peak;
    ``confidence`` is a heuristic 0-1 score, not a calibrated probability.
    """

    t: float
    player_side: str
    peak_speed: float
    arm: str
    confidence: float
    source: str = "motion"            # "motion" | "prior"
    frame: int = -1                   # nearest sample index in the arrays it was detected on
    width_sec: float = 0.0            # contiguous time with raw speed >= half the peak

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PoseWindow:
    """One player's pose arrays (any contiguous slice of a track dump)."""

    kp: np.ndarray                    # (n, 17, 2)
    kpconf: Optional[np.ndarray]      # (n, 17) or None
    present: np.ndarray               # (n,) bool
    timestamps: np.ndarray            # (n,) seconds
    box: Optional[np.ndarray] = None  # (n, 4) x1, y1, x2, y2
    side: str = "near"
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- helpers


def _nanmedian(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _kp_xy(kp, kpconf, present, j: int, min_conf: float) -> np.ndarray:
    """(n, 2) keypoint j, NaN where absent / low confidence / (0, 0) placeholder."""
    xy = np.asarray(kp[:, j, :], dtype=np.float64)
    ok = np.asarray(present, dtype=bool) & np.isfinite(xy).all(axis=1)
    ok &= ~((xy[:, 0] == 0) & (xy[:, 1] == 0))
    if kpconf is not None:
        c = np.asarray(kpconf[:, j], dtype=np.float64)
        ok &= ~(np.isfinite(c) & (c < min_conf))  # NaN conf (not recorded) counts as ok
    out = np.full_like(xy, np.nan)
    out[ok] = xy[ok]
    return out


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt((v ** 2).sum(axis=-1))


def _rolling_nanmedian(x: np.ndarray, half: int) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x.copy()
    pad = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    return _nanmedian(win, axis=1)


def _body_scale(kp, kpconf, present, box, timestamps, p: ShotParams) -> np.ndarray:
    """Per-frame body scale in pixels: rolling median of max(shoulder width, torso/2)."""
    c = p.kp_conf_min
    ls, rs = _kp_xy(kp, kpconf, present, SHOULDER["left"], c), _kp_xy(kp, kpconf, present, SHOULDER["right"], c)
    lh, rh = _kp_xy(kp, kpconf, present, HIP["left"], c), _kp_xy(kp, kpconf, present, HIP["right"], c)
    sw = _norm(ls - rs)
    torso = _norm(_mid(ls, rs) - _mid(lh, rh))
    raw = np.fmax(sw, 0.5 * torso)
    if box is not None:
        bh = np.asarray(box[:, 3] - box[:, 1], dtype=np.float64)
        fb = np.where(np.asarray(present, bool) & np.isfinite(bh) & (bh > 0), p.box_scale_ratio * bh, np.nan)
        raw = np.where(np.isfinite(raw), raw, fb)
    raw = np.where(raw > 1.0, raw, np.nan)
    dt = _nominal_dt(timestamps)
    half = max(1, int(round(p.scale_window_sec / dt)))
    sc = _rolling_nanmedian(raw, half)
    glob = _nanmedian(raw) if np.isfinite(raw).any() else np.nan
    return np.where(np.isfinite(sc), sc, glob)


def _nominal_dt(timestamps) -> float:
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) < 2:
        return 0.2
    d = np.diff(ts)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.2


def _reference(kp, kpconf, present, box, p: ShotParams) -> Tuple[np.ndarray, np.ndarray]:
    """Body reference point (mid-shoulder, else a box-derived proxy) and its type (1/2, 0 = none)."""
    c = p.kp_conf_min
    shc = _mid(_kp_xy(kp, kpconf, present, SHOULDER["left"], c), _kp_xy(kp, kpconf, present, SHOULDER["right"], c))
    ref = shc.copy()
    rtype = np.where(np.isfinite(shc).all(axis=1), 1, 0)
    if box is not None:
        b = np.asarray(box, dtype=np.float64)
        proxy = np.stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 1] + 0.22 * (b[:, 3] - b[:, 1])], axis=1)
        use = (rtype == 0) & np.asarray(present, bool) & np.isfinite(proxy).all(axis=1)
        ref[use] = proxy[use]
        rtype = np.where(use, 2, rtype)
    return ref, rtype


def _step_valid(present, box, timestamps, rtype, p: ShotParams) -> np.ndarray:
    """(n-1,) True where samples i-1 -> i are consecutive, same identity, same reference type."""
    ts = np.asarray(timestamps, dtype=np.float64)
    pres = np.asarray(present, dtype=bool)
    ok = pres[1:] & pres[:-1]
    ok &= np.diff(ts) <= p.max_gap_factor * _nominal_dt(ts)
    ok &= (rtype[1:] == rtype[:-1]) & (rtype[1:] > 0)
    if box is not None:
        b = np.asarray(box, dtype=np.float64)
        cen = np.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0], axis=1)
        h = b[:, 3] - b[:, 1]
        jump = _norm(np.diff(cen, axis=0))
        hm = _mid(h[1:], h[:-1])
        ratio = np.fmax(h[1:], h[:-1]) / np.fmax(np.fmin(h[1:], h[:-1]), 1e-6)
        with np.errstate(invalid="ignore"):
            bad = (jump > p.max_jump_box_h * hm) | (ratio > p.max_box_h_ratio)
        ok &= ~np.nan_to_num(bad, nan=False).astype(bool)
    return ok


def _crossed_cheaper(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per step: (both pairings measurable, crossed pairing < half the labelled cost)."""
    same = _norm(a[1:] - a[:-1]) + _norm(b[1:] - b[:-1])
    cross = _norm(a[1:] - b[:-1]) + _norm(b[1:] - a[:-1])
    fin = np.isfinite(same) & np.isfinite(cross)
    with np.errstate(invalid="ignore"):
        return fin, fin & (2.0 * cross < same)


def arm_speeds(kp, kpconf, present, timestamps, box=None, params: ShotParams = DEFAULT_PARAMS) -> Dict[str, np.ndarray]:
    """Per-step wrist speed relative to the body, in body scales per second.

    Returns ``tm`` (n-1,) step midpoint times, ``left`` / ``right`` (n-1,) speeds
    (NaN where the step is not measurable), ``scale`` (n,) body scale in pixels.
    """
    p = params
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(ts)
    out: Dict[str, np.ndarray] = {"tm": (ts[1:] + ts[:-1]) / 2.0 if n > 1 else np.zeros(0)}
    scale = _body_scale(kp, kpconf, present, box, ts, p) if n else np.zeros(0)
    out["scale"] = scale
    if n < 2:
        out["left"] = out["right"] = np.zeros(0)
        return out
    ref, rtype = _reference(kp, kpconf, present, box, p)
    ok = _step_valid(present, box, ts, rtype, p)
    dt = np.diff(ts)
    sc = _mid(scale[1:], scale[:-1])
    rel = {}
    for arm in ("left", "right"):
        rel[arm] = (_kp_xy(kp, kpconf, present, WRIST[arm], p.kp_conf_min) - ref,
                    _kp_xy(kp, kpconf, present, ELBOW[arm], p.kp_conf_min) - ref)
    # Per-step L/R swap repair: YOLO occasionally swaps the left/right labels for one
    # frame (~2 % of steps with the wrists apart on beta-tennis 2R), which reads as both
    # wrists teleporting. If the crossed pairing is less than half the cost of the
    # labelled one, the step is measured with the crossed pairing.
    fin_w, sw_w = _crossed_cheaper(rel["left"][0], rel["right"][0])
    fin_e, sw_e = _crossed_cheaper(rel["left"][1], rel["right"][1])
    swapped = np.where(fin_w, sw_w, fin_e & sw_e)  # elbows decide only where wrists can't
    out["swapped"] = swapped
    for arm in ("left", "right"):
        w, e = rel[arm]
        wo, eo = rel[OTHER_ARM[arm]]
        prev_w = np.where(swapped[:, None], wo[:-1], w[:-1])
        prev_e = np.where(swapped[:, None], eo[:-1], e[:-1])
        dw = _norm(w[1:] - prev_w)
        de = _norm(e[1:] - prev_e)
        d = np.where(np.isfinite(dw), dw, p.elbow_gain * de)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = d / (dt * sc)
        out[arm] = np.where(ok & np.isfinite(v), v, np.nan)
    return out


def _head_xy(kp, kpconf, present, scale, p: ShotParams) -> np.ndarray:
    """(n, 2) head point: nose, else the mean of eyes/ears, else mid-shoulder (y lifted by
    0.6 body scale, as in ``_head_y``). NaN where none is measurable."""
    c = p.kp_conf_min
    nose = _kp_xy(kp, kpconf, present, NOSE, c)
    face = np.stack([_kp_xy(kp, kpconf, present, j, c) for j in (L_EYE, R_EYE, L_EAR, R_EAR)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        face = np.nanmean(face, axis=0)
    sh = _mid(_kp_xy(kp, kpconf, present, SHOULDER["left"], c), _kp_xy(kp, kpconf, present, SHOULDER["right"], c))
    sh = sh - np.stack([np.zeros(len(sh)), 0.6 * np.asarray(scale, dtype=np.float64)], axis=1)
    out = np.where(np.isfinite(nose).all(axis=1)[:, None], nose, face)
    return np.where(np.isfinite(out).all(axis=1)[:, None], out, sh)


def handedness_evidence(
    kp,
    kpconf,
    present,
    timestamps,
    box=None,
    params: ShotParams = DEFAULT_PARAMS,
    *,
    side: str = "near",
    windows: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Hitting arm from the serve geometry, label-free. See ``estimate_handedness``.

    Returns {"arm", "confidence", "votes_right", "votes_left", "abstained", "episodes",
    "source"} with source "serve" (votes decided it) or "fallback" ("right", too little or
    too inconsistent evidence).
    """
    p = params
    ts = np.asarray(timestamps, dtype=np.float64)
    kp = np.asarray(kp, dtype=np.float64)
    kpconf = None if kpconf is None else np.asarray(kpconf, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    box = None if box is None else np.asarray(box, dtype=np.float64)
    out = {"arm": "right", "confidence": 0.0, "votes_right": 0, "votes_left": 0, "abstained": 0, "episodes": 0,
           "source": "fallback"}
    if len(ts) == 0:
        return out
    scale = _body_scale(kp, kpconf, present, box, ts, p)
    head = _head_xy(kp, kpconf, present, scale, p)
    wr = np.stack([_kp_xy(kp, kpconf, present, WRIST[a], p.kp_conf_min) for a in ("left", "right")])  # (2, n, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = (head[None, :, 1] - wr[:, :, 1]) / scale[None, :]          # wrist height above the head, bs
    up_f = np.where(np.isfinite(up), up, -np.inf)
    top = np.argmax(up_f, axis=0)                                      # highest wrist, whatever its label
    up_max = up_f[top, np.arange(len(ts))]
    with np.errstate(invalid="ignore", divide="ignore"):
        lateral = (wr[top, np.arange(len(ts)), 0] - head[:, 0]) / scale * CAMERA_RIGHT_SIGN.get(side, 1.0)
    over = (up_max > p.overhead_margin) & np.isfinite(lateral)
    if windows is not None:
        episodes = [np.where(over & (ts >= lo) & (ts <= hi))[0] for lo, hi in windows]
    else:
        idx = np.where(over)[0]
        cuts = np.where(np.diff(ts[idx]) > p.serve_window_sec)[0] + 1 if len(idx) else []
        episodes = np.split(idx, cuts) if len(idx) else []
    votes = {"right": 0, "left": 0}
    for ep in episodes:
        if len(ep) == 0:
            continue
        out["episodes"] += 1
        v = float(np.median(lateral[ep]))
        if abs(v) < p.side_margin:
            out["abstained"] += 1
        else:
            votes["right" if v > 0 else "left"] += 1
    r, lft = votes["right"], votes["left"]
    conf = abs(r - lft) / (r + lft + 2.0)  # Laplace-shrunk agreement: 32 vs 0 -> 0.94, 2 vs 0 -> 0.5
    out.update(votes_right=r, votes_left=lft, confidence=round(conf, 3))
    if r + lft > 0 and conf >= p.handedness_min_confidence:
        out.update(arm="right" if r >= lft else "left", source="serve")
    return out


def estimate_handedness(
    kp,
    kpconf,
    present,
    timestamps,
    box=None,
    params: ShotParams = DEFAULT_PARAMS,
    *,
    side: str = "near",
    windows: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[str, float]:
    """Hitting arm from where the hands go overhead, mostly on serves. Returns (arm, confidence).

    Physics: a server tosses the ball up in front of and to the hitting-arm side of the
    head, and strikes it above the hitting shoulder; overheads are struck the same way.
    So whenever a hand is clearly above the head (``overhead_margin`` body scales: the
    toss arm at the top of the toss, then the racket arm at contact), that highest hand
    is on the hitting-arm side of the head. The highest hand is picked by image height,
    not by its keypoint label, and its side is the sign of its image x offset from the
    head, turned into the player's own side with ``CAMERA_RIGHT_SIGN`` (near player seen
    from behind: camera-right = their right; far player facing the camera: mirrored).

    Each episode votes with the median offset over its overhead frames; an offset under
    ``side_margin`` body scales abstains. Episodes are the given ``windows`` (pass the
    serve windows ``[start - pre_pad_sec, start + serve_window_sec]`` of the points when
    they are known), else runs of overhead frames split at gaps > ``serve_window_sec``.
    ``confidence = |right - left| / (votes + 2)``; below ``handedness_min_confidence``, or
    with no vote at all, the answer is the population prior "right" (about 9 in 10
    players) with that low confidence. Pass an override instead of trusting it whenever
    the hand is known (``match_handedness(override=...)``).

    Why not wrist speed (the pre-2026-10-12 estimator): on the beta-tennis rear camera
    YOLO labels the near player's wrists mostly by image side, so the racket wrist is
    labelled "left" whenever it crosses the body (serve and forehand follow-through), and
    the free arm moves fast too (toss, two-handed backhand). On 2R the 95th-percentile
    speeds came out left 8.55 / right 8.15 bs/s and the estimate said "left" for two
    right-handers. The serve geometry voted right in 32 of 32 near-served points.
    """
    ev = handedness_evidence(kp, kpconf, present, timestamps, box, params, side=side, windows=windows)
    return ev["arm"], float(ev["confidence"])


def _smooth121(x: np.ndarray) -> np.ndarray:
    """NaN-aware [1, 2, 1] / 4 smoothing; stays NaN where the centre is NaN."""
    n = len(x)
    if n < 3:
        return x.copy()
    w = np.array([1.0, 2.0, 1.0])
    pad = np.concatenate([[np.nan], x, [np.nan]])
    win = np.lib.stride_tricks.sliding_window_view(pad, 3)
    fin = np.isfinite(win)
    num = np.where(fin, win, 0.0) @ w
    den = fin.astype(np.float64) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        y = num / den
    return np.where(np.isfinite(x), y, np.nan)


def speed_threshold(values: np.ndarray, params: ShotParams = DEFAULT_PARAMS) -> float:
    """``max(abs_floor, median + rel_k * 1.4826 * MAD)`` over the finite values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < params.min_samples_for_stats:
        return 1.5 * params.abs_floor
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return max(params.abs_floor, med + params.rel_k * 1.4826 * mad)


def _prominent(s: np.ndarray, tm: np.ndarray, i: int, p: ShotParams) -> bool:
    """Topographic prominence test, bounded in time: walk each way from the peak until a
    higher sample or ``prominence_window_sec``; the valley on that side is the lowest
    finite sample passed. The peak must rise >= ``prominence_frac`` * peak above the
    higher valley (one-sided if a side has no finite samples). Rejects ripples on a
    running plateau and the lesser bump of a double-humped swing."""
    valleys = []
    for step in (-1, 1):
        j, lo = i + step, np.inf
        while 0 <= j < len(s) and abs(tm[j] - tm[i]) <= p.prominence_window_sec + 1e-9:
            if np.isfinite(s[j]):
                if s[j] > s[i]:
                    break
                lo = min(lo, float(s[j]))
            j += step
        if np.isfinite(lo):
            valleys.append(lo)
    if not valleys:
        return True
    return s[i] - max(valleys) >= p.prominence_frac * s[i]


def _parabolic(y_m: float, y_0: float, y_p: float) -> Tuple[float, float]:
    """Vertex offset (in samples, clipped to +-0.5) and height of the parabola through 3 points."""
    if not (np.isfinite(y_m) and np.isfinite(y_p)):
        return 0.0, y_0
    den = y_m - 2.0 * y_0 + y_p
    if den >= 0:  # not a strict maximum
        return 0.0, y_0
    d = float(np.clip(0.5 * (y_m - y_p) / den, -0.5, 0.5))
    return d, float(y_0 - 0.25 * (y_m - y_p) * d)


# ------------------------------------------------------------------ detection


def detect_swings(
    kp,
    kpconf,
    present,
    timestamps,
    side: str = "near",
    *,
    box=None,
    handedness: Optional[str] = None,
    speed_ref: Optional[float] = None,
    params: ShotParams = DEFAULT_PARAMS,
) -> List[SwingEvent]:
    """Swing peaks of one player in one window (normally one point).

    Signal: wrist speed relative to the mid-shoulder point, in body scales/s, with a
    per-step L/R swap repair (``arm_speeds``). By default the signal is the faster of
    the two wrists: on the beta-tennis rear camera the near player's shoulder width is
    ~15-30 px, YOLO's left/right wrist labels make the two wrists' speed distributions
    nearly identical (95th-percentile ratio 1.06 on 2R), and a "near" side holds two
    different players across change-overs, so a single dominant arm is not reliable.
    ``params.dominant_only=True`` restores the hitting-arm-only signal (other wrist as a
    fallback where the hitting wrist is unmeasurable) for when per-player handedness is
    known (Phase 1 identity).

    The signal is smoothed [1, 2, 1]/4 and thresholded per window (``speed_threshold``);
    a candidate is a local maximum that also rises ``prominence_frac`` above the
    surrounding valleys (``_prominent``; rejects ripples on a running plateau); kept by
    non-maximum suppression with ``refractory_sec``; the peak time is the vertex of a
    parabola through the smoothed peak and its neighbours.

    ``handedness`` should come from ``estimate_handedness`` over the whole match; if
    None it is estimated from this window (unreliable for short windows).

    Swing-likelihood gate: with ``speed_ref`` (the side's median in-point swing peak over
    the match, ``match_swing_reference``), peaks below ``swing_gate_rel * speed_ref`` are
    dropped. Physics: a stroke is the fastest arm motion a player makes, and one player's
    strokes are similar in speed through a match, so a peak far below their own typical
    stroke is something else (walking with the racket, bouncing the ball, towel, catching
    a ball). The per-window threshold alone cannot see this: in a quiet window it adapts
    down to whatever moves. Relative, not absolute, so it holds for slower or faster
    players and any body-scale error. 0.65 is the largest value that keeps >= 95 % of the
    labelled in-point swings on the 2R pseudo-labels (no hold-out: one match).
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    p = params
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) < 3:
        return []
    kp = np.asarray(kp, dtype=np.float64)
    kpconf = None if kpconf is None else np.asarray(kpconf, dtype=np.float64)
    box = None if box is None else np.asarray(box, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    if handedness is None:
        handedness, _ = estimate_handedness(kp, kpconf, present, ts, box, p, side=side)
    sp = arm_speeds(kp, kpconf, present, ts, box, p)
    dom, off = sp[handedness], sp[OTHER_ARM[handedness]]
    if p.dominant_only:
        raw = np.where(np.isfinite(dom), dom, off)
    else:
        raw = np.fmax(dom, off)
    s = _smooth121(raw)
    thr = speed_threshold(s, p)
    tm = sp["tm"]
    m = len(s)
    neg = np.where(np.isfinite(s), s, -np.inf)
    cand = []
    for i in range(m):
        if not np.isfinite(s[i]) or s[i] < thr:
            continue
        left = neg[i - 1] if i > 0 else -np.inf
        right = neg[i + 1] if i + 1 < m else -np.inf
        if s[i] >= left and s[i] > right and _prominent(s, tm, i, p):
            cand.append(i)
    cand.sort(key=lambda i: -s[i])
    accepted: List[SwingEvent] = []
    for i in cand:
        ym = s[i - 1] if i > 0 else np.nan
        yp = s[i + 1] if i + 1 < m else np.nan
        d, peak = _parabolic(ym, s[i], yp)
        step = (tm[i + 1] - tm[i]) if (d > 0 and i + 1 < m) else (tm[i] - tm[i - 1]) if i > 0 else _nominal_dt(ts)
        t = float(tm[i] + d * step)
        if any(abs(t - e.t) < p.refractory_sec for e in accepted):
            continue
        # arm whose wrist carried the signal at this step
        if p.dominant_only:
            arm = handedness if np.isfinite(dom[i]) else OTHER_ARM[handedness]
        else:
            arm = handedness if not np.isfinite(off[i]) or (np.isfinite(dom[i]) and dom[i] >= off[i]) else OTHER_ARM[handedness]
        # width: contiguous raw samples at >= half the raw peak around i
        rp = raw[i]
        lo = hi = i
        while lo - 1 >= 0 and np.isfinite(raw[lo - 1]) and raw[lo - 1] >= 0.5 * rp:
            lo -= 1
        while hi + 1 < m and np.isfinite(raw[hi + 1]) and raw[hi + 1] >= 0.5 * rp:
            hi += 1
        width = round(float((hi - lo + 1) * _nominal_dt(ts)), 6)
        # confidence: margin over threshold x keypoint quality of the carrying wrist
        if kpconf is not None:
            q = _nanmedian(kpconf[i:i + 2, WRIST[arm]])
            q = 0.5 if not np.isfinite(q) else float(np.clip(q, 0.0, 1.0))
        else:
            q = 0.5
        margin = float(np.clip((peak / thr - 1.0) / 1.5, 0.0, 1.0))
        conf = float(np.clip(0.5 * margin + 0.5 * q, 0.0, 1.0))
        frame = int(np.argmin(np.abs(ts - t)))
        accepted.append(SwingEvent(t=t, player_side=side, peak_speed=float(peak), arm=arm,
                                   confidence=conf, source="motion", frame=frame, width_sec=width))
    if speed_ref is not None and np.isfinite(speed_ref) and speed_ref > 0 and p.swing_gate_rel > 0:
        accepted = [e for e in accepted if e.peak_speed >= p.swing_gate_rel * speed_ref]
    accepted.sort(key=lambda e: e.t)
    return accepted


# ------------------------------------------------------------- classification


def _head_y(kp, kpconf, present, i: int, scale: float, p: ShotParams) -> float:
    c = p.kp_conf_min
    sl = slice(i, i + 1)
    nose = _kp_xy(kp[sl], None if kpconf is None else kpconf[sl], present[sl], NOSE, c)[0]
    if np.isfinite(nose[1]):
        return float(nose[1])
    ys = [_kp_xy(kp[sl], None if kpconf is None else kpconf[sl], present[sl], j, c)[0][1] for j in (L_EYE, R_EYE, L_EAR, R_EAR)]
    ys = [y for y in ys if np.isfinite(y)]
    if ys:
        return float(np.mean(ys))
    sh = [_kp_xy(kp[sl], None if kpconf is None else kpconf[sl], present[sl], SHOULDER[a], c)[0][1] for a in ("left", "right")]
    sh = [y for y in sh if np.isfinite(y)]
    if sh and np.isfinite(scale):
        return float(np.mean(sh) - 0.6 * scale)
    return float("nan")


def _orientation(kp, kpconf, present, arm: str, side: str, p: ShotParams) -> float:
    """+1 if the hitting arm's shoulder sits at larger image x than the other shoulder.

    Window median of the labelled shoulder offset (stable through rotation at contact);
    falls back to the camera prior (near player seen from behind: right shoulder at
    larger x; far player seen from the front: the opposite).
    """
    hs = _kp_xy(kp, kpconf, present, SHOULDER[arm], p.kp_conf_min)[:, 0]
    os_ = _kp_xy(kp, kpconf, present, SHOULDER[OTHER_ARM[arm]], p.kp_conf_min)[:, 0]
    d = _nanmedian(hs - os_) if np.isfinite(hs - os_).any() else np.nan
    if np.isfinite(d) and d != 0:
        return float(np.sign(d))
    prior = 1.0 if side == "near" else -1.0
    return prior if arm == "right" else -prior


def serve_posture(event: SwingEvent, window: PoseWindow, params: ShotParams = DEFAULT_PARAMS,
                  post_sec: Optional[float] = None) -> float:
    """Highest hand above the head (body scales; either wrist, whatever its label) over the
    samples in ``[t - serve_pre_sec, t + post_sec]`` (default ``contact_max_dt``); -inf
    when not measurable. A value above ``overhead_margin`` is the serve / overhead
    posture (toss arm up, or racket arm up at contact)."""
    p = params
    post = p.contact_max_dt if post_sec is None else post_sec
    ts = np.asarray(window.timestamps, dtype=np.float64)
    pres = np.asarray(window.present, dtype=bool)
    sel = np.where(pres & (ts >= event.t - p.serve_pre_sec - 1e-9) & (ts <= event.t + post + 1e-9))[0]
    if len(sel) == 0:
        return float("-inf")
    kp = np.asarray(window.kp, dtype=np.float64)
    kc = None if window.kpconf is None else np.asarray(window.kpconf, dtype=np.float64)
    box = None if window.box is None else np.asarray(window.box, dtype=np.float64)
    scale = _body_scale(kp, kc, pres, box, ts, p)
    head = _head_xy(kp, kc, pres, scale, p)
    best = float("-inf")
    for a in ("left", "right"):
        wy = _kp_xy(kp, kc, pres, WRIST[a], p.kp_conf_min)[sel, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            up = (head[sel, 1] - wy) / scale[sel]
        if np.isfinite(up).any():
            best = max(best, float(np.nanmax(up)))
    return best


def collapse_serve_pairs(events: Sequence[SwingEvent], windows: Mapping[str, PoseWindow], point_start: float,
                         params: ShotParams = DEFAULT_PARAMS, server: Optional[str] = None
                         ) -> Tuple[List[SwingEvent], Dict[int, float], List[SwingEvent]]:
    """Drop the toss when one serve was detected twice (toss, then contact).

    Physics: the toss arm rises fast and goes overhead about 1 s before the racket strikes
    the ball, so a serve often reads as two same-side swing peaks. The ball has to be in
    the air before it can be struck, so of such a pair the *later* peak is the contact,
    whatever their speeds (the generic same-side suppression keeps the faster one, and on
    beta-tennis 2R that was the toss in 6 of 15 pairs). Toss to contact took 0.8-1.25 s on
    2R; the server's next real shot cannot come within ``serve_pair_sec`` of the serve
    (the ball has to reach the receiver and come back: >= 2.3 s on 2R), so the later peak
    needs no posture of its own (at 5 fps the racket is overhead for about one sample and
    is often missed).

    Rule, per side that may serve (``server`` None or equal): its first motion event is
    within ``serve_window_sec`` of ``point_start`` and shows the serve posture
    (``serve_posture`` > ``overhead_margin``, looking up to ``toss_post_sec`` after the
    peak: the toss arm's speed peaks while it rises, the hand tops out 0.1-0.4 s later),
    and its next event is within
    ``serve_pair_sec`` -> the first is dropped (it is the toss of that serve). Returns
    (events, {id(kept contact event): toss posture height}, [dropped toss events]).
    """
    p = params
    drop, contacts = set(), {}
    for side in SIDES:
        if server is not None and server != side:
            continue
        own = [e for e in events if e.player_side == side and e.source == "motion"]
        if len(own) < 2 or side not in windows:
            continue
        e1, e2 = own[0], own[1]
        if e1.t - point_start > p.serve_window_sec or e2.t - e1.t > p.serve_pair_sec:
            continue
        up = serve_posture(e1, windows[side], p, post_sec=p.toss_post_sec)
        if up > p.overhead_margin:
            drop.add(id(e1))
            contacts[id(e2)] = up
    return [e for e in events if id(e) not in drop], contacts, [e for e in events if id(e) in drop]


def _forehand_backhand_image(event: SwingEvent, kp, kpconf, present, ts, scale, side: str, hand: str,
                             p: ShotParams) -> Dict[str, Any]:
    """Forehand / backhand from where the hands are while the stroke is prepared and hit.

    Physics: a groundstroke is prepared and struck on one side of the body (the racket
    side for a forehand, the other side for a backhand, one- or two-handed) and only the
    follow-through, after the speed peak, crosses the body. So the side is read from the
    samples in ``[t - fbh_prep_sec, t]`` (back-swing to contact; ``t`` is the speed peak,
    ~contact at 5 fps): the mean lateral offset of the hands (the centroid of whichever
    wrists are measurable, so a label that jumps between the two wrists does not matter)
    from the body midline (mean of shoulder-centre and hip-centre x), in body scales,
    turned into the player's own side with ``CAMERA_RIGHT_SIGN`` and mirrored for a
    left-hander. Positive -> forehand, negative -> backhand, |mean| < ``side_margin`` ->
    unknown. Falls back to the first sample after the peak when no sample precedes it.

    Measured on the 2R pseudo-labels (78 labelled FH/BH events, hand = right): this rule
    0.846; the same centroid at the pre-peak sample only 0.795; labels-based rule with the
    right hand 0.69 of the FH/BH-predicted ones. Using the post-peak sample (follow-through)
    gives 0.36, i.e. the crossing is real.
    """
    c = p.kp_conf_min
    sel = np.where(present & (ts >= event.t - p.fbh_prep_sec - 1e-9) & (ts <= event.t + 1e-9))[0]
    if len(sel) == 0:
        sel = np.where(present & (ts > event.t) & (ts <= event.t + p.contact_max_dt))[0][:1]
    vals = []
    for i in sel:
        sl = slice(int(i), int(i) + 1)
        kc = None if kpconf is None else kpconf[sl]
        wx = [_kp_xy(kp[sl], kc, present[sl], WRIST[a], c)[0][0] for a in ("left", "right")]
        wx = [v for v in wx if np.isfinite(v)]
        shc = _mid(_kp_xy(kp[sl], kc, present[sl], SHOULDER["left"], c), _kp_xy(kp[sl], kc, present[sl], SHOULDER["right"], c))[0][0]
        hpc = _mid(_kp_xy(kp[sl], kc, present[sl], HIP["left"], c), _kp_xy(kp[sl], kc, present[sl], HIP["right"], c))[0][0]
        mids = [v for v in (shc, hpc) if np.isfinite(v)]
        s = scale[i]
        if wx and mids and np.isfinite(s) and s > 0:
            vals.append((float(np.mean(wx)) - float(np.mean(mids))) / s)
    if not vals:
        return {"stroke": "unknown", "confidence": 0.0}
    score = float(np.mean(vals)) * CAMERA_RIGHT_SIGN.get(side, 1.0) * (1.0 if hand == "right" else -1.0)
    if abs(score) < p.side_margin:
        return {"stroke": "unknown", "confidence": 0.0}
    conf = float(np.clip(abs(score) / (4.0 * p.side_margin), 0.2, 0.9))
    return {"stroke": "forehand" if score > 0 else "backhand", "confidence": conf}


def classify_stroke(
    event: SwingEvent,
    window: PoseWindow,
    *,
    point_start: Optional[float] = None,
    is_first_of_side: bool = False,
    handedness: Optional[str] = None,
    server: Optional[str] = None,
    baseline_foot_y: Optional[float] = None,
    params: ShotParams = DEFAULT_PARAMS,
) -> Dict[str, Any]:
    """Rule-based stroke type for one motion event. Returns {"stroke", "confidence"}.

    Rules, in precedence order (``s`` = body scale in px at contact; y grows downward):

    1. **serve**: the event is the player's first swing of the point, within
       ``serve_window_sec`` of ``point_start``, the side may serve (``server`` is None
       or equals the side), and some wrist is above the head (nose, else eyes/ears,
       else shoulders - 0.6 s) by > ``overhead_margin * s`` at a sample in
       [t - serve_pre_sec, t + contact_max_dt] (5 fps often catches the trophy/toss
       pose rather than contact). If ``server`` names this side, the first swing in the
       window is called a serve even without the posture (confidence 0.5).
       Limitation: a let / fault + second serve inside one detected point produces two
       serve-like swings; only the first is "serve", the second becomes "overhead".
    2. **overhead**: a hand is above the head by the same margin at a contact sample.
       ``lr_source="image"``: either hand (whatever its label), at the sample at/before
       the speed peak only (else the first after it): the hand over the head after the
       peak is a groundstroke follow-through (a topspin forehand finishes high over the
       other shoulder). On 2R, also reading the post-peak sample made 11 in-point
       overhead calls (9 forehands/backhands, 2 serves), pre-peak only 2 (both serves).
       ``lr_source="labels"``: the hitting wrist label, at the samples just before /
       after t (within ``contact_max_dt``). Limitation: few real overheads to check
       (one labelled on 2R, missed).
    3. **volley** (near player only, and only with ``volley_enabled``; off by default):
       the feet (box bottom) are more than
       ``volley_depth_box_h`` box heights above (in front of) the player's baseline
       position in the point (``baseline_foot_y``: 90th percentile of box bottom) and
       the swing is short (raw speed >= half-peak for <= ``volley_short_sec``).
       Far player: never volley (perspective compresses depth; too few pixels).
       Limitation: no ball, so a groundstroke taken inside the baseline after a short
       ball with a compact swing reads as a volley; a long swing at the net reads as
       forehand/backhand. Why off: a volley is defined by the ball not bouncing, which
       pose cannot see; on the 2R pseudo-labels the rule fired 23 times inside points
       (19 after the handedness / serve fixes, 13 of them on stroke-labelled events: 8
       forehands, 3 backhands, 2 serves) with precision 0 (no labelled volley at all:
       juniors rallying from the baseline), so it only cost accuracy. Re-enable once ball tracking (Phase 3) can confirm a no-bounce contact.
    4. **forehand / backhand**: hitting wrist x relative to the body midline (mean of
       shoulder-centre and hip-centre x) at the last sample at/before the peak (else the
       first one after it: the follow-through crosses the body), signed by the
       hitting arm's own shoulder side (``_orientation``). Same side -> forehand,
       crossed over -> backhand; |offset| < ``side_margin * s`` -> unknown.
       Limitation: at contact the wrist is often in front of the body (depth, invisible
       to a rear camera) so the lateral offset is small; per-frame L/R label swaps
       flip the answer.
    5. otherwise **unknown** (no measurable pose at contact).
    """
    p = params
    if event.source != "motion":
        return {"stroke": "unknown", "confidence": 0.0}
    w = window
    ts = np.asarray(w.timestamps, dtype=np.float64)
    kp = np.asarray(w.kp, dtype=np.float64)
    kpconf = None if w.kpconf is None else np.asarray(w.kpconf, dtype=np.float64)
    present = np.asarray(w.present, dtype=bool)
    box = None if w.box is None else np.asarray(w.box, dtype=np.float64)
    side = event.player_side
    arm = handedness or event.arm
    if len(ts) == 0:
        return {"stroke": "unknown", "confidence": 0.0}
    scale = _body_scale(kp, kpconf, present, box, ts, p)
    # Contact samples: the last sample at/before the peak time, then the first after it.
    # The pre-peak sample comes first because the post-peak one is the follow-through,
    # which crosses the body for forehands and backhands alike.
    near_t = np.abs(ts - event.t) <= p.contact_max_dt
    before = np.where(near_t & present & (ts <= event.t + 1e-9))[0]
    after = np.where(near_t & present & (ts > event.t + 1e-9))[0]
    contact = ([int(before[-1])] if len(before) else []) + ([int(after[0])] if len(after) else [])
    if not contact:
        return {"stroke": "unknown", "confidence": 0.0}

    def above_head(i: int, arms: Sequence[str]) -> float:
        s = scale[i]
        if not np.isfinite(s) or s <= 0:
            return float("-inf")
        hy = _head_y(kp, kpconf, present, i, s, p)
        if not np.isfinite(hy):
            return float("-inf")
        best = float("-inf")
        for a in arms:
            wy = _kp_xy(kp[i:i + 1], None if kpconf is None else kpconf[i:i + 1], present[i:i + 1], WRIST[a], p.kp_conf_min)[0][1]
            if np.isfinite(wy):
                best = max(best, (hy - wy) / s)
        return best

    image_lr = p.lr_source == "image"
    if p.lr_source not in LR_SOURCES:
        raise ValueError(f"lr_source must be one of {LR_SOURCES}, got {p.lr_source!r}")
    # image mode: the highest hand whatever its label (the racket wrist's label is not reliable)
    hit_arms = ("left", "right") if image_lr else (arm,)
    # image mode: only the sample at/before the speed peak (contact or just before). The
    # hand over the head *after* the peak is a groundstroke follow-through (a topspin
    # forehand finishes high over the other shoulder), not an overhead.
    oh_samples = [i for i in contact if ts[i] <= event.t + 1e-9] or contact[:1] if image_lr else contact
    hit_up = max(above_head(i, hit_arms) for i in oh_samples)
    serve_side_ok = server is None or server == side
    in_serve_window = point_start is not None and (event.t - point_start) <= p.serve_window_sec
    if is_first_of_side and in_serve_window and serve_side_ok:
        pre = [i for i in range(len(ts)) if event.t - p.serve_pre_sec <= ts[i] <= event.t + p.contact_max_dt and present[i]]
        any_up = max([above_head(i, ("left", "right")) for i in pre] or [float("-inf")])
        if any_up > p.overhead_margin:
            return {"stroke": "serve", "confidence": float(np.clip(0.6 + 0.4 * (any_up - p.overhead_margin), 0.0, 1.0))}
        if server == side:
            return {"stroke": "serve", "confidence": 0.5}
    if hit_up > p.overhead_margin:
        return {"stroke": "overhead", "confidence": float(np.clip(0.5 + 0.5 * (hit_up - p.overhead_margin), 0.0, 1.0))}

    i0 = contact[0]
    if p.volley_enabled and side == "near" and box is not None and np.isfinite(box[i0]).all():
        bh_all = box[:, 3] - box[:, 1]
        bh = _nanmedian(np.where(present, bh_all, np.nan))
        base = baseline_foot_y
        if base is None:
            feet = np.where(present, box[:, 3], np.nan)
            base = float(np.nanpercentile(feet, 90)) if np.isfinite(feet).any() else None
        if base is not None and np.isfinite(bh) and bh > 0:
            depth = (base - box[i0, 3]) / bh
            if depth > p.volley_depth_box_h and event.width_sec <= p.volley_short_sec + 1e-9:
                return {"stroke": "volley", "confidence": float(np.clip(0.4 + 0.6 * (depth - p.volley_depth_box_h), 0.0, 0.9))}

    c = p.kp_conf_min
    if image_lr:
        return _forehand_backhand_image(event, kp, kpconf, present, ts, scale, side, handedness or "right", p)
    for i in contact:
        sl = slice(i, i + 1)
        kc = None if kpconf is None else kpconf[sl]
        wx = _kp_xy(kp[sl], kc, present[sl], WRIST[arm], c)[0][0]
        shc = _mid(_kp_xy(kp[sl], kc, present[sl], SHOULDER["left"], c), _kp_xy(kp[sl], kc, present[sl], SHOULDER["right"], c))[0][0]
        hpc = _mid(_kp_xy(kp[sl], kc, present[sl], HIP["left"], c), _kp_xy(kp[sl], kc, present[sl], HIP["right"], c))[0][0]
        mids = [v for v in (shc, hpc) if np.isfinite(v)]
        s = scale[i]
        if not (np.isfinite(wx) and mids and np.isfinite(s) and s > 0):
            continue
        orient = _orientation(kp, kpconf, present, arm, side, p)
        score = (wx - float(np.mean(mids))) * orient / s
        if abs(score) < p.side_margin:
            return {"stroke": "unknown", "confidence": 0.0}
        conf = float(np.clip(abs(score) / (4.0 * p.side_margin), 0.2, 0.9))
        return {"stroke": "forehand" if score > 0 else "backhand", "confidence": conf}
    return {"stroke": "unknown", "confidence": 0.0}


# ------------------------------------------------------------------ per point


def _point_bounds(point) -> Tuple[float, float]:
    if isinstance(point, Mapping):
        return float(point["start"]), float(point["end"])
    s, e = point
    return float(s), float(e)


def _side_arrays(tracks: Mapping[str, Any], side: str):
    return (np.asarray(tracks[f"{side}_kp"], dtype=np.float64),
            np.asarray(tracks[f"{side}_kpconf"], dtype=np.float64) if f"{side}_kpconf" in tracks else None,
            np.asarray(tracks[f"{side}_present"], dtype=bool),
            np.asarray(tracks[f"{side}_box"], dtype=np.float64) if f"{side}_box" in tracks else None)


def serve_windows(points, params: ShotParams = DEFAULT_PARAMS) -> List[Tuple[float, float]]:
    """``[start - pre_pad_sec, start + serve_window_sec]`` per point (the serve's frames)."""
    return [(b[0] - params.pre_pad_sec, b[0] + params.serve_window_sec) for b in map(_point_bounds, points)]


def match_handedness_evidence(
    tracks: Mapping[str, Any],
    params: ShotParams = DEFAULT_PARAMS,
    *,
    points=None,
    override: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per side: ``handedness_evidence`` over the whole dump (call once per match).

    ``points`` (list of (start, end) or {"start", "end"}) restricts the evidence to the
    serve windows; without them every overhead episode in the dump votes. ``override``
    ({"near": "left"}, ...) wins for the sides it names (source "override", confidence 1).
    """
    ts = np.asarray(tracks["timestamps"], dtype=np.float64)
    win = serve_windows(points, params) if points is not None else None
    out = {}
    for side in SIDES:
        if override and override.get(side):
            arm = str(override[side]).lower()
            if arm not in OTHER_ARM:
                raise ValueError(f"handedness override for {side} must be 'left' or 'right', got {arm!r}")
            out[side] = {"arm": arm, "confidence": 1.0, "source": "override"}
            continue
        kp, kc, pres, box = _side_arrays(tracks, side)
        out[side] = handedness_evidence(kp, kc, pres, ts, box, params, side=side, windows=win)
    return out


def match_handedness(
    tracks: Mapping[str, Any],
    params: ShotParams = DEFAULT_PARAMS,
    *,
    points=None,
    override: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """{"near": arm, "far": arm}; see ``match_handedness_evidence``. One "near" side holds
    both players across change-overs, so this is per camera side; feed per-player
    handedness from Phase 1 identity through ``override`` / ``handedness`` once it exists."""
    ev = match_handedness_evidence(tracks, params, points=points, override=override)
    return {side: ev[side]["arm"] for side in SIDES}


def match_swing_reference(
    tracks: Mapping[str, Any],
    points,
    params: ShotParams = DEFAULT_PARAMS,
    *,
    handedness: Optional[Mapping[str, str]] = None,
) -> Dict[str, Optional[float]]:
    """{side: median peak speed (bs/s) of the ungated swings inside the points}, the
    reference of the swing-likelihood gate (see ``detect_swings``); None for a side with
    fewer than ``swing_ref_min_events`` swings. Call once per match with its points."""
    p = params
    ts = np.asarray(tracks["timestamps"], dtype=np.float64)
    hand = handedness or match_handedness(tracks, p, points=points)
    out: Dict[str, Optional[float]] = {}
    for side in SIDES:
        kp, kc, pres, box = _side_arrays(tracks, side)
        peaks = []
        for pt in points:
            s0, e0 = _point_bounds(pt)
            sel = np.where((ts >= s0 - p.pre_pad_sec) & (ts <= e0))[0]
            if len(sel) == 0:
                continue
            sl = slice(int(sel[0]), int(sel[-1]) + 1)
            evs = detect_swings(kp[sl], None if kc is None else kc[sl], pres[sl], ts[sl], side=side,
                                box=None if box is None else box[sl], handedness=hand.get(side), params=p)
            peaks += [e.peak_speed for e in evs if s0 - 0.2 <= e.t <= e0]
        out[side] = float(np.median(peaks)) if len(peaks) >= p.swing_ref_min_events else None
    return out


def analyze_point_shots(
    point,
    tracks: Mapping[str, Any],
    server: Optional[str] = None,
    *,
    handedness: Optional[Mapping[str, str]] = None,
    speed_ref: Optional[Mapping[str, Optional[float]]] = None,
    params: ShotParams = DEFAULT_PARAMS,
) -> Dict[str, Any]:
    """Shots of one point: both players' swings merged with the alternation prior.

    ``point`` is (start, end) seconds or {"start", "end"}; ``tracks`` is the dump
    mapping (npz or dict). ``server`` ("near"/"far") comes from Phase 1 when available.
    ``handedness`` defaults to ``match_handedness(tracks)`` (pass it in when looping
    over many points). ``speed_ref`` ({side: float or None}, from
    ``match_swing_reference``) turns on the swing-likelihood gate of ``detect_swings``;
    without it there is no gate (one point is too few swings for a reference).

    Merge rules (each counted in the result):
    * opposite-side events closer than ``opp_dedupe_sec``: keep the more confident
      (the opponent's split-step / preparation is not a shot) -> ``deduped``.
    * same-side consecutive events closer than ``same_side_min_gap_sec`` (the ball
      cannot cross the court and come back that fast): keep the faster -> ``suppressed``.
    * the first event is not a serve: a prior serve is inserted for the other side
      (or for ``server``) ``serve_prior_lead_sec`` before it if it comes at least
      ``serve_prior_min_lead_sec`` after the point start.
    * same-side consecutive events further apart: one prior shot by the opponent at the
      midpoint (the minimum that restores alternation).
    Prior events have ``source="prior"``, ``peak_speed`` 0, and are counted in
    ``prior_filled``.
    """
    p = params
    start, end = _point_bounds(point)
    ts_all = np.asarray(tracks["timestamps"], dtype=np.float64)
    if handedness is None:
        handedness = match_handedness(tracks, p)
    sel = np.where((ts_all >= start - p.pre_pad_sec) & (ts_all <= end))[0]
    events: List[SwingEvent] = []
    windows: Dict[str, PoseWindow] = {}
    baseline: Dict[str, Optional[float]] = {}
    for side in SIDES:
        kp, kc, pres, box = _side_arrays(tracks, side)
        if len(sel) == 0:
            continue
        sl = slice(int(sel[0]), int(sel[-1]) + 1)
        w = PoseWindow(kp=kp[sl], kpconf=None if kc is None else kc[sl], present=pres[sl],
                       timestamps=ts_all[sl], box=None if box is None else box[sl], side=side)
        windows[side] = w
        baseline[side] = None
        if w.box is not None:
            feet = np.where(w.present, w.box[:, 3], np.nan)
            if np.isfinite(feet).any():
                baseline[side] = float(np.nanpercentile(feet, 90))
        evs = detect_swings(w.kp, w.kpconf, w.present, w.timestamps, side=side, box=w.box,
                            handedness=handedness.get(side), speed_ref=(speed_ref or {}).get(side), params=p)
        events.extend(e for e in evs if start - 0.2 <= e.t <= end)
    events.sort(key=lambda e: e.t)
    n_motion = len(events)

    def collapse(evs: List[SwingEvent], same_side: bool, gap: float, key) -> Tuple[List[SwingEvent], int]:
        out: List[SwingEvent] = []
        dropped = 0
        for e in evs:
            if out and (out[-1].player_side == e.player_side) == same_side and e.t - out[-1].t < gap:
                dropped += 1
                if key(e) > key(out[-1]):
                    out[-1] = e
                continue
            out.append(e)
        return out, dropped

    # serve toss + contact of one serve: keep the contact (collapse_serve_pairs); opponent
    # overlap: keep the more confident; other same-side doubles (swing + recovery): keep
    # the faster swing (the hit is the fastest motion).
    events, serve_contacts, tosses = collapse_serve_pairs(events, windows, start, p, server=server)
    events, deduped = collapse(events, False, p.opp_dedupe_sec, lambda e: e.confidence)
    events, suppressed = collapse(events, True, p.same_side_min_gap_sec, lambda e: e.peak_speed)

    labelled: List[Tuple[SwingEvent, Dict[str, Any]]] = []
    seen = set()
    for e in events:
        w = windows[e.player_side]
        if id(e) in serve_contacts:  # contact of a collapsed toss/contact pair
            st = {"stroke": "serve",
                  "confidence": float(np.clip(0.6 + 0.4 * (serve_contacts[id(e)] - p.overhead_margin), 0.0, 1.0))}
        else:
            st = classify_stroke(e, w, point_start=start, is_first_of_side=e.player_side not in seen,
                                 handedness=handedness.get(e.player_side), server=server,
                                 baseline_foot_y=baseline.get(e.player_side), params=p)
        seen.add(e.player_side)
        labelled.append((e, st))

    prior = 0
    if labelled and labelled[0][1]["stroke"] != "serve":
        first = labelled[0][0]
        srv = server if server in SIDES else ("far" if first.player_side == "near" else "near")
        lead = first.t - start
        if srv != first.player_side and lead >= p.serve_prior_min_lead_sec:
            t = max(start, first.t - p.serve_prior_lead_sec)
            pe = SwingEvent(t=float(t), player_side=srv, peak_speed=0.0, arm=handedness.get(srv, "right"),
                            confidence=0.3, source="prior")
            labelled.insert(0, (pe, {"stroke": "serve", "confidence": 0.3}))
            prior += 1
    filled: List[Tuple[SwingEvent, Dict[str, Any]]] = []
    for item in labelled:
        if filled and filled[-1][0].player_side == item[0].player_side:
            a, b = filled[-1][0], item[0]
            other = "far" if a.player_side == "near" else "near"
            pe = SwingEvent(t=float((a.t + b.t) / 2.0), player_side=other, peak_speed=0.0,
                            arm=handedness.get(other, "right"), confidence=0.2, source="prior")
            filled.append((pe, {"stroke": "unknown", "confidence": 0.0}))
            prior += 1
        filled.append(item)

    shots = []
    for e, st in filled:
        shots.append({"t": round(e.t, 3), "player_side": e.player_side, "stroke": st["stroke"],
                      "stroke_confidence": round(float(st["confidence"]), 3), "confidence": round(e.confidence, 3),
                      "source": e.source, "arm": e.arm, "peak_speed": round(e.peak_speed, 2),
                      "width_sec": round(e.width_sec, 2)})
    return {"start": start, "end": end, "shots": shots, "rally_length": len(shots), "prior_filled": prior,
            "motion_events": n_motion, "serve_pairs": len(serve_contacts),
            "serve_tosses": [{"t": round(e.t, 3), "player_side": e.player_side} for e in tosses], "deduped": deduped, "suppressed": suppressed,
            "handedness": dict(handedness)}
