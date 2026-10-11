"""Stroke-labelling helper (roadmap Phase 2): one 3-frame PNG strip per near-player swing.

PYTHONPATH=src $PY scripts/eval/export_shot_frames.py --tracks X.tracks.npz --video V.mp4
    --gt match.tdproj --out-dir D [--n 300] [--seed 0] [--side near] [--include-between]
    [--hand left|right]

Population (all on the chosen side, motion events only, never prior-filled ones):

* every swing ``analysis.shots.detect_swings`` finds inside a GT point (same window and
  handedness as ``analyze_point_shots``), *before* the alternation merge. ``merged=1``
  marks the swings that survive as shots of ``analyze_point_shots`` (their predicted
  stroke is the one it returns); ``merged=0`` swings were dropped by the opponent
  de-dupe / same-side suppression and are classified with ``classify_stroke`` in the
  same context, so labels also tell whether the merge dropped real shots.
* ``--include-between``: swings detected in the gaps between GT points
  (``point_index=-1``; no serve context). They measure what the detector fires on when
  the point boundaries are loose; they are kept apart in the scorer.

Sampling: when the population exceeds ``--n``, picks are stratified by predicted stroke
(equal share per stroke, leftovers redistributed) and, inside each stroke, spread evenly
over the match in time (random phase from ``--seed``).

Frames: decoded with PyAV (never cv2 for video) at t - 0.25 s, t, t + 0.25 s (nearest
decoded frame), sorted by time, one seek + one sequential pass per point (or per gap).
Each frame is cropped to the union of the player's boxes within +-0.3 s of t expanded by
60 % (and widened to at least 3:4 so the racket stays in), else the lower-centre half of
the frame; the three crops are tiled side by side into ``<event_id>.png`` (~1200 px wide,
labelled "-0.25s / 0 / +0.25s"). File names carry nothing but the event id, so a labeller
looking at the PNGs does not see the prediction.

Outputs in --out-dir: ``events.csv`` (event_id, t, point_index, in_point, merged,
predicted_stroke, confidence, event_confidence, arm, assumed_hand, crop, png) and
``labels.csv`` (event_id, t, point_index, label, handedness, label_confidence, labeller,
notes; label columns empty).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[1] / "src", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis.shots import (  # noqa: E402
    DEFAULT_PARAMS,
    PoseWindow,
    _side_arrays,
    analyze_point_shots,
    classify_stroke,
    detect_swings,
    match_handedness,
)
from score_shots import load_points  # noqa: E402

OFFSETS = (-0.25, 0.0, 0.25)
EXPAND = 0.6
BOX_WINDOW_SEC = 0.3
MIN_ASPECT = 0.75  # crop width / height
TILE_W = 400
EVENT_FIELDS = ["event_id", "t", "point_index", "in_point", "merged", "predicted_stroke", "confidence",
                "event_confidence", "arm", "assumed_hand", "crop", "png"]
LABEL_FIELDS = ["event_id", "t", "point_index", "label", "handedness", "label_confidence", "labeller", "notes"]


# ------------------------------------------------------------------ geometry / tiling


def crop_rect(frame_wh: Tuple[int, int], box: Optional[Sequence[float]], expand: float = EXPAND,
              min_aspect: float = MIN_ASPECT) -> Tuple[int, int, int, int]:
    """Integer (x1, y1, x2, y2) crop in frame pixels.

    ``box`` (x1, y1, x2, y2, frame pixels) grows by ``expand`` in each dimension around
    its centre, is widened to ``min_aspect`` (w/h) if narrower, then clipped to the frame.
    No/degenerate box -> the lower-centre half of the frame (x in [W/4, 3W/4], y in [H/2, H]).
    """
    w, h = int(frame_wh[0]), int(frame_wh[1])
    if box is None or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        return w // 4, h // 2, (3 * w) // 4, h
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    bw, bh = (box[2] - box[0]) * (1 + expand), (box[3] - box[1]) * (1 + expand)
    bw = max(bw, min_aspect * bh)
    x1, x2 = int(round(max(0.0, cx - bw / 2))), int(round(min(float(w), cx + bw / 2)))
    y1, y2 = int(round(max(0.0, cy - bh / 2))), int(round(min(float(h), cy + bh / 2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return w // 4, h // 2, (3 * w) // 4, h
    return x1, y1, x2, y2


def tile_crops(frames: Sequence[Optional[np.ndarray]], rect: Tuple[int, int, int, int], tile_w: int = TILE_W,
               captions: Optional[Sequence[str]] = None, gap: int = 4) -> np.ndarray:
    """Crop each HxWx3 uint8 frame to ``rect``, resize to ``tile_w`` wide (same height for
    all), and tile left to right with a ``gap``-px white separator. A missing frame
    (None) becomes a grey tile. Captions are drawn top-left when given."""
    import cv2  # image ops only

    x1, y1, x2, y2 = rect
    cw, ch = x2 - x1, y2 - y1
    tile_h = max(1, int(round(tile_w * ch / cw)))
    tiles = []
    for k, fr in enumerate(frames):
        if fr is None:
            t = np.full((tile_h, tile_w, 3), 128, np.uint8)
        else:
            c = np.ascontiguousarray(fr[y1:y2, x1:x2])
            t = cv2.resize(c, (tile_w, tile_h), interpolation=cv2.INTER_AREA if cw > tile_w else cv2.INTER_LINEAR)
        if captions:
            cv2.putText(t, captions[k], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(t, captions[k], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
        if k < len(frames) - 1:
            tiles.append(np.full((tile_h, gap, 3), 255, np.uint8))
    return np.concatenate(tiles, axis=1)


def event_box(tracks, side: str, t: float, scale: Tuple[float, float], window: float = BOX_WINDOW_SEC):
    """Union of the side's present boxes within +-window of t, in frame pixels (None if none)."""
    ts = np.asarray(tracks["timestamps"], dtype=np.float64)
    m = (np.abs(ts - t) <= window + 1e-9) & np.asarray(tracks[f"{side}_present"], bool)
    if not m.any():
        return None
    b = np.asarray(tracks[f"{side}_box"], dtype=np.float64)[m]
    b = b[np.isfinite(b).all(axis=1)]
    if not len(b):
        return None
    sx, sy = scale
    return np.array([b[:, 0].min() * sx, b[:, 1].min() * sy, b[:, 2].max() * sx, b[:, 3].max() * sy])


# ------------------------------------------------------------------ events


def collect_swings(tracks, points, side: str = "near", include_between: bool = False,
                   hand_override: Optional[str] = None) -> List[dict]:
    """Pre-merge motion swings of ``side`` (see module docstring), time-sorted.

    ``hand_override`` replaces the match-level handedness estimate for ``side`` (a
    counterfactual: the swing set is unchanged, only the stroke rules see another arm)."""
    p = DEFAULT_PARAMS
    hand = match_handedness(tracks, p)
    if hand_override:
        hand[side] = hand_override
    ts = np.asarray(tracks["timestamps"], dtype=np.float64)
    kp, kc, pres, box = _side_arrays(tracks, side)

    def window(lo, hi):
        sel = np.where((ts >= lo) & (ts <= hi))[0]
        if len(sel) == 0:
            return None
        sl = slice(int(sel[0]), int(sel[-1]) + 1)
        return PoseWindow(kp=kp[sl], kpconf=None if kc is None else kc[sl], present=pres[sl], timestamps=ts[sl],
                          box=None if box is None else box[sl], side=side)

    out = []
    for pi, (s, e) in enumerate(points):
        w = window(s - p.pre_pad_sec, e)
        if w is None:
            continue
        evs = [x for x in detect_swings(w.kp, w.kpconf, w.present, w.timestamps, side=side, box=w.box,
                                        handedness=hand.get(side), params=p) if s - 0.2 <= x.t <= e]
        shots = [sh for sh in analyze_point_shots((s, e), tracks, handedness=hand, params=p)["shots"]
                 if sh["source"] == "motion" and sh["player_side"] == side]
        feet = np.where(w.present, w.box[:, 3], np.nan) if w.box is not None else None
        base = float(np.nanpercentile(feet, 90)) if feet is not None and np.isfinite(feet).any() else None
        for j, ev in enumerate(evs):
            hit = next((sh for sh in shots if abs(sh["t"] - round(ev.t, 3)) < 1e-6), None)
            if hit is not None:
                stroke, conf = hit["stroke"], hit["stroke_confidence"]
            else:
                st = classify_stroke(ev, w, point_start=s, is_first_of_side=(j == 0), handedness=hand.get(side),
                                     baseline_foot_y=base, params=p)
                stroke, conf = st["stroke"], round(float(st["confidence"]), 3)
            out.append({"t": round(ev.t, 3), "point_index": pi, "in_point": 1, "merged": int(hit is not None),
                        "predicted_stroke": stroke, "confidence": conf, "event_confidence": round(ev.confidence, 3),
                        "arm": ev.arm, "assumed_hand": hand.get(side)})
    if include_between:
        bounds = [(-np.inf, -np.inf)] + list(points) + [(np.inf, np.inf)]
        for (_, prev_e), (next_s, _) in zip(bounds, bounds[1:]):
            lo, hi = max(float(ts[0]), prev_e), min(float(ts[-1]), next_s)
            if hi - lo < 2.0:
                continue
            w = window(lo, hi)
            if w is None:
                continue
            for ev in detect_swings(w.kp, w.kpconf, w.present, w.timestamps, side=side, box=w.box,
                                    handedness=hand.get(side), params=p):
                if not (lo + 0.2 <= ev.t <= hi - 0.2):
                    continue
                st = classify_stroke(ev, w, point_start=None, handedness=hand.get(side), params=p)
                out.append({"t": round(ev.t, 3), "point_index": -1, "in_point": 0, "merged": 0,
                            "predicted_stroke": st["stroke"], "confidence": round(float(st["confidence"]), 3),
                            "event_confidence": round(ev.confidence, 3), "arm": ev.arm,
                            "assumed_hand": hand.get(side)})
    out.sort(key=lambda r: r["t"])
    return out


def stratified_pick(events: Sequence[dict], n: int, seed: int = 0, key: str = "predicted_stroke") -> List[int]:
    """Indices (sorted) of up to n events: equal share per ``key`` value (leftover quota
    goes to the larger strata), evenly spaced in time order within each stratum."""
    if n >= len(events):
        return list(range(len(events)))
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, ev in enumerate(events):
        groups[ev[key]].append(i)
    quota = {g: 0 for g in groups}
    left = n
    open_g = sorted(groups, key=lambda g: len(groups[g]))
    while left > 0 and open_g:
        share = max(1, left // len(open_g))
        for g in list(open_g):
            take = min(share, len(groups[g]) - quota[g], left)
            quota[g] += take
            left -= take
            if quota[g] >= len(groups[g]):
                open_g.remove(g)
            if left == 0:
                break
    pick = []
    for g, idx in groups.items():
        k = quota[g]
        if k <= 0:
            continue
        idx = sorted(idx, key=lambda i: events[i]["t"])
        step = len(idx) / k
        phase = rng.uniform(0, step)
        pick.extend(idx[min(len(idx) - 1, int(phase + j * step))] for j in range(k))
    return sorted(set(pick))


# ------------------------------------------------------------------ decode


def _video_meta(path):
    import av  # runtime video decode is PyAV-only

    with av.open(str(path)) as c:
        v = c.streams.video[0]
        start = float(v.start_time * v.time_base) if v.start_time is not None else 0.0
        return start, int(v.codec_context.width), int(v.codec_context.height)


def grab_frames(path, groups: Sequence[Sequence[float]], start_offset: float = 0.0, tol: float = 0.05):
    """For each group of time-sorted target times (seconds from the video start), one seek
    to the group's first time and one sequential decode to its last; returns, per group,
    the nearest decoded RGB frame for each target (None if none within ``tol`` s)."""
    import av

    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        tb = float(st.time_base)
        for targets in groups:
            targets = list(targets)
            res: List[Optional[np.ndarray]] = [None] * len(targets)
            if not targets:
                out.append(res)
                continue
            best = [np.inf] * len(targets)
            c.seek(int(max(0.0, start_offset + targets[0] - 1.0) / tb), stream=st, backward=True, any_frame=False)
            last = targets[-1]
            for fr in c.decode(st):
                if fr.pts is None:
                    continue
                ft = fr.pts * tb - start_offset
                if ft < targets[0] - tol:
                    continue
                arr = None
                for k, tt in enumerate(targets):
                    d = abs(ft - tt)
                    if d <= tol and d < best[k]:
                        if arr is None:
                            arr = fr.to_ndarray(format="rgb24")
                        best[k], res[k] = d, arr
                if ft > last + tol:
                    break
            out.append(res)
    return out


# ------------------------------------------------------------------ main


def find_tdproj(path: str) -> str:
    if os.path.isdir(path):
        c = [g for g in sorted(glob.glob(os.path.join(path, "*.tdproj"))) if not os.path.basename(g).startswith("._")]
        if not c:
            raise SystemExit(f"no .tdproj in {path}")
        return c[0]
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True, help=".tdproj (or its directory) / segment CSV with the points")
    ap.add_argument("--gt-fps", type=float, default=59.94)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side", default="near", choices=["near", "far"])
    ap.add_argument("--include-between", action="store_true", help="also swings between GT points (point_index -1)")
    ap.add_argument("--hand", choices=["left", "right"], default=None,
                    help="override the estimated handedness of --side (counterfactual stroke rules)")
    a = ap.parse_args(argv)
    import cv2  # PNG write (image op)

    tracks = dict(np.load(a.tracks, allow_pickle=True))
    if f"{a.side}_kp" not in tracks:
        raise SystemExit(f"{a.tracks}: no player keys; re-dump with scripts/eval/dump_heatmap_tracks.py")
    points, _ = load_points(tracks, find_tdproj(a.gt), a.gt_fps)
    events = collect_swings(tracks, points, a.side, a.include_between, a.hand)
    pick = [events[i] for i in stratified_pick(events, a.n, a.seed)]
    for k, ev in enumerate(pick, 1):
        ev["event_id"] = f"ev{k:04d}"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v_start, vw, vh = _video_meta(a.video)
    ref = np.asarray(tracks.get("ref_size", [vw, vh]), dtype=np.float64)
    scale = (vw / float(ref[0]), vh / float(ref[1]))

    # one decode pass per point (gaps grouped by the next point index they precede)
    by_group: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for ev in pick:
        gkey = (ev["point_index"], 0) if ev["in_point"] else (-1, int(ev["t"] // 60))
        by_group[gkey].append(ev)
    groups = sorted(by_group.values(), key=lambda g: g[0]["t"])
    targets = [sorted({round(ev["t"] + o, 4) for ev in g for o in OFFSETS}) for g in groups]
    frames = grab_frames(a.video, targets, v_start)
    captions = ["-0.25s", "0", "+0.25s"]
    done = 0
    for g, tg, fr in zip(groups, targets, frames):
        lut = dict(zip(tg, fr))
        for ev in g:
            imgs = [lut.get(round(ev["t"] + o, 4)) for o in OFFSETS]
            rect = crop_rect((vw, vh), event_box(tracks, a.side, ev["t"], scale))
            ev["crop"] = "box" if rect != crop_rect((vw, vh), None) else "fallback"
            png = out / f"{ev['event_id']}.png"
            cv2.imwrite(str(png), tile_crops(imgs, rect, captions=captions)[:, :, ::-1])
            ev["png"] = png.name  # relative to --out-dir (no machine paths in the CSV)
            done += 1
        print(f"[{done}/{len(pick)}] frames written", flush=True)
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        w.writeheader()
        w.writerows({k: ev.get(k, "") for k in EVENT_FIELDS} for ev in pick)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        w.writerows({"event_id": ev["event_id"], "t": ev["t"], "point_index": ev["point_index"]} for ev in pick)
    n_in = sum(1 for ev in events if ev["in_point"])
    print(f"points: {len(points)}, swings: {len(events)} ({n_in} in points), exported: {len(pick)} -> {out}")
    print(json.dumps({"crop_fallback": sum(1 for ev in pick if ev["crop"] == "fallback")}))
    return pick


if __name__ == "__main__":
    main()
