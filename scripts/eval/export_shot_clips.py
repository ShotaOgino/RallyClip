"""Labelling helper (roadmap Phase 2): cut short clips around detected shot events and
write a CSV template a human fills in with the true stroke type.

PYTHONPATH=src $PY scripts/eval/export_shot_clips.py --tracks X.tracks.npz --video V.mp4 --out-dir D
    [--n 60] [--seed 0] [--clip-sec 1.2] [--gt match.tdproj --gt-fps 59.94 | --cfg JSON] [--include-prior]

Clips are cut with the runtime's PyAV writer (segmentation.segment.segment_video,
libx264 re-encode; frame-accurate). `labels_template.csv` columns: event_id, clip, t,
side, source, predicted_stroke, stroke_confidence, event_confidence, peak_speed,
point_start, point_end, label (empty), notes (empty). Suggested label vocabulary:
serve, forehand, backhand, volley, overhead, slice, not_a_shot.
Each clip is a full-resolution re-encode (~1 s of work per clip at 1080p); sample a
few dozen events, not a whole match.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[1] / "src", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis.shots import SIDES, analyze_point_shots, match_handedness, match_swing_reference  # noqa: E402
from score_shots import load_points, prepare_far_track  # noqa: E402

FIELDS = ["event_id", "clip", "t", "side", "source", "predicted_stroke", "stroke_confidence",
          "event_confidence", "peak_speed", "point_start", "point_end", "label", "notes"]


def collect_events(tracks, points, include_prior=False):
    hand = match_handedness(tracks, points=points)
    ref = match_swing_reference(tracks, points, handedness=hand)
    events = []
    for s, e in points:
        r = analyze_point_shots((s, e), tracks, handedness=hand, speed_ref=ref)
        for shot in r["shots"]:
            if shot["source"] != "motion" and not include_prior:
                continue
            events.append({**shot, "point_start": round(s, 3), "point_end": round(e, 3)})
    return events


def video_info(path):
    import av  # runtime video decode is PyAV-only

    with av.open(str(path)) as c:
        v = c.streams.video[0]
        start = float(v.start_time * v.time_base) if v.start_time is not None else 0.0
        if c.duration:
            dur = float(c.duration) / 1e6
        elif v.duration is not None:
            dur = float(v.duration * v.time_base)
        else:
            dur = float("inf")
    return start, dur


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-sec", type=float, default=1.2)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--gt-fps", type=float, default=59.94)
    ap.add_argument("--cfg", default=None, help="JSON HeatmapDecodeConfig overrides (points by decode)")
    ap.add_argument("--far-source", default="auto", choices=["auto", "assigned", "raw"],
                    help="far track source, as in score_shots.py")
    ap.add_argument("--include-prior", action="store_true", help="also sample prior-filled (unobserved) events")
    a = ap.parse_args(argv)

    from segmentation.segment import segment_video

    tracks = dict(np.load(a.tracks, allow_pickle=True))
    if any(f"{s}_kp" not in tracks for s in SIDES):
        raise SystemExit(f"{a.tracks}: no player keys; re-dump with scripts/eval/dump_heatmap_tracks.py")
    tracks, far_used, _ = prepare_far_track(tracks, a.far_source)
    points, source = load_points(tracks, a.gt, a.gt_fps, json.loads(a.cfg) if a.cfg else None)
    events = collect_events(tracks, points, a.include_prior)
    rng = np.random.default_rng(a.seed)
    k = min(a.n, len(events))
    pick = sorted(rng.choice(len(events), size=k, replace=False).tolist()) if k else []
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v_start, v_dur = video_info(a.video)
    half = a.clip_sec / 2.0
    rows = []
    for n_done, idx in enumerate(pick, 1):
        ev = events[idx]
        eid = f"ev{idx:05d}"
        clip = f"{eid}_{ev['player_side']}_{ev['t']:.2f}.mp4"
        lo = max(0.0, ev["t"] - half)
        hi = min(v_dur, ev["t"] + half)
        segment_video(str(a.video), [(v_start + lo, v_start + hi)], str(out / clip))
        rows.append({"event_id": eid, "clip": clip, "t": ev["t"], "side": ev["player_side"], "source": ev["source"],
                     "predicted_stroke": ev["stroke"], "stroke_confidence": ev["stroke_confidence"],
                     "event_confidence": ev["confidence"], "peak_speed": ev["peak_speed"],
                     "point_start": ev["point_start"], "point_end": ev["point_end"], "label": "", "notes": ""})
        print(f"[{n_done}/{k}] {clip}", flush=True)
    with open(out / "labels_template.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"points: {len(points)} (from {source}), far track: {far_used}, events: {len(events)}, "
          f"exported: {len(rows)} -> {out}")
    return rows


if __name__ == "__main__":
    main()
