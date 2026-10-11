"""Sweep HeatmapDecodeConfig over dumped tracks and score against GT.
PYTHONPATH=src $PY scripts/eval/score_heatmap_decode.py --tracks X.npz --gt match.tdproj|segments.csv --gt-fps 59.94 [--grid | --cfg JSON] [--end-pad S]
GT: a beta-tennis .tdproj (validation!=deleted clips, warm-up before the first clip ignored) or a
segment_id,first_value,last_value CSV in frames at --gt-fps.
"""

import argparse
import ast
import csv
import json
import numpy as np
from infer.heatmap_decode import HeatmapDecodeConfig, decode_heatmap_segments


def load_gt(path, fps):
    if path.endswith(".tdproj"):
        d = json.load(open(path))
        clips = d["timeline"]["tracks"][0]["clips"]
        return sorted(
            (float(c["inTime"]), float(c["outTime"]))
            for c in clips
            if c.get("validation") != "deleted"
        )
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append((float(r["first_value"]) / fps, float(r["last_value"]) / fps))
    return sorted(out)


def union_len(iv):
    total, cs, ce = 0.0, None, None
    for s, e in sorted(iv):
        if cs is None or s > ce:
            if cs is not None:
                total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs if cs is not None else 0.0)


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def frame_metrics(pred, gt, end_t, step=0.1):
    t = np.arange(0, end_t, step)

    def mask(segs):
        m = np.zeros_like(t, dtype=bool)
        for s, e in segs:
            m |= (t >= s) & (t <= e)
        return m

    p, g = mask(pred), mask(gt)
    tp = (p & g).sum()
    return (tp / p.sum() if p.sum() else 0.0, tp / g.sum() if g.sum() else 0.0)


def score(pred, gt, iou_thr=0.5):
    # beta-tennis protocol: ignore warm-up (anything ending before the first GT clip)
    t0, t1 = gt[0][0], gt[-1][1]
    pred = [p for p in pred if p[1] > t0]
    bt_miss = sum(1 for g in gt if not any(iou(p, g) > 0 for p in pred))
    bt_fp = sum(1 for p in pred if not any(iou(p, g) > 0 for g in gt))
    clipped = [(max(s, t0), min(e, t1)) for s, e in pred if min(e, t1) > max(s, t0)]
    inter = sum(
        max(0.0, min(a[1], b[1]) - max(a[0], b[0])) for a in gt for b in clipped
    )
    pl, tl = union_len(clipped), union_len(gt)
    bt_tf1 = 2 * inter / (pl + tl) if pl + tl else 0.0
    # greedy 1:1 matching by IoU
    pairs = sorted(
        (
            (iou(p, g), i, j)
            for i, p in enumerate(pred)
            for j, g in enumerate(gt)
            if iou(p, g) >= iou_thr
        ),
        reverse=True,
    )
    up, ug, matched = set(), set(), []
    for v, i, j in pairs:
        if i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        matched.append((i, j))
    tp = len(matched)
    P = tp / len(pred) if pred else 0.0
    R = tp / len(gt) if gt else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0

    # merged: predictions covering >=50% of >=2 GT points
    def cov(g, p):
        inter = max(0.0, min(g[1], p[1]) - max(g[0], p[0]))
        return inter / (g[1] - g[0]) if g[1] > g[0] else 0

    merged = sum(1 for p in pred if sum(1 for g in gt if cov(g, p) >= 0.5) >= 2)
    covered = sum(1 for g in gt if any(cov(g, p) >= 0.5 for p in pred))
    fp_pure = sum(1 for p in pred if all(iou(p, g) == 0 for g in gt))
    so = [pred[i][0] - gt[j][0] for i, j in matched]
    eo = [pred[i][1] - gt[j][1] for i, j in matched]
    end_t = max([e for _, e in gt] + [e for _, e in pred]) + 5
    fP, fR = frame_metrics(pred, gt, end_t)
    return dict(
        bt_miss=bt_miss,
        bt_fp=bt_fp,
        bt_tf1=round(bt_tf1, 3),
        n_pred=len(pred),
        n_gt=len(gt),
        tp=tp,
        P=round(P, 3),
        R=round(R, 3),
        F1=round(F, 3),
        fP=round(fP, 3),
        fR=round(fR, 3),
        merged=merged,
        gt_covered=covered,
        missed=len(gt) - covered,
        pure_fp=fp_pure,
        start_med=round(float(np.median(so)), 2) if so else None,
        end_med=round(float(np.median(eo)), 2) if eo else None,
        start_mae=round(float(np.mean(np.abs(so))), 2) if so else None,
        end_mae=round(float(np.mean(np.abs(eo))), 2) if eo else None,
    )


ap = argparse.ArgumentParser()
ap.add_argument("--tracks", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--gt-fps", type=float, required=True)
ap.add_argument("--grid", action="store_true")
ap.add_argument("--json-out", default=None)
ap.add_argument(
    "--cfg", default=None, help="JSON dict of HeatmapDecodeConfig overrides"
)
ap.add_argument(
    "--end-pad",
    type=float,
    default=0.0,
    help="HeatmapDecodeConfig.end_pad_sec (runtime pad + clamp + re-merge)",
)
ap.add_argument(
    "--start-pad",
    type=float,
    default=0.0,
    help="seconds added to every predicted start (post-hoc, negative = earlier)",
)
a = ap.parse_args()
d = np.load(a.tracks, allow_pickle=True)
gt = load_gt(a.gt, a.gt_fps)
# Base = the manifest postprocess.params the dump was made with (the dump stores
# str(spec.params)), so "{}" reproduces the run's own decode on any artifact (v0.6.0:
# smooth 1.5, merge 0.0, hybrid floor 4.0). Older dumps without params fall back to the
# v0.5.0 manifest values.
V050_BASE = dict(
    mode="hybrid",
    threshold=0.5,
    peak_threshold=0.3,
    sigma_frames=2.5,
    min_duration_sec=0.3,
    max_duration_sec=60.0,
    hybrid_min_duration_sec=0.0,
    merge_gap_sec=0.5,
)


def _base_from_dump(dump):
    if "params" not in dump.files:
        return dict(V050_BASE)
    try:
        params = ast.literal_eval(str(dump["params"]))
    except (ValueError, SyntaxError):
        return dict(V050_BASE)
    fields = set(HeatmapDecodeConfig.__dataclass_fields__)
    return {k: v for k, v in dict(params).items() if k in fields and v is not None}


base = _base_from_dump(d)


def run(over):
    if a.end_pad and "end_pad_sec" not in over:
        over = {
            **over,
            "end_pad_sec": a.end_pad,
        }  # runtime knob (pads, clamps, re-merges)
    cfg = HeatmapDecodeConfig(**{**base, **over})
    pred = decode_heatmap_segments(
        d["pointness"], d["start_prob"], d["end_prob"], d["timestamps"], cfg
    )
    return pad([(float(s), float(e)) for s, e in pred])


def pad(segs):
    # start pad stays post-hoc (no runtime knob); end pad goes through HeatmapDecodeConfig
    # in run(); for default_segments (already decoded) it is applied post-hoc as a fallback.
    out = [(s + a.start_pad, e) for s, e in segs]
    return [(s, e) for s, e in out if e > s]


rows = []
if a.cfg:
    over = json.loads(a.cfg)
    rows.append((over, score(run(over), gt)))
elif a.grid:
    grid = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        for sm in (None, 1.0, 2.0):
            for gap in (0.0, 0.5):
                grid.append(
                    dict(threshold=thr, smooth_sigma_frames=sm, merge_gap_sec=gap)
                )
    for over in grid:
        rows.append((over, score(run(over), gt)))
else:
    dflt = pad([(float(s), float(e + a.end_pad)) for s, e in d["default_segments"]])
    rows.append(({"default_from_run": True}, score(dflt, gt)))
    rows.append(({}, score(run({}), gt)))

keys = [
    "n_pred",
    "tp",
    "P",
    "R",
    "F1",
    "fP",
    "fR",
    "merged",
    "missed",
    "pure_fp",
    "start_med",
    "end_med",
    "bt_miss",
    "bt_fp",
    "bt_tf1",
]
print("gt points:", len(gt))
print(" | ".join(["cfg"] + keys))
for over, s in rows:
    print(" | ".join([json.dumps(over)] + [str(s[k]) for k in keys]))
if a.json_out:
    json.dump([dict(cfg=o, **s) for o, s in rows], open(a.json_out, "w"), indent=1)
