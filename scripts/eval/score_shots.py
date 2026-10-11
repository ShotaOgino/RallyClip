"""Label-free sanity report for shot events + rule-based stroke types (roadmap Phase 2).

PYTHONPATH=src $PY scripts/eval/score_shots.py --tracks X.tracks.npz [--gt match.tdproj --gt-fps 59.94]
    [--cfg '{"threshold": 0.65, ...}'] [--server near|far|phase1] --json-out out.json

Points come from --gt (beta-tennis .tdproj, validation != deleted; or a
segment_id,first_value,last_value CSV in frames at --gt-fps), else from decoding the
dumped tracks with --cfg (HeatmapDecodeConfig overrides over the shipped defaults),
else from the dump's default_segments. The tracks must carry the near_/far_ player
keys (scripts/eval/dump_heatmap_tracks.py after 2026-10-11).

There are no shot labels yet, so the report is a plausibility check: shots per point,
rally-length histogram, stroke mix per side, fraction of prior-filled events, and the
inter-shot interval (real junior rallies: ~1.0-2.0 s between consecutive shots).

--server phase1 passes each point's Phase 1 server (analysis.players.detect_server on
the same dump; None where it finds none) to analyze_point_shots instead of letting the
shot pass infer it; with the far player observed (upscaled far crop) the shot pass's own
serve inference picks the near side on many far-serve points.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.shots import SIDES, analyze_point_shots, estimate_handedness  # noqa: E402

RALLY_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 9), (10, 10**9)]
DECODE_BASE = dict(mode="hybrid", threshold=0.5, peak_threshold=0.3, sigma_frames=2.5, min_duration_sec=0.3,
                   max_duration_sec=60.0, hybrid_min_duration_sec=0.0, merge_gap_sec=0.5)


def load_gt(path: str, fps: float):
    """Same reading as scripts/eval/score_heatmap_decode.py::load_gt."""
    if path.endswith(".tdproj"):
        d = json.load(open(path))
        clips = d["timeline"]["tracks"][0]["clips"]
        return sorted((float(c["inTime"]), float(c["outTime"])) for c in clips if c.get("validation") != "deleted")
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append((float(r["first_value"]) / fps, float(r["last_value"]) / fps))
    return sorted(out)


def load_points(tracks, gt=None, gt_fps=59.94, cfg=None):
    if gt:
        return load_gt(gt, gt_fps), "gt"
    if cfg is not None:
        from infer.heatmap_decode import HeatmapDecodeConfig, decode_heatmap_segments
        c = HeatmapDecodeConfig(**{**DECODE_BASE, **cfg})
        segs = decode_heatmap_segments(tracks["pointness"], tracks["start_prob"], tracks["end_prob"], tracks["timestamps"], c)
        return [(float(s), float(e)) for s, e in segs], "decode"
    return [(float(s), float(e)) for s, e in np.asarray(tracks["default_segments"])], "default_segments"


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _far_from_raw(d, max_foot_frac=0.55, max_h_ratio=0.8, near_iou=0.3, keep_dist_h=2.0, max_dx_frac=0.3):
    """Script-private far-player track from the dump's raw pre-filter detections.

    The shipped preprocessor's near-court mask drops the far player before assignment
    (far_present ~0 on beta-tennis), so for Phase 2 evaluation the far track is rebuilt
    here. Per frame, candidates are raw detections that (1) do not overlap the assigned
    near box (IoU < near_iou), (2) stand in the upper part of the frame (box bottom <
    max_foot_frac * ref height, and above the near player's feet), (3) are smaller than
    the near player (height < max_h_ratio * near height, or the match median near
    height when near is absent), (4) are centred within max_dx_frac * ref width of the
    frame centre (rejects ball kids / umpire at the sides). Among them: the one nearest the previous far pick if
    within keep_dist_h box heights, else the one nearest the court centre x
    (ref width / 2).
    To be unified with Phase 1's analysis.players.pick_far_player.
    Returns (box, kp, kpconf, present) or None when the dump has no raw keys.
    """
    if "raw_boxes" not in d:
        return None
    rb, rk, rc = d["raw_boxes"], d["raw_kp"], d["raw_kpconf"]
    cnt = np.asarray(d["raw_count"]).astype(int)
    ref_w, ref_h = (float(v) for v in np.asarray(d.get("ref_size", [1280, 720]))[:2])
    nb, npres = np.asarray(d["near_box"], dtype=np.float64), np.asarray(d["near_present"], bool)
    nh_all = nb[:, 3] - nb[:, 1]
    nh_med = float(np.nanmedian(nh_all[npres])) if npres.any() else ref_h / 4
    n = len(cnt)
    box = np.full((n, 4), np.nan)
    kp = np.full((n, 17, 2), np.nan)
    kc = np.full((n, 17), np.nan)
    pres = np.zeros(n, bool)
    prev = None
    for i in range(n):
        best, best_key = None, None
        nh = nh_all[i] if npres[i] and np.isfinite(nh_all[i]) else nh_med
        for j in range(min(cnt[i], rb.shape[1])):
            b = rb[i, j]
            if not np.isfinite(b).all():
                continue
            h = b[3] - b[1]
            if h <= 0 or b[3] >= max_foot_frac * ref_h or h >= max_h_ratio * nh:
                continue
            if npres[i] and (_iou(b, nb[i]) >= near_iou or b[3] >= nb[i, 3]):
                continue
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            if abs(c[0] - ref_w / 2) > max_dx_frac * ref_w:
                continue
            if prev is not None and np.linalg.norm(c - prev[0]) <= keep_dist_h * prev[1]:
                key = (0, float(np.linalg.norm(c - prev[0])))
            else:
                key = (1, abs(float(c[0]) - ref_w / 2))
            if best_key is None or key < best_key:
                best, best_key = j, key
        if best is None:
            continue
        b = rb[i, best]
        box[i], kp[i], kc[i], pres[i] = b, rk[i, best], rc[i, best], True
        prev = (np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), float(b[3] - b[1]))
    return box, kp, kc, pres


def server_consistency(points_out, tdproj_path):
    """Per-game agreement of the inferred server side, using only the tdproj scoreboard's
    game-end times (no server labels): within one game the server never changes side, so
    the share of points whose serve side equals the game's majority is a label-free check
    of serve detection + the prior. Returns None when the file has no game events."""
    try:
        ends = sorted(float(e["atTime"]) for e in json.load(open(tdproj_path))["scoreboard"]["gameEvents"])
    except (KeyError, TypeError, ValueError):
        return None
    games = {}
    for p in points_out:
        g = sum(1 for t in ends if t <= p["start"])
        games.setdefault(g, []).append(next((s["player_side"] for s in p["shots"] if s["stroke"] == "serve"), None))
    agree = tot = 0
    for v in games.values():
        vv = [x for x in v if x]
        if vv:
            agree += max(vv.count(side) for side in SIDES)
            tot += len(vv)
    pattern = " ".join("".join((x or "-")[0] for x in v) for _, v in sorted(games.items()))
    return {"games": len(games), "points_with_serve": tot, "points": len(points_out),
            "agreement": round(agree / tot, 3) if tot else None, "pattern": pattern}


def prepare_far_track(d, far_source="auto"):
    """Swap in the raw-rebuilt far track when asked (or, in auto, when the assigned far
    player is present in < 20 % of frames). Returns (tracks, "assigned"|"raw", assigned far frac)."""
    assigned_far = round(float(np.mean(d["far_present"])), 3)
    used = "assigned"
    if far_source == "raw" or (far_source == "auto" and assigned_far < 0.2):
        far = _far_from_raw(d)
        if far is not None:
            d = dict(d)
            d["far_box"], d["far_kp"], d["far_kpconf"], d["far_present"] = far
            used = "raw"
        elif far_source == "raw":
            raise SystemExit("tracks have no raw_* keys; re-dump with scripts/eval/dump_heatmap_tracks.py")
    return d, used, assigned_far


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "median": round(float(np.median(x)), 3),
            "p10": round(float(np.percentile(x, 10)), 3), "p90": round(float(np.percentile(x, 90)), 3)}


def summarize(points_out):
    lengths = [p["rally_length"] for p in points_out]
    hist = {}
    for lo, hi in RALLY_BINS:
        key = str(lo) if lo == hi else (f"{lo}+" if hi > 1000 else f"{lo}-{hi}")
        hist[key] = sum(1 for n in lengths if lo <= n <= hi)
    mix = {side: Counter() for side in SIDES}
    total = prior = 0
    gaps, same_side_motion = [], []
    for p in points_out:
        shots = p["shots"]
        total += len(shots)
        prior += p["prior_filled"]
        for s in shots:
            mix[s["player_side"]][s["stroke"] if s["source"] == "motion" else "prior:" + s["stroke"]] += 1
        for a, b in zip(shots, shots[1:]):
            gaps.append(b["t"] - a["t"])
        for side in SIDES:
            mt = [s["t"] for s in shots if s["source"] == "motion" and s["player_side"] == side]
            same_side_motion.extend(np.diff(mt).tolist())
    return {
        "points": len(points_out),
        "shots_total": total,
        "shots_per_point": _stats(lengths),
        "rally_length_hist": hist,
        "stroke_mix": {side: dict(sorted(mix[side].items())) for side in SIDES},
        "prior_filled": prior,
        "prior_fraction": round(prior / total, 3) if total else None,
        "motion_events": sum(p["motion_events"] for p in points_out),
        "motion_shots_by_side": {side: sum(1 for p in points_out for s in p["shots"]
                                           if s["source"] == "motion" and s["player_side"] == side) for side in SIDES},
        "deduped": sum(p["deduped"] for p in points_out),
        "suppressed": sum(p["suppressed"] for p in points_out),
        "inter_shot_sec": _stats(gaps),
        "same_side_motion_interval_sec": _stats(same_side_motion),
    }


def run(tracks_path, gt=None, gt_fps=59.94, cfg=None, server=None, far_source="auto"):
    d = dict(np.load(tracks_path, allow_pickle=True))
    missing = [k for k in ("near_kp", "near_present", "far_kp", "far_present") if k not in d]
    if missing:
        raise SystemExit(f"{tracks_path}: no player keys {missing}; re-dump with scripts/eval/dump_heatmap_tracks.py")
    d, used, assigned_far = prepare_far_track(d, far_source)
    points, source = load_points(d, gt, gt_fps, cfg)
    ts = np.asarray(d["timestamps"], dtype=np.float64)
    hand, hand_ratio = {}, {}
    for side in SIDES:
        arm, ratio = estimate_handedness(d[f"{side}_kp"], d.get(f"{side}_kpconf"), d[f"{side}_present"], ts, d.get(f"{side}_box"))
        hand[side], hand_ratio[side] = arm, round(float(ratio), 3)
    servers = [server] * len(points)
    if server == "phase1":
        from analysis.players import PlayerTracks, detect_server

        tracks = PlayerTracks.from_arrays(d)
        cache = {}
        servers = [detect_server(pt, tracks, _foot_cache=cache)["server"] for pt in points]
    out = []
    far_points = 0
    for pt, srv in zip(points, servers):
        r = analyze_point_shots(pt, d, server=srv, handedness=hand)
        m = (ts >= pt[0]) & (ts <= pt[1])
        r["far_present_frac"] = round(float(np.mean(d["far_present"][m])), 3) if m.any() else 0.0
        far_points += r["far_present_frac"] >= 0.5
        out.append(r)
    summary = summarize(out)
    summary["points_with_far_track"] = int(far_points)
    summary["server_consistency"] = server_consistency(out, gt) if gt and gt.endswith(".tdproj") else None
    in_point = np.zeros(len(ts), bool)
    for s, e in points:
        in_point |= (ts >= s) & (ts <= e)
    summary.update({"tracks": str(tracks_path), "points_source": source, "handedness": hand,
                    "handedness_ratio": hand_ratio, "far_source": used, "far_assigned_present_frac": assigned_far,
                    "present_frac": {s: round(float(np.mean(d[f"{s}_present"])), 3) for s in SIDES},
                    "present_frac_in_points": {s: round(float(np.mean(d[f"{s}_present"][in_point])), 3)
                                               if in_point.any() else None for s in SIDES}})
    return summary, out


def print_report(summary):
    p = print
    p(f"== {Path(summary['tracks']).name}")
    pf, pi = summary["present_frac"], summary["present_frac_in_points"]
    p(f"points: {summary['points']} (from {summary['points_source']})  far track: {summary['far_source']} "
      f"(assigned far present {summary['far_assigned_present_frac']})")
    p(f"points with a far track (>=50% of frames): {summary['points_with_far_track']}/{summary['points']}")
    p(f"present near/far: all frames {pf['near']}/{pf['far']}  in points {pi['near']}/{pi['far']}  handedness: "
      f"near {summary['handedness']['near']} (x{summary['handedness_ratio']['near']}), "
      f"far {summary['handedness']['far']} (x{summary['handedness_ratio']['far']})")
    sp = summary["shots_per_point"]
    p(f"shots: {summary['shots_total']}  per point mean {sp['mean']} median {sp['median']} p10 {sp['p10']} p90 {sp['p90']}")
    p("rally length hist: " + "  ".join(f"{k}:{v}" for k, v in summary["rally_length_hist"].items()))
    ms = summary["motion_shots_by_side"]
    p(f"motion shots near {ms['near']} far {ms['far']}  (raw motion events {summary['motion_events']})")
    p(f"motion events: {summary['motion_events']}  deduped (opp) {summary['deduped']}  suppressed (same side) "
      f"{summary['suppressed']}  prior-filled {summary['prior_filled']} ({summary['prior_fraction']} of shots)")
    for side in SIDES:
        mix = summary["stroke_mix"][side]
        p(f"  {side:4s} " + ("  ".join(f"{k}:{v}" for k, v in mix.items()) or "(none)"))
    sc = summary.get("server_consistency")
    if sc:
        p(f"serve side per game (tdproj game ends): agreement {sc['agreement']} over {sc['points_with_serve']}/"
          f"{sc['points']} points with a serve, {sc['games']} games: {sc['pattern']}")
    g, s = summary["inter_shot_sec"], summary["same_side_motion_interval_sec"]
    p(f"inter-shot interval s: mean {g['mean']} median {g['median']} p10 {g['p10']} p90 {g['p90']} (n={g['n']})")
    p(f"same-side motion interval s: mean {s['mean']} median {s['median']} (n={s['n']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", required=True, nargs="+")
    ap.add_argument("--gt", default=None, help=".tdproj or frames CSV (only with a single --tracks)")
    ap.add_argument("--gt-fps", type=float, default=59.94)
    ap.add_argument("--cfg", default=None, help="JSON dict of HeatmapDecodeConfig overrides (points by decode)")
    ap.add_argument("--server", default=None, choices=[*SIDES, "phase1"],
                    help="server side for every point (testing only), or phase1 = per point from detect_server")
    ap.add_argument("--far-source", default="auto", choices=["auto", "assigned", "raw"],
                    help="far track: the dump's assigned far player, or rebuilt from raw_* detections "
                         "(auto: raw when the assigned far player is present < 20%% of frames)")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    if a.gt and len(a.tracks) > 1:
        ap.error("--gt needs exactly one --tracks")
    cfg = json.loads(a.cfg) if a.cfg else None
    results = []
    for t in a.tracks:
        summary, points = run(t, a.gt, a.gt_fps, cfg, a.server, a.far_source)
        print_report(summary)
        results.append({"summary": summary, "points": points})
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.json_out, "w"), indent=1)
    return results


if __name__ == "__main__":
    main()
