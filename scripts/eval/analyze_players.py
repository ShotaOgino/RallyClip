"""Phase 1 player identity + server analysis on dumped tracks (offline, no video).
Run from repo root:
  PYTHONPATH=src $PY scripts/eval/analyze_players.py --tracks X.tracks.npz \
      [--gt match.tdproj --gt-fps 59.94] [--cfg '{"threshold":0.65,"hybrid_min_duration_sec":1.0,"end_pad_sec":1.6}'] \
      [--json-out out.json]

Points: the tdproj clips when --gt is given (analysis scored on true points), else the
heatmap decode of the dumped tracks (manifest params from the dump + --cfg overrides).
--decoded with --gt analyses the decoded points and scores those that overlap a GT point
(greedy IoU >= 0.3; ids compared up to an A/B relabelling since point 1 may be a warm-up).

Scoreboard-derived reference (when the tdproj has a `scoreboard`): beta-tennis records
game ends (`gameEvents[].atTime` + `gameWinner`; atTime = start of the first point of the
next game; an event at the last clip's start is the match-end marker), the `format`
(shortGame = sets to 4, oneSet = one set to 6), `initialServer` and tiebreak
`serverChanges`. Under the rules (server alternates per game; ends change after the 1st,
3rd, ... game of each set and every 6 tiebreak points) that gives, per point, the serving team and
whether the players have switched ends since the first point. Which team starts on the
near side is not recorded, so the server-side accuracy fits that ONE bit to the
detections (reported as `near_team0`). It is a derived reference, not hand labels.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

from analysis.players import (
    DEFAULT_CONFIG,
    PlayerAnalysisConfig,
    PlayerTracks,
    analyze_points,
    serve_consistency,
)
from infer.heatmap_decode import HeatmapDecodeConfig, decode_heatmap_segments


def load_tdproj(path):
    d = json.load(open(path, encoding="utf-8"))
    clips = sorted(
        (float(c["inTime"]), float(c["outTime"]))
        for c in d["timeline"]["tracks"][0]["clips"]
        if c.get("validation") != "deleted"
    )
    return clips, d.get("scoreboard")


def load_gt_points(path, fps):
    if path.endswith(".tdproj"):
        return load_tdproj(path)
    import csv

    with open(path) as f:
        return sorted((float(r["first_value"]) / fps, float(r["last_value"]) / fps) for r in csv.DictReader(f)), None


SET_TARGET = {"shortGame": 4, "oneSet": 6}  # games to win a set (tiebreak at target-target)


def _set_over(g0, g1, target):
    hi, lo = max(g0, g1), min(g0, g1)
    return (hi >= target and hi - lo >= 2) or (hi == target + 1 and lo == target)


def scoreboard_reference(points, sb, eps=1e-3):
    """Per point: game index, serving team (0/1), ends_swapped (bool vs the first point).

    Ends change after the 1st, 3rd, 5th ... game OF EACH SET (so a set with an odd
    number of games is followed by two changes in a row: its last game, then the first
    game of the next set) and every 6 points inside a tiebreak. Set length from the
    scoreboard `format` (shortGame = first to 4, oneSet = first to 6, win by 2).
    """
    if not sb or not points:
        return None
    last_start = points[-1][0]
    events = sorted(
        ((float(e["atTime"]), int(e.get("gameWinner", 0))) for e in sb.get("gameEvents", [])
         if float(e["atTime"]) < last_start - eps)
    )
    target = SET_TARGET.get(sb.get("format"), 6)
    # per game index: (number within its set, 1-based; is tiebreak)
    game_info = []
    g0 = g1 = 0
    k = 0
    for _t, winner in events + [(float("inf"), None)]:
        k += 1
        game_info.append((k, g0 == g1 == target))
        if winner is None:
            break
        g0, g1 = g0 + (winner == 0), g1 + (winner == 1)
        if _set_over(g0, g1, target):
            g0 = g1 = 0
            k = 0
    times = [t for t, _w in events]
    changes = sorted((float(c["atTime"]), int(c["newServer"])) for c in sb.get("serverChanges", []))
    ref = []
    for s, _e in points:
        g = sum(1 for t in times if t <= s + eps)
        server = int(sb.get("initialServer", 0)) ^ (g % 2)
        g_start = max([t for t in times if t <= s + eps], default=-1.0)
        # serverChanges sit on the clip start, sometimes a few frames after inTime
        tb = [(t, srv) for t, srv in changes if g_start - eps <= t <= s + 0.5]
        if tb:
            server = tb[-1][1]
        num, tiebreak = game_info[min(g, len(game_info) - 1)]
        ref.append({"game": g, "game_in_set": num, "server_team": server, "tiebreak": tiebreak})
    swapped = False
    pts_in_game = 0
    for i, r in enumerate(ref):
        if i > 0:
            prev = ref[i - 1]
            if r["game"] != prev["game"]:
                # games between the two points (normally one boundary)
                for gi in range(prev["game"], r["game"]):
                    num, _tb = game_info[min(gi, len(game_info) - 1)]
                    if num % 2 == 1:
                        swapped = not swapped
                pts_in_game = 0
            elif r["tiebreak"] and pts_in_game % 6 == 0:
                swapped = not swapped
        r["ends_swapped"] = swapped
        pts_in_game += 1
    return ref


def score_against_reference(rows, ref):
    n = len(rows)
    gt_ids = ["B" if r["ends_swapped"] else "A" for r in ref]
    id_acc = sum(1 for row, g in zip(rows, gt_ids) if row["near_id"] == g) / n
    gt_changes = {i for i in range(1, n) if ref[i]["ends_swapped"] != ref[i - 1]["ends_swapped"]}
    pred_changes = {i for i in range(1, n) if rows[i]["side_change_before"]}
    tp = len(gt_changes & pred_changes)
    best = None
    for near_team0 in (0, 1):
        sides = []
        for r in ref:
            near_team = near_team0 ^ int(r["ends_swapped"])
            sides.append("near" if r["server_team"] == near_team else "far")
        det = [(row["server"], s) for row, s in zip(rows, sides) if row["server"] is not None]
        acc = sum(1 for p, g in det if p == g) / len(det) if det else 0.0
        if best is None or acc > best["server_acc"]:
            near_det = [(row["server"], s) for row, s in zip(rows, sides) if row["server"] == "near"]
            best = {
                "near_team0": near_team0,
                "server_acc": round(acc, 3),
                "server_coverage": round(len(det) / n, 3),
                "server_acc_all": round(sum(1 for row, s in zip(rows, sides) if row["server"] == s) / n, 3),
                "near_raise_precision": round(sum(1 for p, g in near_det if g == "near") / len(near_det), 3) if near_det else None,
                "gt_near_serves": sides.count("near"),
                "gt_sides": sides,
            }
    # identity-independent server accuracy: does the predicted server_id sequence match
    # the GT serving team up to a relabelling? (needs both server and identity right)
    sid = [(row["server_id"], r["server_team"]) for row, r in zip(rows, ref) if row["server_id"] is not None]
    sid_acc = max(
        sum(1 for p, t in sid if (p == "A") == (t == m)) / len(sid) if sid else 0.0 for m in (0, 1)
    )
    return {
        "identity_acc": round(id_acc, 3),
        "side_changes_gt": sorted(gt_changes),
        "side_changes_pred": sorted(pred_changes),
        "side_change_P": round(tp / len(pred_changes), 3) if pred_changes else None,
        "side_change_R": round(tp / len(gt_changes), 3) if gt_changes else None,
        "server_id_acc": round(sid_acc, 3),
        **{k: v for k, v in best.items() if k != "gt_sides"},
        "_gt_sides": best["gt_sides"],
    }


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def score_decoded(rows, points, gt_points, ref):
    """Server / identity accuracy on decoded points matched to GT points."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(points) for j, g in enumerate(gt_points)
                    if iou(p, g) >= 0.3), reverse=True)
    used_p, used_g, match = set(), set(), {}
    for _v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        match[i] = j
    idx = sorted(match)
    if not idx:
        return None
    best = None
    for near_team0 in (0, 1):
        sides = {}
        for i in idx:
            r = ref[match[i]]
            sides[i] = "near" if r["server_team"] == (near_team0 ^ int(r["ends_swapped"])) else "far"
        det = [i for i in idx if rows[i]["server"] is not None]
        acc = sum(1 for i in det if rows[i]["server"] == sides[i]) / len(det) if det else 0.0
        if best is None or acc > best[0]:
            best = (acc, near_team0, len(det))
    id_acc = max(
        sum(1 for i in idx if (rows[i]["near_id"] == "A") == (ref[match[i]]["ends_swapped"] == flip)) / len(idx)
        for flip in (False, True)
    )
    return {"matched": len(idx), "n_decoded": len(points), "server_acc": round(best[0], 3),
            "server_coverage": round(best[2] / len(idx), 3), "near_team0": best[1],
            "identity_acc_relabelled": round(id_acc, 3)}


def decode_points(d, cfg_over):
    params = {}
    if "params" in d.files:
        try:
            params = ast.literal_eval(str(d["params"]))
        except (ValueError, SyntaxError):
            params = {}
    valid = set(HeatmapDecodeConfig.__dataclass_fields__)
    merged = {k: v for k, v in params.items() if k in valid and v is not None}
    merged.update({k: v for k, v in (cfg_over or {}).items() if k in valid})
    cfg = HeatmapDecodeConfig(**merged)
    segs = decode_heatmap_segments(d["pointness"], d["start_prob"], d["end_prob"], d["timestamps"], cfg)
    return [(float(s), float(e)) for s, e in segs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--gt", default=None, help="beta-tennis .tdproj (points + scoreboard) or segments CSV")
    ap.add_argument("--gt-fps", type=float, default=59.94)
    ap.add_argument("--cfg", default=None, help="JSON HeatmapDecodeConfig overrides (decoded points only)")
    ap.add_argument("--player-cfg", default=None, help="JSON PlayerAnalysisConfig overrides")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--verbose", action="store_true", help="print one line per point")
    ap.add_argument("--decoded", action="store_true", help="with --gt: analyse decoded points, score matched ones")
    a = ap.parse_args()

    d = np.load(a.tracks, allow_pickle=False)
    if "near_present" not in d.files:
        print(f"{a.tracks}: no player arrays (old dump); re-run dump_heatmap_tracks.py")
        return 2
    tracks = PlayerTracks.from_arrays({k: d[k] for k in d.files})
    pcfg = PlayerAnalysisConfig(**json.loads(a.player_cfg)) if a.player_cfg else DEFAULT_CONFIG

    sb, gt_points = None, None
    if a.gt:
        gt_points, sb = load_gt_points(a.gt, a.gt_fps)
    if a.gt and not a.decoded:
        points, source = gt_points, "gt"
    else:
        points, source = decode_points(d, json.loads(a.cfg) if a.cfg else None), "decoded"
    rows = analyze_points(points, tracks, pcfg)
    cons = serve_consistency(rows)
    ref = scoreboard_reference(gt_points, sb) if sb else None
    scored = score_against_reference(rows, ref) if ref and source == "gt" else None
    scored_decoded = score_decoded(rows, points, gt_points, ref) if ref and source == "decoded" else None

    name = Path(a.tracks).name.replace(".tracks.npz", "")
    n = len(rows)
    print(f"== {name}: {n} points ({source}), near present {tracks.near.present.mean():.2f}, "
          f"far present {tracks.far.present.mean():.3f}")
    print(f"server: near {cons['server_counts']['near']}, far {cons['server_counts']['far']}, "
          f"none {sum(1 for r in rows if r['server'] is None)} (none_frac {cons['none_frac']})")
    bs = cons["by_side"]
    print(f"by_side runs: median {bs['median_run']}, short(1-2) frac {bs['short_run_frac']}, runs {bs['runs']}")
    if "by_player" in cons:
        bp = cons["by_player"]
        print(f"by_player runs: median {bp['median_run']}, 4-7 frac {bp['runs_4_to_7_frac']}, "
              f"short(1-2) frac {bp['short_run_frac']}, runs {bp['runs']}")
    changes = [r["start_s"] for r in rows if r["side_change_before"]]
    print(f"side changes: {len(changes)} at {[round(t) for t in changes]}")
    print(f"identity confidence: mean {np.mean([r['identity_confidence'] for r in rows]):.2f}, "
          f"last {rows[-1]['identity_confidence'] if rows else None}")
    if scored:
        print(f"vs scoreboard: server_acc {scored['server_acc']} (coverage {scored['server_coverage']}, "
              f"all-points {scored['server_acc_all']}, near_team0 fitted={scored['near_team0']}), "
              f"near-raise precision {scored['near_raise_precision']}, identity_acc {scored['identity_acc']}, "
              f"server_id_acc {scored['server_id_acc']}, side changes P {scored['side_change_P']} "
              f"R {scored['side_change_R']} (gt {len(scored['side_changes_gt'])}, pred {len(scored['side_changes_pred'])})")
    if scored_decoded:
        print(f"vs scoreboard (decoded points matched to GT): {scored_decoded}")
    if a.verbose:
        gts = scored["_gt_sides"] if scored else [None] * n
        ref = ref if scored else None
        for i, (r, g) in enumerate(zip(rows, gts)):
            ref_i = ref[i] if ref else {}
            print(f"  {i:3d} {r['start_s']:8.1f}-{r['end_s']:7.1f} srv {str(r['server']):5s} "
                  f"({r['server_confidence']:.2f}) gt {str(g):5s} near {r['near_id']} "
                  f"chg {int(r['side_change_before'])} gtswap {int(ref_i.get('ends_swapped', 0))} "
                  f"game {ref_i.get('game', '-')} np {r['near_present_frac']:.2f}")
    if a.json_out:
        out = {"tracks": a.tracks, "source": source, "points": rows, "consistency": cons}
        if scored:
            out["scoreboard_eval"] = {k: v for k, v in scored.items() if not k.startswith("_")}
        if scored_decoded:
            out["scoreboard_eval_decoded"] = scored_decoded
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
