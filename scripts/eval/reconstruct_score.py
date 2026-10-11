"""Score timeline from a winner sequence, or rules check of a detected server sequence.

PYTHONPATH=src $PY scripts/eval/reconstruct_score.py --winners-csv F_winner.csv \
    [--first-server A|B] [--format bo3|bo3-mtb10|bo1|bo5] [--no-ad] [--unit auto|game|point] [--json-out out.json]
PYTHONPATH=src $PY scripts/eval/reconstruct_score.py --servers-json points.json \
    [--winners-csv F] [--format ...] [--min-game-points 4] [--json-out out.json]

--winners-csv: beta-tennis Swing/VideoEditing *_winner.csv (TrimDeck rows are GAME
events, so --unit auto picks "game" when score_after grows by one game per row).
Winner code 0 -> "A" (named from winner_name), 1 -> "B".

--servers-json accepts either
  * a plain JSON list of "A" / "B" / null (one per detected point), or
  * a list of dicts with "server" in {"near", "far", "A", "B", null} plus "near_id" /
    "far_id" ("A"/"B") mapping the ends to players (Phase 1 per-point output), or
  * a dict {"points": [...]} wrapping either list.
With --winners-csv (point-level) the exact server sequence is replayed; otherwise game
boundaries are inferred by dynamic programming (tiebreaks not modelled).
"""
import argparse
import json
import sys

from analysis.score import MatchFormat, check_winner_csv, infer_first_server_and_check, winners_from_winner_csv


def load_servers(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("points", data.get("per_point", []))
    out = []
    for k, item in enumerate(data):
        if item is None or item in ("A", "B"):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise SystemExit("servers-json item %d: expected 'A'/'B'/null or a dict, got %r" % (k, item))
        s = item.get("server")
        if s in ("near", "far"):
            s = item.get(s + "_id")
            if s not in ("A", "B"):
                s = None  # end known, identity unknown
        elif s not in ("A", "B"):
            s = None
        out.append(s)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winners-csv")
    ap.add_argument("--servers-json")
    ap.add_argument("--first-server", choices=("A", "B"), default="A")
    ap.add_argument("--format", default="bo3", choices=("bo3", "bo3-mtb10", "bo1", "bo5"))
    ap.add_argument("--no-ad", action="store_true")
    ap.add_argument("--unit", default="auto", choices=("auto", "game", "point"))
    ap.add_argument("--min-game-points", type=int, default=4)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    if not a.winners_csv and not a.servers_json:
        ap.error("need --winners-csv and/or --servers-json")
    fmt = MatchFormat.from_name(a.format, no_ad=a.no_ad, first_server=a.first_server)
    result = {}

    if a.servers_json:
        servers = load_servers(a.servers_json)
        winners = None
        if a.winners_csv:
            winners = winners_from_winner_csv(a.winners_csv)["winners"]
        res = infer_first_server_and_check(servers, fmt, winners, min_game_points=a.min_game_points)
        result["server_check"] = res
        agree = "n/a" if res["agreement"] is None else "%.3f" % res["agreement"]
        print("mode=%s first_server=%s agreement=%s compared=%d none=%d games=%d cost=%s"
              % (res["mode"], res["first_server"], agree, res["n_compared"], res["n_none"],
                 len(res["games"]), res["cost_by_first_server"]))
        print("disagreeing point indexes: %s" % (res["disagreements"] or "none"))
        for g in res["games"]:
            print("  game %3d  points %4d-%4d  server %s" % (g["game_index"], g["start"], g["end"] - 1, g["server"]))
    elif a.winners_csv:
        res = check_winner_csv(a.winners_csv, fmt, a.unit)
        result["score_check"] = res
        names = res["names"]
        print("A=%s  B=%s  unit=%s  rows=%d  first_server=%s  format=%s%s"
              % (names["A"], names["B"], res["unit"], res["n_rows"], a.first_server, a.format,
                 " no-ad" if a.no_ad else ""))
        for r, t in zip(res["timeline"], res["at_time"]):
            flag = {None: "?", "per_set": "ok", "cumulative": "ok(cum)"}.get(r["csv_match"], "?")
            if r["index"] in res["mismatches"]:
                flag = "MISMATCH"
            ends = " <ends>" if r.get("change_of_ends_after") or r.get("change_of_ends_before") else ""
            print("%3d  t=%8s  win %s  srv %s  %-22s csv %-10s %s%s"
                  % (r["index"], "%.1f" % t if t is not None else "-", r["winner"], r["server"],
                     r["score_after"], r["csv_score_after"], flag, ends))
        print("final: %s  match_over=%s  mismatches=%s"
              % (res["final_score"], res["match_over"], res["mismatches"] or "none"))

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=True)
        print("wrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
