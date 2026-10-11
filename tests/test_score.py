"""Tennis score state machine, reconstruction and server-sequence check (src/analysis/score.py)."""

import ast
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from analysis import score as S
from analysis.score import (
    MatchFormat,
    advance,
    advance_game,
    check_winner_csv,
    infer_first_server_and_check,
    initial_state,
    parse_score_after,
    reconstruct,
    reconstruct_games,
    winners_from_winner_csv,
)

ROOT = Path(__file__).resolve().parents[1]

# Verbatim copy of beta-tennis Swing/VideoEditing/ITF_SENKO-CUP_..._after5s_winner.csv
# (TrimDeck game events: one row per GAME; score_after = games summed over the match).
ITF_CSV = """point_no,atTime,segment_id,validation,winner,winner_name,score_after
1,155.99,,,0,"UCHINO, Takayuki",1-0
2,409.78,,,0,"UCHINO, Takayuki",2-0
3,631.4,,,0,"UCHINO, Takayuki",3-0
4,834.9,,,0,"UCHINO, Takayuki",4-0
5,987.99,,,0,"UCHINO, Takayuki",5-0
6,1621.99,,,0,"UCHINO, Takayuki",6-0
7,1813.11,,,0,"UCHINO, Takayuki",7-0
8,2000.53,,,1,"GOTO, Yusuke",7-1
9,2371.57,,,1,"GOTO, Yusuke",7-2
10,2509.17,,,0,"UCHINO, Takayuki",8-2
11,2707.1,,,0,"UCHINO, Takayuki",9-2
12,2908.27,,,1,"GOTO, Yusuke",9-3
13,3139.64,,,1,"GOTO, Yusuke",9-4
14,3478.14,,,0,"UCHINO, Takayuki",10-4
15,3686.45,,,1,"GOTO, Yusuke",10-5
16,4236.83,,,0,"UCHINO, Takayuki",11-5
17,4553.88,,,0,"UCHINO, Takayuki",12-5
18,4795.46,,,1,"GOTO, Yusuke",12-6
"""
ITF_REAL = Path("/Users/shota/Documents/beta-tennis/Swing/VideoEditing/"
                "ITF_SENKO-CUP_MT1000_Yamanashi_MS40_QF_Takayuki-UCHINO_Yusuke-GOTO_after5s_winner.csv")


def play(state, seq):
    for w in seq:
        state = advance(state, w)
    return state


def love_games(state, game_winners):
    for w in game_winners:
        state = play(state, w * 4)
    return state


def to_six_all(fmt=None):
    return love_games(initial_state(fmt), "AB" * 6)


# --------------------------------------------------------------------------- games


def test_love_game_rolls_server_and_ends():
    s = play(initial_state(), "AAAA")
    assert s.games == (1, 0) and s.points == (0, 0)
    assert s.server == "B" and s.sides == "swapped"
    assert s.score_str() == "1-0 (0-0)"


def test_deuce_and_advantage_sequence():
    s = play(initial_state(), "AAABBB")
    assert s.score_str() == "0-0 (40-40)"
    s = advance(s, "A")
    assert s.game_str() == "AD-40"
    s = advance(s, "B")
    assert s.game_str() == "40-40"
    s = advance(s, "B")
    assert s.game_str() == "40-AD"
    s = play(s, "AAA")
    assert s.games == (1, 0) and s.points_played == 12


def test_points_display():
    s = play(initial_state(), "ABA")  # 30-15
    assert s.score_str() == "0-0 (30-15)"


def test_no_ad_deciding_point():
    fmt = MatchFormat(no_ad=True)
    s = play(initial_state(fmt), "AAABBB")
    assert s.game_str() == "40-40"
    s = advance(s, "B")
    assert s.games == (0, 1) and s.points == (0, 0)


def test_example_score_string():
    s = love_games(initial_state(), "ABABABABAA" + "AABAB")  # set 1 6-4, set 2 3-2
    s = play(s, "AAB")
    assert s.score_str() == "6-4 3-2 (30-15)"


# --------------------------------------------------------------------------- tiebreak


def test_tiebreak_server_rotation_and_end_changes():
    s = to_six_all()
    assert s.in_tiebreak and s.games == (6, 6)
    due = s.game_server
    assert due == "A"  # 12 games played, A served game 1
    assert s.sides == "same"  # 6 odd games in the set -> 6 changes
    servers, sides = [], []
    for w in "ABABABABABAB":  # 6-6 in tiebreak points
        servers.append(s.server)
        sides.append(s.sides)
        s = advance(s, w)
    o = "B"
    assert servers == [due, o, o, due, due, o, o, due, due, o, o, due]
    # ends change after 6 and 12 tiebreak points
    assert sides[:6] == ["same"] * 6 and sides[6:] == ["swapped"] * 6
    assert s.sides == "same" and s.score_str() == "6-6 (TB 6-6)"
    s = play(s, "BB")  # win by two: 6-8
    assert s.sets[0].games == (6, 7) and str(s.sets[0]) == "6-7(6)"
    # next set: receiver of the first tiebreak point serves; ends change after game 13
    assert s.server == "B" and s.sides == "swapped"
    assert s.score_str() == "6-7(6) 0-0 (0-0)"


def test_tiebreak_needs_two_clear_points():
    s = play(to_six_all(), "AAAAAA" + "BBBBBB")  # 6-6
    s = advance(s, "A")
    assert s.in_tiebreak
    s = advance(s, "A")
    assert str(s.sets[0]) == "7-6(6)"


def test_advantage_set_without_tiebreak():
    s = to_six_all(MatchFormat(tiebreak_at=None))
    assert not s.in_tiebreak
    s = love_games(s, "AA")
    assert s.sets[0].games == (8, 6)


def test_match_tiebreak_in_lieu_of_third_set():
    fmt = MatchFormat.from_name("bo3-mtb10")
    s = love_games(initial_state(fmt), "A" * 6 + "B" * 6)
    assert s.is_match_tiebreak and s.in_tiebreak and s.set_index == 2
    assert s.score_str() == "6-0 0-6 [0-0]"
    first = s.server
    assert first == s.game_server
    s = play(s, "AAB")
    assert s.score_str() == "6-0 0-6 [2-1]"
    s = play(s, "BBBBBBBAAAAAA")  # 8-8
    assert not s.match_over and s.points == (8, 8)
    s = play(s, "AA")
    assert s.match_over and s.winner == "A"
    assert s.score_str() == "6-0 0-6 [10-8]"


def test_set_and_match_over():
    s = love_games(initial_state(), "A" * 6)
    assert s.sets_won == (1, 0) and s.set_index == 1 and not s.match_over
    s = love_games(s, "A" * 6)
    assert s.match_over and s.winner == "A" and s.score_str() == "6-0 6-0"
    with pytest.raises(ValueError):
        advance(s, "B")
    with pytest.raises(ValueError):
        reconstruct(["A"] * 49)  # 48 points finish 6-0 6-0


def test_best_of_one_and_five():
    assert love_games(initial_state(MatchFormat.from_name("bo1")), "A" * 6).match_over
    s = love_games(initial_state(MatchFormat.from_name("bo5")), "A" * 12)
    assert not s.match_over and s.sets_won == (2, 0)


def test_format_validation():
    with pytest.raises(ValueError):
        MatchFormat(first_server="C")
    with pytest.raises(ValueError):
        MatchFormat.from_name("bo7")
    with pytest.raises(ValueError):
        advance(initial_state(), "X")


# --------------------------------------------------------------------------- ends schedule


def _ends_after(game_winners, fmt=None):
    return [g["index"] + 1 for g in reconstruct_games(game_winners, fmt) if g["change_of_ends_after"]]


def test_change_of_ends_schedule_even_set():
    # set 1 6-4 (10 games, even): changes after games 1,3,5,7,9; none at the set end;
    # set 2 changes after its game 1 (match game 11) and 3 (13).
    assert _ends_after("ABABABABAA" + "AAA") == [1, 3, 5, 7, 9, 11, 13]


def test_change_of_ends_schedule_odd_set():
    # set 1 6-3 (9 games, odd): change at the set end (game 9) AND after game 1 of set 2.
    assert _ends_after("AABABABAA" + "BB") == [1, 3, 5, 7, 9, 10]


def test_servers_continue_across_sets():
    games = reconstruct_games("AABABABAA" + "BB")
    assert [g["server"] for g in games] == list("ABABABABA" + "BA")
    assert games[8]["set_over"] and games[9]["game_in_set"] == 1


def test_advance_game_requires_boundary():
    with pytest.raises(ValueError):
        advance_game(advance(initial_state(), "A"), "A")


# --------------------------------------------------------------------------- reconstruct


def test_reconstruct_fields():
    rec = reconstruct(list("AAAA" + "BBBB" + "AB"))
    assert len(rec) == 10
    assert [r["server"] for r in rec] == list("AAAA" + "BBBB" + "AA")
    assert [r["change_of_ends_before"] for r in rec] == [False] * 4 + [True] + [False] * 5
    assert [r["game_index"] for r in rec] == [0] * 4 + [1] * 4 + [2] * 2
    assert rec[3]["game_over"] and not rec[2]["game_over"]
    assert rec[0]["score_before"] == "0-0 (0-0)" and rec[0]["score_after"] == "0-0 (15-0)"
    assert rec[9]["score_after"] == "1-1 (15-15)"
    assert rec[4]["sides"] == "swapped" and rec[0]["set_index"] == 0
    assert not any(r["tiebreak"] for r in rec)


def test_reconstruct_tiebreak_flags_and_ends():
    winners = list("AAAABBBB" * 6) + list("ABABABAB") + ["A", "A", "A"]
    rec = reconstruct(winners)
    tb = [r for r in rec if r["tiebreak"]]
    assert len(tb) == 11 and tb[0]["game_index"] == 12
    # end change inside the tiebreak happens before its 7th point
    assert tb[6]["change_of_ends_before"] and not tb[5]["change_of_ends_before"]
    assert rec[-1]["set_over"] and rec[-1]["score_after"] == "7-6(4) 0-0 (0-0)"


# --------------------------------------------------------------------------- CSV parsing


def test_parse_score_after():
    assert parse_score_after("1-0") == [(1, 0)]
    assert parse_score_after("6-4 2-1") == [(6, 4), (2, 1)]
    assert parse_score_after("6-4 7-6(5) [10-8]") == [(6, 4), (7, 6), (10, 8)]
    assert parse_score_after("6-4 3-2 (30-15)") == [(6, 4), (3, 2)]
    assert parse_score_after("6-6 (TB 3-2)") == [(6, 6)]
    assert parse_score_after("6-4 3-2 (AD-40)") == [(6, 4), (3, 2)]
    assert parse_score_after("") == []
    with pytest.raises(ValueError):
        parse_score_after("six-four")


def test_winners_from_winner_csv(tmp_path):
    p = tmp_path / "m_winner.csv"
    p.write_text(ITF_CSV, encoding="utf-8")
    d = winners_from_winner_csv(str(p))
    assert d["names"] == {"A": "UCHINO, Takayuki", "B": "GOTO, Yusuke"}
    assert "".join(d["winners"]) == "AAAAAAABBAABBABAAB"
    assert d["at_time"][0] == 155.99 and d["score_after"][-1] == "12-6"


def test_winner_csv_name_code_conflict(tmp_path):
    p = tmp_path / "bad_winner.csv"
    p.write_text("point_no,atTime,segment_id,validation,winner,winner_name,score_after\n"
                 '1,1.0,,,0,"X, Y",1-0\n2,2.0,,,1,"X, Y",1-1\n', encoding="utf-8")
    with pytest.raises(ValueError):
        winners_from_winner_csv(str(p))


def _assert_itf(res):
    assert res["unit"] == "game" and res["n_rows"] == 18 and res["n_checked"] == 18
    assert res["mismatches"] == []
    assert res["final_score"] == "6-0 6-6" and not res["match_over"]
    tl = res["timeline"]
    assert [r["games_total_after"] for r in tl][-1] == (12, 6)
    assert tl[5]["score_after"] == "6-0 0-0" and tl[5]["set_over"]
    # TrimDeck serverOverride=0 on the last event == alternation: A serves the 6-6 tiebreak
    assert "".join(r["server"] for r in tl) == "AB" * 9


def test_check_winner_csv_itf_fixture(tmp_path):
    p = tmp_path / "itf_winner.csv"
    p.write_text(ITF_CSV, encoding="utf-8")
    _assert_itf(check_winner_csv(str(p)))


@pytest.mark.skipif(not ITF_REAL.exists(), reason="beta-tennis checkout not present")
def test_check_winner_csv_itf_real_file():
    _assert_itf(check_winner_csv(str(ITF_REAL)))


def test_check_winner_csv_point_unit(tmp_path):
    rows = ["point_no,atTime,segment_id,validation,winner,winner_name,score_after"]
    seq = "AAAA" + "BBBB" + "AB"
    after = ["0-0"] * 3 + ["1-0"] * 4 + ["1-1"] * 3
    for k, (w, sa) in enumerate(zip(seq, after)):
        code, name = ("0", "P, One") if w == "A" else ("1", "Q, Two")
        rows.append('%d,%d,,,%s,"%s",%s' % (k + 1, k, code, name, sa))
    p = tmp_path / "pts_winner.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = check_winner_csv(str(p))
    assert res["unit"] == "point" and res["mismatches"] == []
    res_bad = check_winner_csv(str(p), unit="game")
    assert res_bad["mismatches"]


# --------------------------------------------------------------------------- server check


def _simulate(first="A", n_points=160, seed=0, p_server=0.65):
    """Realistic winners (server wins 65%), cut before any tiebreak."""
    rng = random.Random(seed)
    s = initial_state(MatchFormat(first_server=first))
    winners, servers, game_ids = [], [], []
    while len(winners) < n_points and not s.match_over and not s.in_tiebreak:
        srv = s.server
        w = srv if rng.random() < p_server else S.other(srv)
        winners.append(w)
        servers.append(srv)
        game_ids.append(s.games_played)
        s = advance(s, w)
    return winners, servers, game_ids


@pytest.mark.parametrize("first", ["A", "B"])
def test_infer_perfect_sequence(first):
    winners, servers, game_ids = _simulate(first=first, seed=3)
    assert len(servers) > 60
    res = infer_first_server_and_check(servers, MatchFormat())
    assert res["mode"] == "segmentation"
    assert res["first_server"] == first
    assert res["agreement"] == 1.0 and res["disagreements"] == [] and res["n_none"] == 0
    assert res["implied_servers"] == servers
    true_bounds = [k for k in range(1, len(game_ids)) if game_ids[k] != game_ids[k - 1]]
    got_bounds = [g["start"] for g in res["games"][1:]]
    assert got_bounds == true_bounds
    assert res["cost_by_first_server"][S.other(first)] > 0


def test_infer_with_noise_and_missing():
    winners, servers, _ = _simulate(first="B", seed=7)
    n = len(servers)
    rng = random.Random(11)
    idx = list(range(n))
    rng.shuffle(idx)
    flipped = sorted(idx[: n // 10])
    missing = sorted(idx[n // 10: n // 10 + n // 12])
    detected = list(servers)
    for k in flipped:
        detected[k] = S.other(detected[k])
    for k in missing:
        detected[k] = None
    res = infer_first_server_and_check(detected, MatchFormat())
    assert res["first_server"] == "B"
    assert res["n_none"] == len(missing) and res["n_compared"] == n - len(missing)
    assert 0.85 <= res["agreement"] <= 0.97
    # A flip on the first/last point of a game is indistinguishable from a shifted game
    # boundary when winners are unknown, so the DP absorbs some flips (12/16 here).
    hit = set(flipped) & set(res["disagreements"])
    assert len(hit) >= 0.7 * len(flipped)
    assert len(res["disagreements"]) <= len(flipped)  # DP never does worse than the truth
    implied_ok = sum(1 for a, b in zip(res["implied_servers"], servers) if a == b) / n
    assert implied_ok >= 0.92


def test_infer_with_known_winners_handles_tiebreak():
    winners = list("AAAABBBB" * 6) + list("ABABABAB") + ["A", "A", "A"] + list("BBBB")
    truth = [r["server"] for r in reconstruct(winners, MatchFormat(first_server="B"))]
    detected = [None if k % 9 == 0 else s for k, s in enumerate(truth)]
    res = infer_first_server_and_check(detected, MatchFormat(), winners)
    assert res["mode"] == "winners" and res["first_server"] == "B"
    assert res["agreement"] == 1.0 and res["implied_servers"] == truth
    assert len(res["games"]) == 14 and res["games"][12]["end"] - res["games"][12]["start"] == 11
    with pytest.raises(ValueError):
        infer_first_server_and_check(detected[:-1], MatchFormat(), winners)


def test_infer_edge_cases():
    res = infer_first_server_and_check([None] * 6)
    assert res["agreement"] is None and res["n_compared"] == 0
    res = infer_first_server_and_check([])
    assert res["games"] == [] and res["agreement"] is None
    res = infer_first_server_and_check(["A", "A"])  # partial last game allowed
    assert res["agreement"] == 1.0
    with pytest.raises(ValueError):
        infer_first_server_and_check(["A", "A"], allow_partial_last_game=False)
    with pytest.raises(ValueError):
        infer_first_server_and_check(["near"])


# --------------------------------------------------------------------------- module hygiene + script


def test_score_module_is_stdlib_only():
    tree = ast.parse((ROOT / "src" / "analysis" / "score.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods <= set(sys.stdlib_module_names), mods - set(sys.stdlib_module_names)


def _load_script():
    path = ROOT / "scripts" / "eval" / "reconstruct_score.py"
    spec = importlib.util.spec_from_file_location("reconstruct_score_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_winners_csv_and_servers_json(tmp_path, capsys):
    mod = _load_script()
    csv_path = tmp_path / "itf_winner.csv"
    csv_path.write_text(ITF_CSV, encoding="utf-8")
    out = tmp_path / "out.json"
    assert mod.main(["--winners-csv", str(csv_path), "--json-out", str(out)]) == 0
    printed = capsys.readouterr().out
    printed.encode("ascii")  # ASCII-only status prints
    assert "final: 6-0 6-6" in printed
    data = json.loads(out.read_text())
    assert data["score_check"]["mismatches"] == []

    pts = [{"server": "near", "near_id": "A", "far_id": "B"}] * 4 + \
          [{"server": "far", "near_id": "A", "far_id": "B"}] * 4 + [{"server": None}] + ["A"]
    sj = tmp_path / "points.json"
    sj.write_text(json.dumps({"points": pts}))
    assert mod.load_servers(str(sj)) == list("AAAABBBB") + [None, "A"]
    out2 = tmp_path / "out2.json"
    assert mod.main(["--servers-json", str(sj), "--json-out", str(out2)]) == 0
    res = json.loads(out2.read_text())["server_check"]
    assert res["first_server"] == "A" and res["agreement"] == 1.0
    capsys.readouterr().out.encode("ascii")


def test_reconstruct_sets_after_keeps_tiebreak_annotation():
    winners = list("AAAABBBB" * 6) + list("AAAAAAA") + list("AAAA")
    rec = reconstruct(winners)
    assert rec[-1]["sets_after"] == "7-6(0) 1-0"
    assert parse_score_after(rec[-1]["sets_after"]) == [(7, 6), (1, 0)]
