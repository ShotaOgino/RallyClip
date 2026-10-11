"""Tennis score state machine, score reconstruction and server-sequence checking.

Pure Python, standard library only (roadmap Phase 4, label-free part).

Players are always the labels ``"A"`` and ``"B"``. Who is on the near / far end of
the court is Phase 1's job; this module only tracks whether the ends are the
``"same"`` as at the start of the match or ``"swapped"``.

Rules implemented (ITF):

* Games: 0/15/30/40, deuce, advantage; or no-ad (deciding point at 40-40).
* Sets: first to ``games_per_set`` by two; tiebreak at ``tiebreak_at`` all (first to
  ``tiebreak_points`` by two). Optional match tiebreak (first to
  ``match_tiebreak_points`` by two) in lieu of the deciding set.
* Service: alternates every game, continuing across sets; a tiebreak counts as one
  game. In a tiebreak the due server serves point 1, then two points each.
* Ends: change after every odd game of a set (1st, 3rd, ...; a set-ending tiebreak
  is game 13, so ends change after it, and a set with an even number of games
  changes after game 1 of the next set). In a tiebreak, also every 6 points.

Public API: :class:`MatchFormat`, :class:`ScoreState`, :func:`initial_state`,
:func:`advance`, :func:`advance_game`, :func:`reconstruct`,
:func:`reconstruct_games`, :func:`infer_first_server_and_check`,
:func:`parse_score_after`, :func:`winners_from_winner_csv`, :func:`check_winner_csv`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

PLAYERS = ("A", "B")
_POINT_NAMES = ("0", "15", "30", "40")


def other(player: str) -> str:
    """Return the opponent label."""
    return "B" if _check_player(player) == "A" else "A"


def _check_player(player: object) -> str:
    if player not in PLAYERS:
        raise ValueError(f"player must be 'A' or 'B', got {player!r}")
    return player  # type: ignore[return-value]


# --------------------------------------------------------------------------- format


@dataclass(frozen=True)
class MatchFormat:
    """Match rules. Defaults: best of 3 tiebreak sets, advantage scoring, A serves first."""

    sets_to_win: int = 2
    games_per_set: int = 6
    tiebreak_at: Optional[int] = 6  # None = advantage set (no tiebreak)
    tiebreak_points: int = 7
    final_set_match_tiebreak: bool = False  # 10-point match tiebreak instead of the deciding set
    match_tiebreak_points: int = 10
    no_ad: bool = False  # deciding point at deuce
    first_server: str = "A"

    def __post_init__(self) -> None:
        _check_player(self.first_server)
        if self.sets_to_win < 1 or self.games_per_set < 1:
            raise ValueError("sets_to_win and games_per_set must be >= 1")
        if self.tiebreak_at is not None and self.tiebreak_at < 1:
            raise ValueError("tiebreak_at must be >= 1 or None")
        if self.tiebreak_points < 1 or self.match_tiebreak_points < 1:
            raise ValueError("tiebreak point targets must be >= 1")

    @classmethod
    def from_name(cls, name: str, **overrides) -> "MatchFormat":
        """Presets: ``bo3`` (default), ``bo3-mtb10`` (match tiebreak for set 3),
        ``bo1`` (one tiebreak set), ``bo5``. Keyword overrides are applied on top."""
        presets = {
            "bo3": {},
            "bo3-mtb10": {"final_set_match_tiebreak": True},
            "bo1": {"sets_to_win": 1},
            "bo5": {"sets_to_win": 3},
        }
        if name not in presets:
            raise ValueError(f"unknown format {name!r}; choose from {sorted(presets)}")
        return cls(**{**presets[name], **overrides})


# --------------------------------------------------------------------------- state


@dataclass(frozen=True)
class SetResult:
    games: Tuple[int, int]
    tiebreak: Optional[Tuple[int, int]] = None  # tiebreak points when known
    match_tiebreak: bool = False

    @property
    def winner(self) -> str:
        return "A" if self.games[0] > self.games[1] else "B"

    def __str__(self) -> str:
        if self.match_tiebreak:
            if self.tiebreak is None:
                return "[%s]" % ("1-0" if self.games[0] else "0-1")
            return "[%d-%d]" % self.tiebreak
        base = "%d-%d" % self.games
        if self.tiebreak is not None:
            base += "(%d)" % min(self.tiebreak)
        return base


@dataclass(frozen=True)
class ScoreState:
    """Immutable score. ``points`` holds raw point counts of the current game (or
    tiebreak points while ``in_tiebreak``). ``game_server`` is the server of the
    current game (the due server of a tiebreak); use :attr:`server` for the next point."""

    fmt: MatchFormat = field(default_factory=MatchFormat)
    sets: Tuple[SetResult, ...] = ()
    games: Tuple[int, int] = (0, 0)
    points: Tuple[int, int] = (0, 0)
    game_server: str = "A"
    swapped: bool = False
    games_played: int = 0  # games completed in the match (a tiebreak counts as one)
    points_played: int = 0
    winner: Optional[str] = None

    # -- derived -----------------------------------------------------------------
    @property
    def match_over(self) -> bool:
        return self.winner is not None

    @property
    def set_index(self) -> int:
        """0-based index of the set in play (== number of completed sets)."""
        return len(self.sets)

    @property
    def sets_won(self) -> Tuple[int, int]:
        a = sum(1 for s in self.sets if s.winner == "A")
        return (a, len(self.sets) - a)

    @property
    def is_match_tiebreak(self) -> bool:
        """True when the set in play is a match tiebreak (deciding set replaced)."""
        if not self.fmt.final_set_match_tiebreak or self.match_over:
            return False
        last = self.fmt.sets_to_win - 1
        return self.sets_won == (last, last)

    @property
    def in_tiebreak(self) -> bool:
        if self.match_over:
            return False
        if self.is_match_tiebreak:
            return True
        tb = self.fmt.tiebreak_at
        return tb is not None and self.games == (tb, tb)

    @property
    def server(self) -> str:
        """Who serves the next point."""
        if self.in_tiebreak:
            k = sum(self.points)
            return self.game_server if ((k + 1) // 2) % 2 == 0 else other(self.game_server)
        return self.game_server

    @property
    def sides(self) -> str:
        """``"same"`` or ``"swapped"`` ends relative to the start of the match."""
        return "swapped" if self.swapped else "same"

    def game_str(self) -> str:
        """Current game score from A's perspective, e.g. ``"30-15"``, ``"AD-40"``."""
        a, b = self.points
        if self.in_tiebreak:
            return "%d-%d" % (a, b)
        if a >= 3 and b >= 3:
            if a == b:
                return "40-40"
            return "AD-40" if a > b else "40-AD"
        return "%s-%s" % (_POINT_NAMES[a], _POINT_NAMES[b])

    def sets_str(self) -> str:
        """Sets and current-set games only, e.g. ``"6-4 3-2"`` (no point score)."""
        parts = [str(s) for s in self.sets]
        if not self.match_over and not self.is_match_tiebreak:
            parts.append("%d-%d" % self.games)
        return " ".join(parts)

    def score_str(self) -> str:
        """Full score, A first: ``"6-4 3-2 (30-15)"``; tiebreak ``"6-6 (TB 3-2)"``;
        match tiebreak ``"6-4 4-6 [3-2]"``; finished match ``"6-4 7-6(5)"``."""
        if self.match_over:
            return self.sets_str()
        if self.is_match_tiebreak:
            head = " ".join(str(s) for s in self.sets)
            return (head + " " if head else "") + "[%d-%d]" % self.points
        if self.in_tiebreak:
            return "%s (TB %s)" % (self.sets_str(), self.game_str())
        return "%s (%s)" % (self.sets_str(), self.game_str())


def initial_state(fmt: Optional[MatchFormat] = None) -> ScoreState:
    fmt = fmt or MatchFormat()
    return ScoreState(fmt=fmt, game_server=fmt.first_server)


def _idx(player: str) -> int:
    return 0 if _check_player(player) == "A" else 1


def _bump(pair: Tuple[int, int], i: int) -> Tuple[int, int]:
    return (pair[0] + 1, pair[1]) if i == 0 else (pair[0], pair[1] + 1)


def _finish_game(state: ScoreState, winner: str, tb_points: Optional[Tuple[int, int]]) -> ScoreState:
    """Award the current game (or tiebreak) to ``winner`` and roll games/sets/ends."""
    i = _idx(winner)
    fmt = state.fmt
    was_tiebreak = state.in_tiebreak
    was_mtb = state.is_match_tiebreak
    games = _bump(state.games, i)
    if was_mtb:
        set_won = True
    elif was_tiebreak:
        set_won = True
    else:
        set_won = games[i] >= fmt.games_per_set and games[i] - games[1 - i] >= 2
    games_in_set = sum(games)  # a tiebreak counts as one game; a match tiebreak is game 1
    sets = state.sets
    winner_of_match: Optional[str] = None
    if set_won:
        sets = sets + (SetResult(games=games, tiebreak=tb_points if was_tiebreak else None,
                                 match_tiebreak=was_mtb),)
        a = sum(1 for s in sets if s.winner == "A")
        if a >= fmt.sets_to_win:
            winner_of_match = "A"
        elif len(sets) - a >= fmt.sets_to_win:
            winner_of_match = "B"
        games = (0, 0)
    swapped = state.swapped
    if winner_of_match is None and games_in_set % 2 == 1:
        swapped = not swapped
    return replace(
        state,
        sets=sets,
        games=games,
        points=(0, 0),
        game_server=other(state.game_server),
        swapped=swapped,
        games_played=state.games_played + 1,
        winner=winner_of_match,
    )


def advance(state: ScoreState, winner: str) -> ScoreState:
    """Return the state after ``winner`` ("A"/"B") wins the next point."""
    if state.match_over:
        raise ValueError("match is already over (%s)" % state.score_str())
    i = _idx(winner)
    pts = _bump(state.points, i)
    played = replace(state, points_played=state.points_played + 1)
    if state.in_tiebreak:
        target = state.fmt.match_tiebreak_points if state.is_match_tiebreak else state.fmt.tiebreak_points
        if pts[i] >= target and pts[i] - pts[1 - i] >= 2:
            return _finish_game(played, winner, pts)
        swapped = state.swapped
        if sum(pts) % 6 == 0:
            swapped = not swapped
        return replace(played, points=pts, swapped=swapped)
    if state.fmt.no_ad:
        won = pts[i] >= 4
    else:
        won = pts[i] >= 4 and pts[i] - pts[1 - i] >= 2
    if won:
        return _finish_game(played, winner, None)
    return replace(played, points=pts)


def advance_game(state: ScoreState, winner: str) -> ScoreState:
    """Award a whole game (or a whole tiebreak) to ``winner``; for game-level logs
    such as the beta-tennis TrimDeck ``gameEvents``. Tiebreak points are unknown, so
    a tiebreak set prints as ``"7-6"`` and the intra-tiebreak end changes are not
    tracked (:attr:`ScoreState.sides` after a tiebreak is therefore not reliable)."""
    if state.match_over:
        raise ValueError("match is already over (%s)" % state.score_str())
    if state.points != (0, 0):
        raise ValueError("advance_game needs a game boundary (points are %s)" % (state.points,))
    _check_player(winner)
    return _finish_game(state, winner, None)


# --------------------------------------------------------------------------- reconstruction


def reconstruct(winners: Sequence[str], fmt: Optional[MatchFormat] = None) -> List[Dict]:
    """Replay per-point winners. One dict per point::

        {index, winner, server, sides, score_before, score_after, sets_after,
         change_of_ends_before, game_index, set_index, tiebreak,
         game_over, set_over, match_over}

    ``game_index`` counts games in the whole match (0-based; a tiebreak is one game).
    ``change_of_ends_before`` is True when the players changed ends between the
    previous point and this one. Raises ValueError for a point after match end."""
    state = initial_state(fmt)
    out: List[Dict] = []
    prev_swapped = state.swapped
    for k, w in enumerate(winners):
        if state.match_over:
            raise ValueError("point %d played after the match ended (%s)" % (k, state.score_str()))
        before = state
        after = advance(before, w)
        out.append({
            "index": k,
            "winner": w,
            "server": before.server,
            "sides": before.sides,
            "score_before": before.score_str(),
            "score_after": after.score_str(),
            "sets_after": after.sets_str(),
            "change_of_ends_before": k > 0 and before.swapped != prev_swapped,
            "game_index": before.games_played,
            "set_index": before.set_index,
            "tiebreak": before.in_tiebreak,
            "game_over": after.games_played > before.games_played,
            "set_over": after.set_index > before.set_index,
            "match_over": after.match_over,
        })
        prev_swapped = before.swapped
        state = after
    return out


def reconstruct_games(game_winners: Sequence[str], fmt: Optional[MatchFormat] = None) -> List[Dict]:
    """Replay per-game winners (one entry per game; a tiebreak is one entry). One dict
    per game: {index, winner, server, set_index, game_in_set, tiebreak, score_before,
    score_after, games_total_after, change_of_ends_after, set_over, match_over}.
    ``score_*`` are set strings ("6-0 2-1"); ``games_total_after`` is (A, B) games
    summed over the match; ``server`` is the game's server (the due server of a tiebreak)."""
    state = initial_state(fmt)
    out: List[Dict] = []
    total = (0, 0)
    for k, w in enumerate(game_winners):
        if state.match_over:
            raise ValueError("game %d played after the match ended (%s)" % (k, state.score_str()))
        before = state
        after = advance_game(before, w)
        total = _bump(total, _idx(w))
        out.append({
            "index": k,
            "winner": w,
            "server": before.game_server,
            "set_index": before.set_index,
            "game_in_set": sum(before.games) + 1,
            "tiebreak": before.in_tiebreak,
            "score_before": before.sets_str(),
            "score_after": after.sets_str(),
            "games_total_after": total,
            "change_of_ends_after": after.swapped != before.swapped,
            "set_over": after.set_index > before.set_index,
            "match_over": after.match_over,
        })
        state = after
    return out


# --------------------------------------------------------------------------- server check


def _segments_from_game_index(game_ids: Sequence[int], servers: Sequence[str]) -> List[Dict]:
    games: List[Dict] = []
    for k, g in enumerate(game_ids):
        if not games or games[-1]["game_index"] != g:
            games.append({"game_index": g, "start": k, "end": k + 1, "server": servers[k]})
        else:
            games[-1]["end"] = k + 1
    return games


def _agreement(detected: Sequence[Optional[str]], implied: Sequence[str]) -> Tuple[Optional[float], int, List[int]]:
    compared = [(k, d) for k, d in enumerate(detected) if d is not None]
    bad = [k for k, d in compared if d != implied[k]]
    if not compared:
        return None, 0, bad
    return (len(compared) - len(bad)) / len(compared), len(compared), bad


def _segment_dp(detected: Sequence[Optional[str]], first: str, min_len: int, max_len: int,
                partial_first: bool, partial_last: bool) -> Tuple[Optional[int], List[Tuple[int, int]]]:
    """Min-disagreement split of points into consecutive games with alternating
    servers starting with ``first``. Returns (cost, [(start, end), ...])."""
    n = len(detected)
    # mis[s][k] = number of non-None detections in [0, k) that are != s
    mis = {s: [0] * (n + 1) for s in PLAYERS}
    for k, d in enumerate(detected):
        for s in PLAYERS:
            mis[s][k + 1] = mis[s][k] + (1 if d is not None and d != s else 0)
    inf = float("inf")
    # dp[i][p]: best cost covering [0, i) with whole games, next game parity p
    dp = [[inf, inf] for _ in range(n + 1)]
    back: List[List[Optional[int]]] = [[None, None] for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n):
        for p in (0, 1):
            if dp[i][p] == inf:
                continue
            s = first if p == 0 else other(first)
            for length in range(1, max_len + 1):
                j = i + length
                if j > n:
                    break
                short_ok = (partial_first and i == 0) or (partial_last and j == n)
                if length < min_len and not short_ok:
                    continue
                c = dp[i][p] + mis[s][j] - mis[s][i]
                if c < dp[j][1 - p]:
                    dp[j][1 - p] = c
                    back[j][1 - p] = i
    p_end = 0 if dp[n][0] <= dp[n][1] else 1
    if dp[n][p_end] == inf:
        return None, []
    segs: List[Tuple[int, int]] = []
    j, p = n, p_end
    while j > 0:
        i = back[j][p]
        assert i is not None
        segs.append((i, j))
        j, p = i, 1 - p
    segs.reverse()
    return int(dp[n][p_end]), segs


def infer_first_server_and_check(
    servers_detected: Sequence[Optional[str]],
    fmt: Optional[MatchFormat] = None,
    winners: Optional[Sequence[str]] = None,
    *,
    min_game_points: int = 4,
    max_game_points: int = 30,
    allow_partial_first_game: bool = False,
    allow_partial_last_game: bool = True,
) -> Dict:
    """Score a detected per-point server sequence ("A"/"B"/None) against the rules.

    * ``winners`` given: replay the score for both first-server choices; the implied
      server sequence is exact (tiebreak rotation included).
    * ``winners`` unknown: a dynamic programme places game boundaries (games of
      ``min_game_points``..``max_game_points`` points, server constant within a game,
      alternating across games) to minimise disagreements. Tiebreaks are not
      modelled in this mode (a tiebreak looks like one long game with mixed servers).

    Returns ``{mode, first_server, games: [{game_index, start, end, server}],
    implied_servers, agreement, n_compared, n_none, disagreements, cost_by_first_server}``.
    ``agreement`` is over non-None detections (None if there are none); ``end`` is
    exclusive. ``fmt.first_server`` is ignored: both choices are tried, A wins ties."""
    fmt = fmt or MatchFormat()
    detected = list(servers_detected)
    for d in detected:
        if d is not None:
            _check_player(d)
    n = len(detected)
    results = {}
    if winners is not None:
        if len(winners) != n:
            raise ValueError("winners (%d) and servers_detected (%d) differ in length" % (len(winners), n))
        for first in PLAYERS:
            rec = reconstruct(winners, replace(fmt, first_server=first))
            implied = [r["server"] for r in rec]
            games = _segments_from_game_index([r["game_index"] for r in rec], implied)
            _, _, bad = _agreement(detected, implied)
            results[first] = (len(bad), implied, games)
        mode = "winners"
    else:
        if min_game_points < 1 or max_game_points < min_game_points:
            raise ValueError("need 1 <= min_game_points <= max_game_points")
        for first in PLAYERS:
            cost, segs = _segment_dp(detected, first, min_game_points, max_game_points,
                                     allow_partial_first_game, allow_partial_last_game)
            if cost is None:
                continue
            implied: List[str] = []
            games = []
            for g, (i, j) in enumerate(segs):
                s = first if g % 2 == 0 else other(first)
                implied.extend([s] * (j - i))
                games.append({"game_index": g, "start": i, "end": j, "server": s})
            results[first] = (cost, implied, games)
        if not results:
            raise ValueError("no valid game segmentation for %d points with min_game_points=%d"
                             % (n, min_game_points))
        mode = "segmentation"
    best = min(results, key=lambda f: (results[f][0], PLAYERS.index(f)))
    _, implied, games = results[best]
    agreement, n_compared, bad = _agreement(detected, implied)
    return {
        "mode": mode,
        "first_server": best,
        "games": games,
        "implied_servers": implied,
        "agreement": agreement,
        "n_compared": n_compared,
        "n_none": n - n_compared,
        "disagreements": bad,
        "cost_by_first_server": {f: results[f][0] for f in results},
    }


# --------------------------------------------------------------------------- beta-tennis CSV

_SET_TOKEN = re.compile(r"^\[?(\d+)-(\d+)\]?(?:\(\d+\))?$")


def parse_score_after(s: str) -> List[Tuple[int, int]]:
    """Parse ``"1-0"``, ``"6-4 2-1"``, ``"6-4 7-6(5) [10-8]"`` into per-set (A, B)
    tuples. Tiebreak annotations ``(5)`` are dropped; a trailing point score in
    parentheses (``"(30-15)"``, ``"(TB 3-2)"``) is ignored; ``[x-y]`` (match
    tiebreak) is returned as ``(x, y)``. Empty string -> ``[]``."""
    text = re.sub(r"\((?:TB\s*)?[0-9AD]+-[0-9AD]+\)", " ", (s or "").strip())
    out: List[Tuple[int, int]] = []
    for tok in text.split():
        m = _SET_TOKEN.match(tok)
        if not m:
            raise ValueError("cannot parse score token %r in %r" % (tok, s))
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def winners_from_winner_csv(path: str) -> Dict:
    """Read a beta-tennis ``Swing/VideoEditing/*_winner.csv`` (columns point_no,
    atTime, segment_id, validation, winner, winner_name, score_after).

    ``winner`` 0/1 is mapped to "A"/"B" via ``winner_name``: A is the name that
    appears with code 0 (TrimDeck ``team0``, the top scoreboard row). A name seen with
    both codes raises ValueError. Note: TrimDeck rows are *game* events
    (``gameEvents.gameWinner``), not points; see :func:`check_winner_csv`.

    Returns ``{winners, names: {"A": str|None, "B": str|None}, at_time, score_after, point_no}``."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("winner") or "").strip():
                continue
            rows.append(r)
    code_to_name: Dict[str, str] = {}
    name_to_code: Dict[str, str] = {}
    for r in rows:
        code = r["winner"].strip()
        if code not in ("0", "1"):
            raise ValueError("winner must be 0 or 1, got %r" % code)
        name = (r.get("winner_name") or "").strip()
        if not name:
            continue
        if name_to_code.setdefault(name, code) != code:
            raise ValueError("name %r appears with both winner codes" % name)
        if code_to_name.setdefault(code, name) != name:
            raise ValueError("winner code %s maps to two names: %r, %r" % (code, code_to_name[code], name))
    label = {"0": "A", "1": "B"}
    return {
        "winners": [label[r["winner"].strip()] for r in rows],
        "names": {"A": code_to_name.get("0"), "B": code_to_name.get("1")},
        "at_time": [float(r["atTime"]) if (r.get("atTime") or "").strip() else None for r in rows],
        "score_after": [(r.get("score_after") or "").strip() for r in rows],
        "point_no": [int(r["point_no"]) if (r.get("point_no") or "").strip() else None for r in rows],
    }


def _detect_unit(score_after: Sequence[str]) -> str:
    """"game" when every row's total games grows by exactly one, else "point"."""
    prev = 0
    for s in score_after:
        try:
            tot = sum(a + b for a, b in parse_score_after(s))
        except ValueError:
            return "point"
        if tot != prev + 1:
            return "point"
        prev = tot
    return "game" if score_after else "point"


def _score_matches(parsed: List[Tuple[int, int]], sets_str: str, total: Tuple[int, int]) -> Optional[str]:
    """How ``parsed`` matches the reconstruction: "per_set", "cumulative" or None."""
    rec = parse_score_after(sets_str)
    trimmed = rec[:-1] if len(rec) > 1 and rec[-1] == (0, 0) else rec
    if parsed in (rec, trimmed):
        return "per_set"
    if len(parsed) == 1 and parsed[0] == total:
        return "cumulative"
    return None


def check_winner_csv(path: str, fmt: Optional[MatchFormat] = None, unit: str = "auto") -> Dict:
    """Replay a winner CSV and compare its ``score_after`` column row by row.

    ``unit`` is "game" (TrimDeck game events), "point", or "auto" (game when the
    column's total games grow by one per row). A row matches when ``score_after``
    equals the reconstructed per-set games or, for a single ``a-b`` token, the games
    summed over the match (TrimDeck's CSV export is cumulative)."""
    data = winners_from_winner_csv(path)
    if unit == "auto":
        unit = _detect_unit(data["score_after"])
    if unit not in ("game", "point"):
        raise ValueError("unit must be game, point or auto")
    rows: List[Dict] = []
    if unit == "game":
        for rec, csv_score in zip(reconstruct_games(data["winners"], fmt), data["score_after"]):
            rows.append({**rec, "csv_score_after": csv_score})
    else:
        total = (0, 0)
        for rec, csv_score in zip(reconstruct(data["winners"], fmt), data["score_after"]):
            if rec["game_over"]:
                total = _bump(total, _idx(rec["winner"]))
            rows.append({**rec, "score_after": rec["sets_after"], "games_total_after": total,
                         "csv_score_after": csv_score})
    mismatches = []
    modes = set()
    for r in rows:
        if not r["csv_score_after"]:
            r["csv_match"] = None
            continue
        how = _score_matches(parse_score_after(r["csv_score_after"]), r["score_after"], r["games_total_after"])
        r["csv_match"] = how
        if how is None:
            mismatches.append(r["index"])
        else:
            modes.add(how)
    last = rows[-1] if rows else None
    return {
        "unit": unit,
        "names": data["names"],
        "n_rows": len(rows),
        "n_checked": sum(1 for r in rows if r["csv_match"] is not None or r["index"] in mismatches),
        "mismatches": mismatches,
        "match_modes": sorted(modes),
        "final_score": last["score_after"] if last else "",
        "match_over": bool(last and last["match_over"]),
        "timeline": rows,
        "at_time": data["at_time"],
    }
