"""Serve side per point: "deuce" or "ad", from two independent sources (roadmap Phase 1).

numpy only (no torch / cv2 / av), like ``analysis.players``.

(a) VISUAL - where the players stand at the serve, in court metres. The court
    homography (``preprocessing.court_lines.compute_court_geometry`` ->
    ``homography_px_to_m``, reference pixels -> metres; origin at the net centre,
    x to the camera's right, y away from the camera) maps each player's feet (ankle
    midpoint, else box bottom-centre) onto the court; x = 0 is the centre line.

    Each player's deuce court is on their own right-hand side when facing the net,
    and the serve goes diagonally from the server's deuce court into the receiver's
    deuce court. Facing the net, the NEAR player's right is the camera's right and the
    FAR player's right is the camera's left, so for EITHER role:

        near player on x > 0 (camera right)  -> deuce point
        far player on x < 0 (camera left)    -> deuce point

    The role only changes how far from the centre line the player stands (the server
    stands ~0.3-1.5 m from the centre mark, the receiver ~3-4 m out near the singles
    sideline), so it sets the confidence scale, and which track is used: the server's
    first, else the receiver's (``source: "receiver"`` - the usual case for far serves
    on the beta-tennis footage, whose far track is absent in ~98 % of in-point frames).
    With the server unknown the near player is read with the receiver's (wider) scale
    (``source: "player"``).

    Window: ``pre_s`` before to ``post_s`` after the serve moment = the server's first
    overhead raise (``detect_server`` evidence), else the point start plus the match's
    median raise offset (``default_serve_offset_s`` when no point has a raise).

(b) SCORE - parity of the point's index within its game: even (0, 2, 4, ...) -> deuce,
    odd -> ad, in tiebreaks too (the parity is per tiebreak point). The game of each
    point comes from a winners list (``analysis.score.reconstruct`` ``game_index``), a
    scoreboard log, or label-free from the server sequence
    (``analysis.score.infer_first_server_and_check`` segmentation). A point that starts
    less than ``same_point_gap_s`` after the previous one ended is the second serve of
    the same point (a first-serve fault clipped / decoded separately) and keeps its
    index. Missed points shift the parity for the rest of the game, so the score
    confidence decays with the index.

``combine`` gives the final answer per point (visual when its confidence >=
``min_visual_confidence``, else score) and flags disagreement; ``cross_check`` reports
the agreement per match.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.players import L_ANKLE, R_ANKLE, PlayerTracks, PointLike, SideTrack, _point_bounds

DEUCE, AD = "deuce", "ad"
SERVE_SIDES = (DEUCE, AD)
VISUAL_SOURCES = ("server", "receiver", "player")
SCORE_SOURCE = "score"
ROW_KEYS = (
    "serve_side", "serve_side_confidence", "serve_side_source", "serve_side_score",
    "serve_side_agree", "serve_side_visual", "serve_side_visual_confidence", "serve_side_visual_source",
)


@dataclass(frozen=True)
class ServeSideConfig:
    """Heuristic knobs (not model-contract values; set on beta-tennis 2R, 2026-08-24)."""

    kp_conf_min: float = 0.3
    pre_s: float = 1.0                  # window starts this long before the serve moment
    post_s: float = 0.5                 # ... and ends this long after it
    default_serve_offset_s: float = 1.3  # serve moment after the point start without any raise
    min_frames: int = 3                 # fewer feet observations -> no visual estimate
    server_scale_m: float = 0.25        # position score = 1 - exp(-|x| / scale)
    receiver_scale_m: float = 1.0
    min_depth_m: float = 9.0            # |court y| below this: not at a baseline -> depth_penalty
    depth_penalty: float = 0.5
    min_visual_confidence: float = 0.5  # combine(): the visual answer wins at/above this
    score_confidence: float = 0.9       # game index from winners / a scoreboard log
    segmentation_confidence: float = 0.6  # game index from the detected server sequence
    score_decay: float = 0.95           # score confidence x decay ** point_in_game
    same_point_gap_s: float = 5.0       # gap below this = second serve of the same point


DEFAULT_SERVE_SIDE_CONFIG = ServeSideConfig()


# ------------------------------------------------------------------ geometry


def court_homography(geometry: Any) -> Optional[np.ndarray]:
    """3x3 pixels -> court metres from a geometry dict (``homography_px_to_m``), its
    JSON string, a ``preprocessing.netline_slots.CourtGeometry`` (``px_to_m``) or a 3x3
    array. None when absent / malformed."""
    if geometry is None:
        return None
    if isinstance(geometry, (str, bytes)):
        try:
            geometry = json.loads(geometry)
        except ValueError:
            return None
    if isinstance(geometry, Mapping):
        geometry = geometry.get("homography_px_to_m")
    elif hasattr(geometry, "px_to_m"):
        geometry = geometry.px_to_m
    if geometry is None:
        return None
    h = np.asarray(geometry, dtype=np.float64)
    return h if h.shape == (3, 3) and np.isfinite(h).all() else None


def geometry_to_dict(geometry: Any) -> Optional[Dict[str, Any]]:
    """JSON-safe geometry dict (the ``geometry_json`` npz key): a geometry dict as is,
    or ``{net_line_px, homography_px_to_m}`` from a CourtGeometry. None when absent."""
    if geometry is None:
        return None
    if isinstance(geometry, (str, bytes)):
        try:
            geometry = json.loads(geometry)
        except ValueError:
            return None
    if isinstance(geometry, Mapping):
        return json.loads(json.dumps(dict(geometry)))
    h = court_homography(geometry)
    if h is None:
        return None
    out: Dict[str, Any] = {"homography_px_to_m": h.tolist()}
    if hasattr(geometry, "net_a") and hasattr(geometry, "net_b"):
        out["net_line_px"] = [np.asarray(geometry.net_a, float).tolist(), np.asarray(geometry.net_b, float).tolist()]
    return out


def to_court_m(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(m, 2) pixels -> (m, 2) court metres (NaN rows stay NaN)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    p = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1) @ np.asarray(h, dtype=np.float64).T
    with np.errstate(all="ignore"):
        return p[:, :2] / p[:, 2:3]


def feet_px(side: SideTrack, idx: np.ndarray, kp_conf_min: float = 0.3) -> np.ndarray:
    """(len(idx), 2) feet in pixels: ankle midpoint (one ankle if only one is confident),
    else the box bottom-centre; NaN where the player is absent."""
    idx = np.asarray(idx, dtype=int)
    out = np.full((idx.size, 2), np.nan)
    if idx.size == 0:
        return out
    present = np.asarray(side.present, dtype=bool)[idx]
    kp = np.asarray(side.kp, dtype=np.float64)[idx][:, (L_ANKLE, R_ANKLE)]
    conf = np.asarray(side.kpconf, dtype=np.float64)[idx][:, (L_ANKLE, R_ANKLE)]
    ok = np.isfinite(kp).all(axis=-1) & ~(np.isfinite(conf) & (conf < kp_conf_min))
    with np.errstate(invalid="ignore"):
        ankle = np.where(ok[..., None], kp, 0.0).sum(axis=1) / ok.sum(axis=1)[:, None]
    box = np.asarray(side.box, dtype=np.float64)[idx]
    bottom = np.stack([(box[:, 0] + box[:, 2]) / 2.0, box[:, 3]], axis=1)
    out = np.where(ok.any(axis=1)[:, None], ankle, bottom)
    out[~present] = np.nan
    return out


def side_for_x(player: str, x: float) -> Optional[str]:
    """Serve side implied by a player's court x (metres; both roles, see module doc)."""
    if not np.isfinite(x) or x == 0.0:
        return None
    right_of_camera = x > 0.0
    return DEUCE if right_of_camera == (player == "near") else AD


# ------------------------------------------------------------------ (a) visual


def _position_estimate(tracks: PlayerTracks, player: str, idx: np.ndarray, h: np.ndarray,
                       scale_m: float, cfg: ServeSideConfig) -> Optional[Dict[str, Any]]:
    feet = feet_px(tracks.side(player), idx, cfg.kp_conf_min)
    feet = feet[np.isfinite(feet).all(axis=1)]
    if feet.shape[0] < cfg.min_frames:
        return None
    court = to_court_m(h, feet)
    court = court[np.isfinite(court).all(axis=1)]
    if court.shape[0] < cfg.min_frames:
        return None
    x, y = float(np.median(court[:, 0])), float(np.median(court[:, 1]))
    side = side_for_x(player, x)
    if side is None:
        return None
    agree = float(np.mean(np.sign(court[:, 0]) == np.sign(x)))
    position = 1.0 - math.exp(-abs(x) / max(1e-6, scale_m))
    depth = 1.0 if abs(y) >= cfg.min_depth_m else cfg.depth_penalty
    conf = position * max(0.0, 2.0 * agree - 1.0) * depth
    return {"player": player, "side": side, "confidence": round(conf, 3), "frames": int(court.shape[0]),
            "x_m": round(x, 2), "y_m": round(y, 2), "agree_frac": round(agree, 3)}


def serve_side(
    point: PointLike,
    tracks: PlayerTracks,
    geometry: Any,
    server_side: Optional[str],
    *,
    serve_time: Optional[float] = None,
    cfg: ServeSideConfig = DEFAULT_SERVE_SIDE_CONFIG,
) -> Dict[str, Any]:
    """Visual serve side of one point: {"side", "confidence", "source", "evidence"}.

    ``server_side`` "near" / "far" (``detect_server``) or None; ``serve_time``
    (absolute seconds) defaults to the point start + ``default_serve_offset_s``.
    The server's track is read first (scale ``server_scale_m``), the receiver's
    otherwise (``receiver_scale_m``, source "receiver"). When both are seen they vote:
    agreeing -> 1 - (1 - c1)(1 - c2) under the server's source; disagreeing -> the
    more confident one, confidence = the difference. Without geometry, or with fewer
    than ``min_frames`` feet observations, side None / confidence 0.
    """
    t_start, _t_end = _point_bounds(point)
    t_serve = float(serve_time) if serve_time is not None else t_start + cfg.default_serve_offset_s
    window = [round(t_serve - cfg.pre_s, 2), round(t_serve + cfg.post_s, 2)]
    evidence: Dict[str, Any] = {"serve_time_s": round(t_serve, 2), "window_s": window}
    none = {"side": None, "confidence": 0.0, "source": None, "evidence": evidence}
    h = court_homography(geometry)
    if h is None:
        evidence["reason"] = "no court geometry"
        return none
    idx = tracks.window(t_serve - cfg.pre_s, t_serve + cfg.post_s)
    if server_side in ("near", "far"):
        receiver = "far" if server_side == "near" else "near"
        roles = (("server", server_side, cfg.server_scale_m), ("receiver", receiver, cfg.receiver_scale_m))
    else:
        roles = (("player", "near", cfg.receiver_scale_m), ("player", "far", cfg.receiver_scale_m))
    ests = []
    for role, player, scale in roles:
        est = _position_estimate(tracks, player, idx, h, scale, cfg)
        evidence[player] = est
        if est is not None:
            ests.append((role, est))
    if not ests:
        evidence["reason"] = "players not seen in the serve window"
        return none
    role, best = ests[0]
    conf = best["confidence"]
    if len(ests) > 1:
        role2, other = ests[1]
        if other["side"] == best["side"]:
            conf = 1.0 - (1.0 - conf) * (1.0 - other["confidence"])
        else:
            evidence["players_disagree"] = True
            if other["confidence"] > best["confidence"]:
                role, best, other = role2, other, best
            conf = best["confidence"] - other["confidence"]
    return {"side": best["side"], "confidence": round(conf, 3), "source": role, "evidence": evidence}


def serve_times(
    points: Sequence[PointLike],
    servers: Sequence[Optional[Mapping[str, Any]]],
    cfg: ServeSideConfig = DEFAULT_SERVE_SIDE_CONFIG,
) -> Tuple[List[float], float]:
    """Serve moment per point from ``detect_server`` outputs: start + the server's
    first raise; points without one use start + the match's median raise offset
    (``default_serve_offset_s`` if no point has a raise). Returns (times, offset)."""
    bounds = [_point_bounds(p) for p in points]
    raises: List[Optional[float]] = []
    for srv in servers:
        r = None
        if srv and srv.get("server") in ("near", "far"):
            r = ((srv.get("evidence") or {}).get(srv["server"]) or {}).get("first_raise_s")
        raises.append(None if r is None else float(r))
    known = [r for r in raises if r is not None]
    offset = float(np.median(known)) if known else cfg.default_serve_offset_s
    return [s + (r if r is not None else offset) for (s, _e), r in zip(bounds, raises)], offset


# ------------------------------------------------------------------ (b) score


def points_in_game(
    game_ids: Sequence[Optional[int]],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    same_point_gap_s: float = 0.0,
) -> List[Optional[int]]:
    """0-based index of each point within its game (None where the game is unknown).
    With ``bounds``, a point starting < ``same_point_gap_s`` after the previous one
    (same game) ended is the same point (second serve) and repeats its index."""
    out: List[Optional[int]] = []
    prev_game, k = None, -1
    for i, g in enumerate(game_ids):
        if g is None:
            out.append(None)
            prev_game, k = None, -1
            continue
        if g != prev_game:
            k = 0
        else:
            same = (bounds is not None and same_point_gap_s > 0
                    and bounds[i][0] - bounds[i - 1][1] < same_point_gap_s)
            k = k if same else k + 1
        out.append(k)
        prev_game = g
    return out


def score_serve_sides(
    game_ids: Sequence[Optional[int]],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    confidence: float = DEFAULT_SERVE_SIDE_CONFIG.score_confidence,
    cfg: ServeSideConfig = DEFAULT_SERVE_SIDE_CONFIG,
) -> List[Dict[str, Any]]:
    """Score-parity serve side per point: {"side", "confidence", "source": "score",
    "point_in_game"}. Even index -> deuce, odd -> ad (tiebreaks included: give the
    tiebreak one game id). Confidence = ``confidence`` x ``score_decay`` ** index."""
    ks = points_in_game(game_ids, bounds, cfg.same_point_gap_s if bounds is not None else 0.0)
    return [
        {"side": None, "confidence": 0.0, "source": None, "point_in_game": None} if k is None else
        {"side": DEUCE if k % 2 == 0 else AD, "confidence": round(confidence * cfg.score_decay ** k, 3),
         "source": SCORE_SOURCE, "point_in_game": k}
        for k in ks
    ]


def game_ids_from_games(games: Sequence[Mapping[str, Any]], n: int) -> List[Optional[int]]:
    """Per-point game index from ``infer_first_server_and_check(...)["games"]``
    ({game_index, start, end}, end exclusive)."""
    out: List[Optional[int]] = [None] * n
    for g in games:
        for k in range(int(g["start"]), min(n, int(g["end"]))):
            out[k] = int(g["game_index"])
    return out


def game_ids_from_servers(server_ids: Sequence[Optional[str]], **kwargs: Any) -> Optional[List[Optional[int]]]:
    """Label-free game index per point from a detected server-id sequence ("A"/"B"/
    None) via ``analysis.score.infer_first_server_and_check`` segmentation (first game
    may be partial). None when there is nothing to segment."""
    from analysis.score import infer_first_server_and_check

    if not any(s is not None for s in server_ids):
        return None
    kwargs.setdefault("allow_partial_first_game", True)
    try:
        check = infer_first_server_and_check(list(server_ids), **kwargs)
    except ValueError:
        return None
    return game_ids_from_games(check["games"], len(server_ids))


# ------------------------------------------------------------------ combine / cross-check


def combine(visual: Mapping[str, Any], score: Mapping[str, Any],
            cfg: ServeSideConfig = DEFAULT_SERVE_SIDE_CONFIG) -> Dict[str, Any]:
    """Final per-point answer, as the ``analyze_points`` row fields (``ROW_KEYS``).

    The visual side wins at/above ``min_visual_confidence``, else the score side, else a
    weak visual side. Agreeing sources: 1 - (1 - v)(1 - s); disagreeing: the chosen
    confidence x (1 - other / 2), and ``serve_side_agree`` False flags it."""
    v_side, v_conf = visual.get("side"), float(visual.get("confidence") or 0.0)
    s_side, s_conf = score.get("side"), float(score.get("confidence") or 0.0)
    agree = None if v_side is None or s_side is None else v_side == s_side
    if v_side is not None and (v_conf >= cfg.min_visual_confidence or s_side is None):
        side, src, conf, other = v_side, visual.get("source"), v_conf, s_conf
    elif s_side is not None:
        side, src, conf, other = s_side, SCORE_SOURCE, s_conf, v_conf
    else:
        side, src, conf, other = None, None, 0.0, 0.0
    if agree is True:
        conf = 1.0 - (1.0 - v_conf) * (1.0 - s_conf)
    elif agree is False:
        conf = conf * (1.0 - other / 2.0)
    return {
        "serve_side": side,
        "serve_side_confidence": round(conf, 3),
        "serve_side_source": src,
        "serve_side_score": s_side,
        "serve_side_agree": agree,
        "serve_side_visual": v_side,
        "serve_side_visual_confidence": round(v_conf, 3),
        "serve_side_visual_source": visual.get("source"),
    }


def serve_sides(
    points: Sequence[PointLike],
    tracks: PlayerTracks,
    geometry: Any,
    servers: Sequence[Optional[Mapping[str, Any]]],
    game_ids: Optional[Sequence[Optional[int]]] = None,
    *,
    score_confidence: Optional[float] = None,
    cfg: ServeSideConfig = DEFAULT_SERVE_SIDE_CONFIG,
) -> List[Dict[str, Any]]:
    """``combine``d serve-side fields for every point (visual from ``geometry`` +
    ``servers`` = detect_server outputs; score from ``game_ids`` when given)."""
    bounds = [_point_bounds(p) for p in points]
    times, _offset = serve_times(bounds, servers, cfg)
    visual = [serve_side(b, tracks, geometry, (srv or {}).get("server"), serve_time=t, cfg=cfg)
              for b, srv, t in zip(bounds, servers, times)]
    if game_ids is not None:
        score = score_serve_sides(game_ids, bounds,
                                  confidence=cfg.score_confidence if score_confidence is None else score_confidence,
                                  cfg=cfg)
    else:
        score = [{"side": None, "confidence": 0.0}] * len(bounds)
    return [combine(v, s, cfg) for v, s in zip(visual, score)]


def _agreement(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    return {"n": n, "agree": agree, "agreement": round(agree / n, 3) if n else None}


def cross_check(rows: Sequence[Mapping[str, Any]],
                cfg: ServeSideConfig = DEFAULT_SERVE_SIDE_CONFIG) -> Dict[str, Any]:
    """Per-match agreement of the visual and score serve sides (rows: analyze_points).

    ``overall`` / ``confident`` (visual confidence >= ``min_visual_confidence``) /
    ``by_visual_source`` (server / receiver / player) / ``by_server`` (near / far):
    {n, agree, agreement} over points where both sides exist; ``disagreements``: those
    points; ``visual_confidence`` histogram + quantiles; ``final_source`` counts."""
    both = [(i, r) for i, r in enumerate(rows) if r.get("serve_side_visual") and r.get("serve_side_score")]
    pair = lambda r: (r["serve_side_visual"], r["serve_side_score"])  # noqa: E731
    by_source = {s: _agreement([pair(r) for _i, r in both if r.get("serve_side_visual_source") == s])
                 for s in VISUAL_SOURCES}
    by_server = {s: _agreement([pair(r) for _i, r in both if r.get("server") == s]) for s in ("near", "far")}
    confs = np.array([float(r.get("serve_side_visual_confidence") or 0.0) for r in rows
                      if r.get("serve_side_visual")], dtype=np.float64)
    edges = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0001)
    hist = {f"{lo:.2f}-{min(hi, 1.0):.2f}": int(((confs >= lo) & (confs < hi)).sum())
            for lo, hi in zip(edges[:-1], edges[1:])}
    finals: Dict[str, int] = {}
    for r in rows:
        key = str(r.get("serve_side_source"))
        finals[key] = finals.get(key, 0) + 1
    return {
        "n_points": len(rows),
        "n_visual": int(confs.size),
        "n_score": sum(1 for r in rows if r.get("serve_side_score")),
        "overall": _agreement([pair(r) for _i, r in both]),
        "confident": _agreement([pair(r) for _i, r in both
                                 if float(r.get("serve_side_visual_confidence") or 0.0) >= cfg.min_visual_confidence]),
        "by_visual_source": by_source,
        "by_server": by_server,
        "disagreements": [
            {"index": i, "start_s": r.get("start_s"), "server": r.get("server"),
             "visual": r["serve_side_visual"], "visual_confidence": r.get("serve_side_visual_confidence"),
             "visual_source": r.get("serve_side_visual_source"), "score": r["serve_side_score"]}
            for i, r in both if r["serve_side_visual"] != r["serve_side_score"]
        ],
        "visual_confidence": {
            "hist": hist,
            "quantiles": ({q: round(float(np.quantile(confs, float(q))), 3) for q in ("0.1", "0.25", "0.5", "0.75")}
                          if confs.size else None),
        },
        "final_source": finals,
    }
