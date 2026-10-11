"""Serve side (deuce / ad) per point: src/analysis/serve_side.py. Synthetic court
geometry + player positions only (no video, no model artifact)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.players import PlayerTracks, analyze_points
from analysis.score import MatchFormat, reconstruct
from analysis.serve_side import (
    AD,
    DEUCE,
    ROW_KEYS,
    ServeSideConfig,
    combine,
    court_homography,
    cross_check,
    feet_px,
    game_ids_from_games,
    game_ids_from_servers,
    geometry_to_dict,
    points_in_game,
    score_serve_sides,
    serve_side,
    serve_sides,
    side_for_x,
)

FPS = 5.0
# Straight-on synthetic court, 1920x1080 reference: x = 960 + 50u, y = 600 - 20v (u metres
# to the camera's right, v metres away from the camera; net at v = 0, baselines +-11.885).
M_TO_PX = np.array([[50.0, 0.0, 960.0], [0.0, -20.0, 600.0], [0.0, 0.0, 1.0]])
GEOMETRY = {"homography_px_to_m": np.linalg.inv(M_TO_PX).tolist(), "net_line_px": [[686.0, 600.0], [1234.0, 600.0]]}


def _player(u, v, h=200.0, *, ankles=True):
    """Player dict with the feet at court (u, v); ankle keypoints at the feet."""
    x, y = 960.0 + 50.0 * u, 600.0 - 20.0 * v
    kp = np.tile(np.array([[x, y - h / 2]], dtype=np.float32), (17, 1))
    conf = np.full(17, 0.9, dtype=np.float32)
    kp[15], kp[16] = (x - 8.0, y - 4.0), (x + 8.0, y - 4.0)
    if not ankles:
        conf[15] = conf[16] = 0.1
    return {"box": np.array([x - 30, y - h, x + 30, y], dtype=np.float32), "keypoints": kp, "conf": conf}


def _tracks(duration_s, near=None, far=None, geometry=GEOMETRY):
    """near/far: (u, v) for the whole clip, a callable t -> (u, v) | None, or None."""
    n = int(round(duration_s * FPS))
    t = np.arange(n) / FPS

    def rec(spec, ti):
        pos = spec(ti) if callable(spec) else spec
        return None if pos is None else _player(*pos)

    return PlayerTracks.from_records([rec(near, ti) for ti in t], [rec(far, ti) for ti in t], fps=FPS,
                                     ref_size=(1920, 1080), geometry=geometry)


# ---------------------------------------------------------------- geometry helpers


def test_side_for_x_both_roles():
    # near player: camera right (x > 0) is their right = deuce court; far: mirrored
    assert side_for_x("near", 0.7) == DEUCE and side_for_x("near", -0.7) == AD
    assert side_for_x("far", -0.7) == DEUCE and side_for_x("far", 0.7) == AD
    assert side_for_x("near", float("nan")) is None and side_for_x("near", 0.0) is None


def test_court_homography_accepts_dict_json_object_array():
    h = np.asarray(GEOMETRY["homography_px_to_m"])
    np.testing.assert_allclose(court_homography(GEOMETRY), h)
    np.testing.assert_allclose(court_homography(json.dumps(GEOMETRY)), h)
    np.testing.assert_allclose(court_homography(h), h)

    class Geom:  # preprocessing.netline_slots.CourtGeometry duck type
        px_to_m = h
        net_a, net_b = np.array([686.0, 600.0]), np.array([1234.0, 600.0])

    np.testing.assert_allclose(court_homography(Geom()), h)
    assert geometry_to_dict(Geom())["net_line_px"] == [[686.0, 600.0], [1234.0, 600.0]]
    assert court_homography(None) is None and court_homography({"homography_px_to_m": None}) is None
    assert court_homography("not json") is None and court_homography(np.eye(2)) is None


def test_feet_prefer_ankles_then_box_bottom():
    tracks = _tracks(1, near=(1.0, -12.0))
    feet = feet_px(tracks.near, np.arange(3))
    np.testing.assert_allclose(feet[0], [1010.0, 600.0 + 240.0 - 4.0])  # ankle midpoint
    low = PlayerTracks.from_records([_player(1.0, -12.0, ankles=False)], [None], fps=FPS)
    np.testing.assert_allclose(feet_px(low.near, np.arange(1))[0], [1010.0, 840.0])  # box bottom-centre
    assert np.isnan(feet_px(low.far, np.arange(1))).all()


# ---------------------------------------------------------------- (a) visual


@pytest.mark.parametrize("u,expected", [(0.6, DEUCE), (-0.6, AD)])
def test_near_server_side_from_position(u, expected):
    tracks = _tracks(10, near=(u, -12.3))
    out = serve_side((3.0, 9.0), tracks, GEOMETRY, "near", serve_time=4.0)
    assert out["side"] == expected and out["source"] == "server"
    assert out["confidence"] > 0.85
    ev = out["evidence"]["near"]
    assert ev["frames"] == 8 and abs(ev["x_m"] - u) < 0.05 and ev["agree_frac"] == 1.0
    assert out["evidence"]["far"] is None and out["evidence"]["window_s"] == [3.0, 4.5]


@pytest.mark.parametrize("u,expected", [(-0.8, DEUCE), (0.8, AD)])
def test_far_server_side_when_seen(u, expected):
    tracks = _tracks(10, far=(u, 12.4))
    out = serve_side((3.0, 9.0), tracks, GEOMETRY, "far", serve_time=4.0)
    assert out["side"] == expected and out["source"] == "server"


@pytest.mark.parametrize("u,expected", [(3.5, DEUCE), (-3.5, AD)])
def test_far_server_inferred_from_near_receiver(u, expected):
    """Far track absent: the near receiver stands in their own deuce (camera-right) or ad
    court, diagonally opposite the far server."""
    tracks = _tracks(10, near=(u, -12.5))
    out = serve_side((3.0, 9.0), tracks, GEOMETRY, "far")
    assert out["side"] == expected and out["source"] == "receiver" and out["confidence"] > 0.9
    assert out["evidence"]["serve_time_s"] == 3.0 + ServeSideConfig().default_serve_offset_s


def test_server_and_receiver_vote():
    agree = serve_side((3.0, 9.0), _tracks(10, near=(0.3, -12.3), far=(-3.5, 12.5)), GEOMETRY, "near", serve_time=4.0)
    alone = serve_side((3.0, 9.0), _tracks(10, near=(0.3, -12.3)), GEOMETRY, "near", serve_time=4.0)
    assert agree["side"] == alone["side"] == DEUCE and agree["confidence"] > alone["confidence"]
    clash = serve_side((3.0, 9.0), _tracks(10, near=(0.3, -12.3), far=(3.5, 12.5)), GEOMETRY, "near", serve_time=4.0)
    assert clash["evidence"]["players_disagree"] is True
    assert clash["side"] == AD and clash["source"] == "receiver"  # the more confident one wins
    assert clash["confidence"] < 0.5


def test_unknown_server_uses_wide_scale_and_mid_court_is_penalised():
    tracks = _tracks(10, near=(0.6, -12.3))
    known = serve_side((3.0, 9.0), tracks, GEOMETRY, "near", serve_time=4.0)
    unknown = serve_side((3.0, 9.0), tracks, GEOMETRY, None, serve_time=4.0)
    assert unknown["side"] == DEUCE and unknown["source"] == "player"
    assert unknown["confidence"] < known["confidence"]
    mid = serve_side((3.0, 9.0), _tracks(10, near=(0.6, -5.0)), GEOMETRY, "near", serve_time=4.0)
    assert mid["side"] == DEUCE and mid["confidence"] == pytest.approx(known["confidence"] * 0.5, abs=0.01)


def test_no_geometry_or_no_player_gives_no_side():
    tracks = _tracks(10, near=(0.6, -12.3))
    out = serve_side((3.0, 9.0), tracks, None, "near")
    assert out["side"] is None and out["confidence"] == 0.0 and out["evidence"]["reason"] == "no court geometry"
    empty = serve_side((3.0, 9.0), _tracks(10), GEOMETRY, "far")
    assert empty["side"] is None and empty["source"] is None


def test_serve_window_follows_the_raise():
    """Server walks from the ad to the deuce side after the previous point; the window
    around the raise (detect_server evidence) sees only the deuce position."""
    tracks = _tracks(20, near=lambda t: (-0.6, -12.3) if t < 6.0 else (0.6, -12.3))
    servers = [{"server": "near", "evidence": {"near": {"first_raise_s": 4.5}}}]
    rows = serve_sides([(3.0, 15.0)], tracks, GEOMETRY, servers)
    assert rows[0]["serve_side_visual"] == DEUCE
    early = serve_side((3.0, 15.0), tracks, GEOMETRY, "near", serve_time=4.0)
    assert early["side"] == AD


# ---------------------------------------------------------------- (b) score


def test_score_parity_per_game():
    sides = score_serve_sides([0, 0, 0, 0, 0, 1, 1, 1, None, 2])
    assert [s["side"] for s in sides] == [DEUCE, AD, DEUCE, AD, DEUCE, DEUCE, AD, DEUCE, None, DEUCE]
    assert [s["point_in_game"] for s in sides] == [0, 1, 2, 3, 4, 0, 1, 2, None, 0]
    assert sides[0]["source"] == "score" and sides[0]["confidence"] == 0.9
    assert sides[4]["confidence"] < sides[0]["confidence"]  # later in the game = less sure


def test_score_parity_in_a_tiebreak():
    """6-6 then a 7-point tiebreak: the parity continues per tiebreak point."""
    fmt = MatchFormat(sets_to_win=1)
    winners = []
    for g in range(12):
        winners += ["A" if g % 2 == 0 else "B"] * 4
    winners += ["A", "B", "A", "A", "B", "A", "A", "A"]  # tiebreak 6-2
    rec = reconstruct(winners, fmt)
    tb = [r for r in rec if r["tiebreak"]]
    assert len(tb) == 8 and len({r["game_index"] for r in tb}) == 1
    sides = score_serve_sides([r["game_index"] for r in rec])
    tb_sides = [s["side"] for s, r in zip(sides, rec) if r["tiebreak"]]
    assert tb_sides == [DEUCE, AD] * 4
    # serve rotation inside the tiebreak does not reset the parity
    assert [r["server"] for r in tb][:4] == ["A", "B", "B", "A"]


def test_second_serve_clip_keeps_the_point_index():
    bounds = [(0.0, 10.0), (30.0, 33.0), (35.0, 45.0), (70.0, 80.0)]  # 2 s gap: fault + 2nd serve
    assert points_in_game([0, 0, 0, 0], bounds, 5.0) == [0, 1, 1, 2]
    assert points_in_game([0, 0, 0, 0]) == [0, 1, 2, 3]
    assert [s["side"] for s in score_serve_sides([0, 0, 0, 0], bounds)] == [DEUCE, AD, AD, DEUCE]


def test_game_ids_from_segmentation():
    assert game_ids_from_games([{"game_index": 0, "start": 0, "end": 2}, {"game_index": 1, "start": 2, "end": 5}], 6) \
        == [0, 0, 1, 1, 1, None]
    ids = game_ids_from_servers(["A"] * 4 + ["B"] * 5 + ["A"] * 4)
    assert ids == [0] * 4 + [1] * 5 + [2] * 4
    assert game_ids_from_servers([None, None]) is None


# ---------------------------------------------------------------- combine / cross-check


def test_combine_prefers_confident_visual_and_flags_disagreement():
    v = {"side": AD, "confidence": 0.95, "source": "server"}
    s = {"side": DEUCE, "confidence": 0.9}
    out = combine(v, s)
    assert set(out) == set(ROW_KEYS)
    assert out["serve_side"] == AD and out["serve_side_source"] == "server" and out["serve_side_agree"] is False
    assert out["serve_side_score"] == DEUCE and out["serve_side_confidence"] < 0.95
    weak = combine({"side": AD, "confidence": 0.3, "source": "receiver"}, s)
    assert weak["serve_side"] == DEUCE and weak["serve_side_source"] == "score" and weak["serve_side_agree"] is False
    agree = combine({"side": DEUCE, "confidence": 0.6, "source": "receiver"}, s)
    assert agree["serve_side_agree"] is True and agree["serve_side_confidence"] == pytest.approx(0.96)
    only_score = combine({"side": None, "confidence": 0.0}, s)
    assert only_score["serve_side"] == DEUCE and only_score["serve_side_agree"] is None
    only_weak_visual = combine({"side": AD, "confidence": 0.2, "source": "player"}, {"side": None})
    assert only_weak_visual["serve_side"] == AD and only_weak_visual["serve_side_source"] == "player"
    assert combine({"side": None}, {"side": None})["serve_side"] is None


def test_cross_check_reports_agreement_by_source_and_server():
    rows = []
    for i in range(6):
        server = "near" if i < 3 else "far"
        v = combine({"side": DEUCE if i % 2 == 0 else AD, "confidence": 0.9,
                     "source": "server" if server == "near" else "receiver"},
                    {"side": DEUCE if i % 2 == 0 or i == 5 else AD, "confidence": 0.9})
        rows.append({"start_s": 10.0 * i, "server": server, **v})
    rows.append({"start_s": 70.0, "server": None, **combine({"side": None}, {"side": None})})
    check = cross_check(rows)
    assert check["n_points"] == 7 and check["n_visual"] == 6 and check["n_score"] == 6
    assert check["overall"] == {"n": 6, "agree": 5, "agreement": 0.833}
    assert check["by_server"]["near"]["agreement"] == 1.0 and check["by_server"]["far"]["agree"] == 2
    assert check["by_visual_source"]["receiver"]["n"] == 3 and check["by_visual_source"]["player"]["n"] == 0
    assert [d["index"] for d in check["disagreements"]] == [5]
    assert check["visual_confidence"]["hist"]["0.90-1.00"] == 6
    assert check["final_source"] == {"server": 3, "receiver": 3, "None": 1}
    json.dumps(check)


# ---------------------------------------------------------------- analyze_points


def test_analyze_points_carries_serve_side_rows():
    starts = [10.0, 40.0, 70.0, 100.0]
    us = [3.5, -3.5, 3.5, -3.5]  # near receiver: deuce, ad, deuce, ad (far serves)

    def near(t):
        for s, u in zip(starts, us):
            if s - 2 <= t <= s + 10:
                return (u, -12.5)
        return None

    tracks = _tracks(115, near=near)
    pts = [(s, s + 8.0) for s in starts]
    rows = analyze_points(pts, tracks, game_ids=[0, 0, 0, 0])
    assert set(ROW_KEYS) <= set(rows[0])
    assert [r["server"] for r in rows] == ["far"] * 4  # elimination: near never raises
    assert [r["serve_side"] for r in rows] == [DEUCE, AD, DEUCE, AD]
    assert all(r["serve_side_source"] == "receiver" and r["serve_side_agree"] for r in rows)
    # tracks.geometry is the default; an explicit None-geometry track gives score only
    bare = _tracks(115, near=near, geometry=None)
    rows2 = analyze_points(pts, bare, game_ids=[0, 0, 0, 0])
    assert [r["serve_side_visual"] for r in rows2] == [None] * 4
    assert [r["serve_side"] for r in rows2] == [DEUCE, AD, DEUCE, AD]
    assert {r["serve_side_source"] for r in rows2} == {"score"}


def test_geometry_rides_in_the_players_npz(tmp_path):
    tracks = _tracks(2, near=(0.5, -12.0))
    path = tracks.save_npz(tmp_path / "m_players.npz")
    with np.load(path) as d:
        assert json.loads(str(d["geometry_json"]))["homography_px_to_m"] == GEOMETRY["homography_px_to_m"]
    back = PlayerTracks.from_npz(path)
    assert back.geometry == GEOMETRY
    assert back.truncated(3).geometry == GEOMETRY
    no_geo = _tracks(2, geometry=None).save_npz(tmp_path / "n_players.npz")
    with np.load(no_geo) as d:
        assert "geometry_json" not in d.files


def test_serve_side_module_is_numpy_only():
    text = (Path(__file__).resolve().parents[1] / "src" / "analysis" / "serve_side.py").read_text(encoding="utf-8")
    for banned in ("import torch", "import cv2", "import av", "ultralytics", "onnxruntime"):
        assert banned not in text
