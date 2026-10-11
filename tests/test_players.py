"""Phase 1 player identity / server detection (src/analysis/players.py) + the opt-in
engine wiring (RunRequest.save_player_tracks). Synthetic keypoints only: no video,
no model artifact."""
from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import numpy as np
import pytest

from analysis.players import (
    SIGNATURE_KEYS,
    PlayerTracks,
    SideTrack,
    analyze_points,
    arm_raised,
    body_signature,
    detect_server,
    pack_player_records,
    points_document,
    serve_consistency,
    track_identities,
)
from analysis.serve_side import ROW_KEYS
from rallyclip_core.contracts import RunRequest, RunResult, RuntimeDeps
from rallyclip_core.pipelines import FRAME_STARTEND_HEATMAP, resolve_pipeline_spec

FPS = 5.0


# ---------------------------------------------------------------- synthetic poses


def _pose(cx, foot_y, h, *, raise_arm=False, build="a"):
    """COCO-17 standing pose of height h (px) with feet at foot_y; y grows down.
    build "a"/"b" differ in limb proportions (body signature)."""
    torso_top, hip, knee = (0.80, 0.50, 0.27) if build == "a" else (0.84, 0.58, 0.31)
    sw = 0.12 if build == "a" else 0.16
    y = lambda frac: foot_y - frac * h  # noqa: E731
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[0] = (cx, y(0.93))
    kp[1], kp[2] = (cx - 0.02 * h, y(0.95)), (cx + 0.02 * h, y(0.95))
    kp[3], kp[4] = (cx - 0.04 * h, y(0.94)), (cx + 0.04 * h, y(0.94))
    kp[5], kp[6] = (cx - sw * h, y(torso_top)), (cx + sw * h, y(torso_top))
    kp[7], kp[8] = (cx - 0.15 * h, y(0.62)), (cx + 0.15 * h, y(0.62))
    kp[9], kp[10] = (cx - 0.16 * h, y(0.47)), (cx + 0.16 * h, y(0.47))
    kp[11], kp[12] = (cx - 0.09 * h, y(hip)), (cx + 0.09 * h, y(hip))
    kp[13], kp[14] = (cx - 0.09 * h, y(knee)), (cx + 0.09 * h, y(knee))
    kp[15], kp[16] = (cx - 0.09 * h, y(0.03)), (cx + 0.09 * h, y(0.03))
    if raise_arm:
        kp[8] = (cx + 0.14 * h, y(0.98))
        kp[10] = (cx + 0.12 * h, y(1.12))
    box = np.array([cx - 0.2 * h, foot_y - h, cx + 0.2 * h, foot_y], dtype=np.float32)
    return {"box": box, "keypoints": kp, "conf": np.full(17, 0.9, dtype=np.float32)}


NEAR = dict(cx=640.0, foot_y=690.0, h=300.0)
FAR = dict(cx=640.0, foot_y=250.0, h=90.0)


def _tracks(duration_s, near_fn=None, far_fn=None):
    """near_fn/far_fn(t) -> player dict or None, sampled at FPS."""
    n = int(round(duration_s * FPS))
    t = np.arange(n) / FPS
    near = [near_fn(ti) if near_fn else None for ti in t]
    far = [far_fn(ti) if far_fn else None for ti in t]
    return PlayerTracks.from_records(near, far, fps=FPS)


def _serving(t, start, side_kwargs, *, build="a"):
    # arm up for ~1 s, 0.4-1.4 s after the point start
    return _pose(**side_kwargs, raise_arm=(start + 0.4 <= t <= start + 1.4), build=build)


# ---------------------------------------------------------------- detect_server


def test_detect_server_near_serves():
    tracks = _tracks(20, near_fn=lambda t: _serving(t, 5.0, NEAR), far_fn=lambda t: _pose(**FAR))
    out = detect_server((5.0, 15.0), tracks)
    assert out["server"] == "near"
    assert out["confidence"] > 0.5
    assert out["evidence"]["method"] == "raise"
    assert out["evidence"]["near"]["raise_frames"] >= 3
    assert out["evidence"]["far"]["raise_frames"] == 0


def test_detect_server_far_serves_by_raise():
    tracks = _tracks(20, near_fn=lambda t: _pose(**NEAR), far_fn=lambda t: _serving(t, 5.0, FAR))
    out = detect_server((5.0, 15.0), tracks)
    assert out["server"] == "far"
    assert out["evidence"]["method"] == "raise"
    assert out["confidence"] > 0.5


def test_detect_server_nobody_raises_both_visible_is_none():
    tracks = _tracks(20, near_fn=lambda t: _pose(**NEAR), far_fn=lambda t: _pose(**FAR))
    out = detect_server((5.0, 15.0), tracks)
    assert out["server"] is None
    assert out["confidence"] == 0.0


def test_detect_server_far_by_elimination_when_far_undetected():
    # Real 5 fps dumps rarely detect the far player: a well-seen near player who never
    # raises an arm means the far player served (low, fixed confidence).
    tracks = _tracks(20, near_fn=lambda t: _pose(**NEAR), far_fn=None)
    out = detect_server((5.0, 15.0), tracks)
    assert out["server"] == "far"
    assert out["evidence"]["method"] == "elimination"
    assert 0.0 < out["confidence"] < 0.5


def test_detect_server_missing_keypoints_is_none():
    def near_no_arms(t):
        p = _serving(t, 5.0, NEAR)
        p["conf"] = p["conf"].copy()
        p["conf"][[5, 6, 7, 8, 9, 10]] = 0.05  # shoulders/elbows/wrists unreliable
        return p

    tracks = _tracks(20, near_fn=near_no_arms, far_fn=None)
    out = detect_server((5.0, 15.0), tracks)
    assert out["server"] is None  # cannot see the near arm -> no elimination either
    assert out["evidence"]["near"]["arm_valid_frac"] == 0.0


def test_detect_server_nobody_present():
    tracks = _tracks(20)
    out = detect_server((5.0, 15.0), tracks)
    assert out["server"] is None and out["confidence"] == 0.0


def test_arm_raised_uses_shoulder_fallback_without_head():
    p = _pose(**NEAR, raise_arm=True)
    kp = p["keypoints"].astype(np.float64).copy()
    kp[0:5] = np.nan  # no head keypoints
    raised, valid = arm_raised(kp[None], p["box"][None].astype(np.float64))
    assert raised.tolist() == [True] and valid.tolist() == [True]
    kp_down = _pose(**NEAR)["keypoints"].astype(np.float64)[None]
    raised, valid = arm_raised(kp_down, p["box"][None].astype(np.float64))
    assert raised.tolist() == [False] and valid.tolist() == [True]


# ---------------------------------------------------------------- identities


def _match(points_spec):
    """points_spec: list of (start, end, near_build). Near player always present;
    far player absent (as in the real dumps)."""
    def near_fn(t):
        for s, e, build in points_spec:
            if s - 2 <= t <= e + 2:
                return _pose(**NEAR, build=build)
        return _pose(**NEAR, build=points_spec[0][2])
    return _tracks(points_spec[-1][1] + 5, near_fn=near_fn)


def _points(starts, dur=8.0):
    return [(s, s + dur) for s in starts]


def test_body_signature_separates_builds():
    a = np.stack([_pose(**NEAR, build="a")["keypoints"]] * 5).astype(np.float64)
    b = np.stack([_pose(**NEAR, build="b")["keypoints"]] * 5).astype(np.float64)
    sa, na = body_signature(a)
    sb, nb = body_signature(b)
    assert sa.shape == (len(SIGNATURE_KEYS),) and na == 5 and nb == 5
    assert np.nanmean(np.abs(sa - sb)) > 0.1
    empty, n0 = body_signature(np.full((0, 17, 2), np.nan))
    assert n0 == 0 and np.isnan(empty).all()


def test_track_identities_detects_changeover():
    # 4 points (15 s apart), 100 s changeover break, 4 more points; the near player's
    # build changes and the server side stays "near" (as it does across a changeover).
    starts = [10, 33, 56, 79, 187, 210, 233, 256]
    pts = _points(starts)
    tracks = _match([(s, e, "a" if i < 4 else "b") for i, (s, e) in enumerate(pts)])
    servers = [{"server": "near", "confidence": 0.8}] * 8
    ids = track_identities(pts, tracks, servers)
    assert [r["near_id"] for r in ids] == ["A"] * 4 + ["B"] * 4
    assert [r["far_id"] for r in ids] == ["B"] * 4 + ["A"] * 4
    assert [r["side_change_before"] for r in ids] == [False] * 4 + [True] + [False] * 3
    ev = ids[4]["evidence"]["boundary"]
    assert ev["serve"] == "same" and ev["gap_score"] == 1.0 and ev["signature_score"] > 0
    assert 0.0 < ids[-1]["confidence"] <= 1.0


def test_track_identities_game_break_without_change():
    # 45 s break where the server side flips (a game boundary without change of ends)
    # and the near player looks the same -> no side change.
    starts = [10, 33, 56, 79, 132, 155, 178, 201]
    pts = _points(starts)
    tracks = _match([(s, e, "a") for s, e in pts])
    servers = [{"server": "near", "confidence": 0.8}] * 4 + [{"server": "far", "confidence": 0.3}] * 4
    ids = track_identities(pts, tracks, servers)
    assert {r["near_id"] for r in ids} == {"A"}
    assert ids[4]["evidence"]["boundary"]["serve"] == "flip"
    assert not any(r["side_change_before"] for r in ids)


def test_track_identities_empty():
    assert track_identities([], _tracks(1)) == []


# ---------------------------------------------------------------- consistency


def test_serve_consistency_perfect_alternation():
    # A serves 5, B serves 5, ... with ends changing so the side pattern is near x10, far x10
    ids = ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5
    sides = ["near"] * 10 + ["far"] * 10
    rows = [{"server": s, "server_id": i} for s, i in zip(sides, ids)]
    out = serve_consistency(rows)
    assert out["none_frac"] == 0.0
    assert out["by_player"]["runs"] == [5, 5, 5, 5]
    assert out["by_player"]["short_run_frac"] == 0.0
    assert out["by_player"]["runs_4_to_7_frac"] == 1.0
    assert out["by_side"]["runs"] == [10, 10]


def test_serve_consistency_noise_is_flagged():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        r = rng.random()
        rows.append({"server": None if r < 0.2 else ("near" if r < 0.6 else "far"),
                     "server_id": None if r < 0.2 else ("A" if rng.random() < 0.5 else "B")})
    out = serve_consistency(rows)
    assert out["none_frac"] > 0.1
    assert out["by_player"]["short_run_frac"] > 0.5
    assert out["by_side"]["short_run_frac"] > 0.3


# ---------------------------------------------------------------- analyze / io


def test_analyze_points_rows_and_document():
    starts = [10, 33, 56, 79, 187, 210, 233, 256]
    pts = _points(starts)

    def near_fn(t):
        for i, s in enumerate(starts):
            if s - 2 <= t <= s + 10:
                build = "a" if i < 4 else "b"
                return _pose(**NEAR, build=build, raise_arm=(s + 0.4 <= t <= s + 1.4))
        return None

    tracks = _tracks(starts[-1] + 15, near_fn=near_fn)
    rows = analyze_points([{"start_s": s, "end_s": e} for s, e in pts], tracks)
    assert len(rows) == 8
    assert set(rows[0]) == {
        "start_s", "end_s", "server", "server_confidence", "server_id", "near_id", "far_id",
        "identity_confidence", "side_change_before", "near_present_frac", "far_present_frac",
        *ROW_KEYS,
    }
    assert {r["serve_side_visual"] for r in rows} == {None}  # no court geometry
    # score parity from the server-id segmentation: two games of 4 points
    assert [r["serve_side"] for r in rows] == ["deuce", "ad"] * 4
    assert {r["serve_side_source"] for r in rows} == {"score"}
    assert [r["server"] for r in rows] == ["near"] * 8
    assert [r["server_id"] for r in rows] == ["A"] * 4 + ["B"] * 4
    assert rows[0]["near_present_frac"] == 1.0 and rows[0]["far_present_frac"] == 0.0
    doc = points_document(rows)
    assert doc["schema"] == "rallyclip.points/v1"
    assert doc["side_changes_s"] == [187.0]
    assert doc["serve_side_check"]["n_score"] == 8 and doc["serve_side_check"]["n_visual"] == 0
    json.dumps(doc)  # serialisable


def test_player_tracks_npz_round_trip(tmp_path):
    tracks = _tracks(4, near_fn=lambda t: _pose(**NEAR) if t < 2 else None, far_fn=lambda t: _pose(**FAR))
    path = tracks.save_npz(tmp_path / "x_players.npz", pointness=np.linspace(0, 1, len(tracks)))
    with np.load(path) as d:
        keys = set(d.files)
        assert d["near_kp"].shape == (20, 17, 2) and d["near_present"].dtype == bool
        assert list(d["ref_size"]) == [1280, 720]
    assert {"near_box", "near_kp", "near_kpconf", "near_present", "far_box", "far_kp", "far_kpconf",
            "far_present", "timestamps", "fps", "ref_size", "pointness"} <= keys
    back = PlayerTracks.from_npz(path)
    assert back.fps == FPS and back.ref_size == (1280, 720)
    np.testing.assert_array_equal(back.near.present, tracks.near.present)
    np.testing.assert_array_equal(back.timestamps, tracks.timestamps)
    np.testing.assert_allclose(back.near.kp, tracks.near.kp, equal_nan=True)
    np.testing.assert_allclose(back.far.kpconf, tracks.far.kpconf, equal_nan=True)


def test_pack_player_records_absent_is_nan():
    side = pack_player_records([None, _pose(**NEAR), {"box": [1, 2, 3, 4], "keypoints": np.zeros((17, 2))}])
    assert isinstance(side, SideTrack)
    assert side.present.tolist() == [False, True, True]
    assert np.isnan(side.kp[0]).all() and np.isnan(side.kpconf[2]).all()
    assert side.box[2].tolist() == [1, 2, 3, 4]


def test_from_arrays_without_player_keys_is_absent():
    tracks = PlayerTracks.from_arrays({"timestamps": np.arange(10) / FPS, "fps": FPS})
    assert len(tracks) == 10 and not tracks.near.present.any() and not tracks.far.present.any()
    assert detect_server((0.0, 2.0), tracks)["server"] is None


def test_players_module_is_numpy_only():
    src = Path(__file__).resolve().parents[1] / "src" / "analysis" / "players.py"
    text = src.read_text(encoding="utf-8")
    for banned in ("import torch", "import cv2", "import av", "ultralytics", "onnxruntime"):
        assert banned not in text


# ---------------------------------------------------------------- engine wiring


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


def _manifest(tmp_path: Path) -> Path:
    (tmp_path / "model.onnx").write_bytes(b"x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "pipeline": {"id": "frame_startend_heatmap"},
        "feature_pipeline": {"feature_set": "v1", "target_fps": FPS},
        "inference": {"seq_len_frames": 100, "overlap_frames": 50},
        "postprocess": {"method": "heatmap_hybrid", "params": {"mode": "hybrid", "threshold": 0.5}},
    }), encoding="utf-8")
    return manifest


def _request(tmp_path: Path, **kw) -> RunRequest:
    return RunRequest(
        video_path=tmp_path / "match.mp4", output_dir=tmp_path / "out", output_name="match",
        csv_output_dir=tmp_path / "out", write_csv=True, segment_video=False,
        yolo_weights="pose.onnx", yolo_device=None, model_path=tmp_path / "model.onnx",
        scaler_path=tmp_path / "scaler.json", fps=FPS, seq_len=10, imgsz=960, conf=0.25,
        feature_set="v1", screen_width=1280, screen_height=720, overlap=5, sigma=0.0, low=0.0,
        high=0.0, min_dur_sec=0.0, pipeline_id=FRAME_STARTEND_HEATMAP, **kw,
    )


def _deps() -> RuntimeDeps:
    return RuntimeDeps(
        np=np, PoseExtractor=None, DataPreprocessor=None, FeatureEngineer=None,
        load_scaler_asset=lambda path: _IdentityScaler(), load_model_from_checkpoint=None,
        run_windowed_inference_average_onnx_stream=None, run_windowed_inference_average_torch_stream=None,
        gaussian_filter1d=None, hysteresis_threshold=None, extract_segments_from_binary=None,
        write_segments_csv=lambda *a, **k: None, segment_video=lambda *a, **k: None,
    )


def _run_engine(tmp_path, monkeypatch, *, save_player_tracks, geometry=None):
    from rallyclip_engine import models as engine_models
    from rallyclip_engine.models import build_analysis_model
    import infer

    n = 60
    point = np.full(n, 0.05, dtype=np.float32)
    point[10:40] = 0.95
    start = np.zeros(n, dtype=np.float32)
    start[10] = 1.0
    end = np.zeros(n, dtype=np.float32)
    end[39] = 1.0
    seen = {}

    def fake_stream(request, deps, pcb, ccb, player_sink=None):
        seen["player_sink"] = player_sink
        if player_sink is not None and geometry is not None:
            player_sink.geometry = geometry  # build_feature_stream sets the run's court geometry
        for i in range(n):
            if player_sink is not None:
                player_sink.append((_pose(**NEAR, raise_arm=12 <= i <= 16), None))
            yield np.zeros(4, dtype=np.float32), 0

    monkeypatch.setattr(engine_models, "build_feature_stream", fake_stream)
    monkeypatch.setattr(infer, "run_multitrack_windowed_inference_onnx_stream",
                        lambda path, rows, **k: (list(rows), np.stack([point, start, end]))[1])
    req = _request(tmp_path, **({"save_player_tracks": True} if save_player_tracks else {}))
    spec = resolve_pipeline_spec(req.model_path, _manifest(tmp_path))
    result = build_analysis_model(req, spec, _deps()).run()
    return result, seen


def test_engine_player_outputs_default_off(tmp_path, monkeypatch):
    result, seen = _run_engine(tmp_path, monkeypatch, save_player_tracks=False)
    assert seen["player_sink"] is None  # the default path never tees
    assert result.diagnostics == {"pipeline_id": FRAME_STARTEND_HEATMAP}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["match_segments.csv"]
    assert RunRequest.__dataclass_fields__["save_player_tracks"].default is False


def test_engine_player_outputs_opt_in(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    baseline, _ = _run_engine(tmp_path / "a", monkeypatch, save_player_tracks=False)
    result, seen = _run_engine(tmp_path, monkeypatch, save_player_tracks=True)
    assert result.intervals_sec == baseline.intervals_sec  # segments unaffected
    assert (tmp_path / "out" / "match_segments.csv").read_bytes() == (tmp_path / "a" / "out" / "match_segments.csv").read_bytes()
    npz = Path(result.diagnostics["players_npz"])
    pts = Path(result.diagnostics["points_json"])
    assert npz.name == "match_players.npz" and pts.name == "match_points.json"
    assert npz.parent == tmp_path / "out" and pts.parent == tmp_path / "out"
    with np.load(npz) as d:
        assert d["near_present"].shape == (60,) and d["near_present"].all() and not d["far_present"].any()
        assert d["pointness"].shape == (60,)
    doc = json.loads(pts.read_text(encoding="utf-8"))
    assert doc["schema"] == "rallyclip.points/v1"
    assert len(doc["points"]) == len(result.intervals_sec) == 1
    assert doc["points"][0]["server"] == "near" and doc["points"][0]["server_id"] == "A"


def test_engine_player_outputs_carry_serve_side_and_geometry(tmp_path, monkeypatch):
    """The sink's court geometry (v2 net-line runs) reaches the points rows (visual serve
    side) and the players npz (``geometry_json``). Synthetic 1280x720 court: centre line
    at x = 600 px, near baseline at y ~ 697 px; the near server stands right of it."""
    from preprocessing.netline_slots import CourtGeometry

    m_to_px = np.array([[40.0, 0.0, 600.0], [0.0, -25.0, 400.0], [0.0, 0.0, 1.0]])
    geom = CourtGeometry(np.array([380.0, 400.0]), np.array([820.0, 400.0]), np.linalg.inv(m_to_px), "t")
    result, _seen = _run_engine(tmp_path, monkeypatch, save_player_tracks=True, geometry=geom)
    doc = json.loads(Path(result.diagnostics["points_json"]).read_text(encoding="utf-8"))
    row = doc["points"][0]
    assert row["server"] == "near"
    assert row["serve_side"] == "deuce" and row["serve_side_source"] == "server"
    assert row["serve_side_visual_confidence"] > 0.9
    assert doc["serve_side_check"]["n_visual"] == 1
    with np.load(result.diagnostics["players_npz"]) as d:
        saved = json.loads(str(d["geometry_json"]))
    np.testing.assert_allclose(saved["homography_px_to_m"], np.linalg.inv(m_to_px))
    assert saved["net_line_px"] == [[380.0, 400.0], [820.0, 400.0]]


def test_tee_player_records_keeps_only_kept_frames():
    from rallyclip_engine.models import _tee_player_records

    recs = [({}, 0, "n0", "f0"), ({}, -1, "nx", "fx"), ({}, 1, "n1", None)]
    sink: list = []
    assert list(_tee_player_records(iter(recs), sink)) == recs
    assert sink == [("n0", "f0", None), ("n1", None, None)]


def test_build_feature_stream_player_sink(tmp_path):
    from rallyclip_engine.models import build_feature_stream

    class FakePose:
        def __init__(self, **kw):
            pass

        def iter_pose_frames(self, **kw):
            for i in range(6):
                yield {"boxes": np.array([[0, 0, 2 * i + 2, 4]], dtype=np.float32), "frame": "big"}

    class FakePre:
        def __init__(self, **kw):
            pass

        def compute_court_mask(self, vp):
            return "MASK", {}

        def _source_frame_shape(self, vp):
            return (720, 1280, 3)

        def iter_preprocess_frames(self, pose_stream, mask, w, h):
            for i, _fd in enumerate(pose_stream):  # one pose frame per record, like the real one
                yield ({}, -1 if i % 3 == 2 else 0, {"i": i}, None)

    class FakeFE:
        def __init__(self, **kw):
            pass

        def iter_build_features(self, records):
            for rec in records:
                if rec[1] >= 0:
                    yield np.zeros(3, dtype=np.float32), rec[1]

    deps = _deps()
    deps.PoseExtractor, deps.DataPreprocessor, deps.FeatureEngineer = FakePose, FakePre, FakeFE
    from rallyclip_engine.models import PlayerRecordSink

    sink = PlayerRecordSink()
    rows = list(build_feature_stream(_request(tmp_path), deps, None, None, player_sink=sink))
    assert len(rows) == len(sink) == 4
    assert [near["i"] for near, _far, _raw in sink] == [0, 1, 3, 4]
    # the raw pre-filter pose frame of the same index rides along (only pose keys kept)
    assert [float(raw["boxes"][0, 2]) for _n, _f, raw in sink] == [2.0, 4.0, 8.0, 10.0]
    assert set(sink[0][2]) == {"boxes"}
    assert sink.raw_scale == (1.0, 1.0)
    assert sink.geometry is None  # v1 / classic slots: no court geometry
    assert len(list(build_feature_stream(_request(tmp_path), deps, None, None))) == 4


def test_build_feature_stream_hands_netline_geometry_to_the_sink(tmp_path, monkeypatch):
    """v2 net-line runs compute the court geometry for the slots; the player sink gets
    the same object (serve side) without the feature stream changing."""
    from rallyclip_engine import models as engine_models
    from rallyclip_engine.models import PlayerRecordSink, build_feature_stream

    class FakePose:
        def __init__(self, **kw):
            pass

        def iter_pose_frames(self, **kw):
            assert "crop_window" in kw  # geometry present -> far-court crop pass requested
            yield {"boxes": np.zeros((0, 4), np.float32)}

    class FakePre:
        def __init__(self, **kw):
            pass

        def compute_court_mask(self, vp):
            return "MASK", {}

        def _source_frame_shape(self, vp):
            return (1080, 1920, 3)

        def iter_slot_players(self, pose_stream, mask, w, h, geometry):
            seen["slots_geometry"] = geometry
            for _fd in pose_stream:
                yield 0, {"near": None, "far": None}

    seen = {}
    # No far-baseline corners -> the far crop falls back to the fixed window.
    geom = types.SimpleNamespace(far_left=None, far_right=None, net_a=None, net_b=None)
    monkeypatch.setattr(engine_models, "_court_geometry", lambda request, cfg, size: geom)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"feature_pipeline": {"feature_set": "v2_nearfar", "slot_mode": "netline"}}),
                        encoding="utf-8")
    deps = _deps()
    deps.PoseExtractor, deps.DataPreprocessor = FakePose, FakePre
    req = _request(tmp_path, manifest_path=manifest)
    req = dataclasses.replace(req, feature_set="v2_nearfar", screen_width=1920, screen_height=1080)
    sink = PlayerRecordSink()
    list(build_feature_stream(req, deps, None, None, player_sink=sink))
    assert seen["slots_geometry"] is geom and sink.geometry is geom and len(sink) == 1


def test_run_result_payload_player_keys_only_when_present():
    from rallyclip_api.serialization import run_result_payload

    base = RunResult(frame_segments=[(1, 2)], intervals_sec=[(0.2, 0.4)], diagnostics={"pipeline_id": "x"})
    assert set(run_result_payload(base)) == {"pipeline_id", "intervals", "csv_path", "video_path", "n_segments"}
    rich = RunResult(frame_segments=[], intervals_sec=[], diagnostics={
        "pipeline_id": "x", "players_npz": "/o/m_players.npz", "points_json": "/o/m_points.json"})
    payload = run_result_payload(rich)
    assert payload["players_npz"] == "/o/m_players.npz" and payload["points_json"] == "/o/m_points.json"


@pytest.mark.parametrize("flag,expected", [([], False), (["--players"], True)])
def test_cli_players_flag(monkeypatch, tmp_path, flag, expected):
    import sys

    from helpers.module_stubs import import_cli_main_with_stubs

    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rallyclip", "--video", str(tmp_path / "m.mp4"), *flag])
    seen = {}
    monkeypatch.setattr(cli_main, "build_run_config", lambda args: seen.setdefault("args", args))
    monkeypatch.setattr(cli_main, "run_pipeline", lambda cfg: 0)
    assert cli_main.main() == 0
    assert bool(seen["args"].save_player_tracks) is expected


# ---------------------------------------------------------------- far player from raw


def _raw_frame(dets):
    """dets: list of player dicts -> raw pose frame dict (reference pixels)."""
    return {
        "boxes": np.stack([d["box"] for d in dets]) if dets else np.empty((0, 4), np.float32),
        "keypoints": np.stack([d["keypoints"] for d in dets]) if dets else np.empty((0, 17, 2), np.float32),
        "conf": np.stack([d["conf"] for d in dets]) if dets else np.empty((0, 17), np.float32),
        "box_conf": np.linspace(0.9, 0.5, len(dets)).astype(np.float32),
    }


def test_pick_far_player_from_raw_detections():
    from analysis.players import RawDetections, pick_far_player

    n = 30
    near = [_pose(**NEAR) for _ in range(n)]
    spectator = dict(cx=60.0, foot_y=470.0, h=120.0)  # image side, mid height: not the far player
    frames = []
    for i in range(n):
        dets = [near[i], _pose(**spectator)]
        if i % 5 != 4:  # far player detected in 4 of 5 frames
            dets.append(_pose(cx=600.0 + i, foot_y=250.0, h=90.0))
        frames.append(_raw_frame(dets))
    raw = RawDetections.from_frames(frames)
    assert raw.count.tolist() == [3 if i % 5 != 4 else 2 for i in range(n)]
    near_side = pack_player_records(near)
    far = pick_far_player(raw, near_side)
    assert far.present.tolist() == [i % 5 != 4 for i in range(n)]
    cx = (far.box[far.present, 0] + far.box[far.present, 2]) / 2
    assert np.all(np.abs(cx - 600.0) < 40)

    # PlayerTracks: raw keys -> far re-derived; to_arrays keeps the shipped far_* keys
    tracks = PlayerTracks.from_records(near, [None] * n, fps=FPS, raw=raw)
    assert tracks.far_source == "raw" and tracks.far.present.sum() == 24
    arrays = tracks.to_arrays()
    assert not arrays["far_present"].any() and arrays["raw_boxes"].shape == (n, 8, 4)
    back = PlayerTracks.from_arrays(arrays)
    assert back.far_source == "raw"
    np.testing.assert_array_equal(back.far.present, tracks.far.present)


def test_pick_far_player_rejects_side_people_and_falls_back():
    from analysis.players import RawDetections, pick_far_player

    near = [_pose(**NEAR) for _ in range(10)]
    frames = [_raw_frame([near[i], _pose(cx=1220.0, foot_y=480.0, h=110.0)]) for i in range(10)]
    far = pick_far_player(RawDetections.from_frames(frames), pack_player_records(near))
    assert not far.present.any()
    # no raw keys -> shipped far record is used as-is
    shipped = PlayerTracks.from_records(near, [_pose(**FAR)] * 10, fps=FPS)
    assert shipped.far_source == "shipped" and shipped.far.present.all()


def test_shipped_far_record_wins_over_raw_pick_per_frame():
    """v0.6.0 net-line slots fill the far slot from the crop pass: keep it where present
    and only fill the frames it misses from the raw full-frame pick."""
    from analysis.players import RawDetections

    n = 20
    near = [_pose(**NEAR) for _ in range(n)]
    frames = [_raw_frame([near[i], _pose(cx=600.0 + i, foot_y=250.0, h=90.0)]) for i in range(n)]
    shipped = [_pose(**FAR) if i < 10 else None for i in range(n)]
    tracks = PlayerTracks.from_records(near, shipped, fps=FPS, raw=RawDetections.from_frames(frames))
    assert tracks.far_source == "shipped+raw"
    assert tracks.far.present.all()
    shipped_side = pack_player_records(shipped)
    np.testing.assert_array_equal(tracks.far.box[:10], shipped_side.box[:10])
    cx = (tracks.far.box[10:, 0] + tracks.far.box[10:, 2]) / 2
    assert np.all(np.abs(cx - (600.0 + np.arange(10, n))) < 1e-3)
    # the npz keeps the shipped record as far_*; reloading re-derives the same merge
    back = PlayerTracks.from_arrays(tracks.to_arrays())
    assert back.far_source == "shipped+raw"
    np.testing.assert_array_equal(back.far.present, tracks.far.present)
    np.testing.assert_allclose(back.far.box, tracks.far.box)
