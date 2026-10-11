"""HeatmapHybridModel wiring + heatmap pipeline resolution.

Covers the engine-side glue that the pure-decode unit tests (test_heatmap_decode)
do not: manifest -> PipelineSpec resolution, dispatch to HeatmapHybridModel, and
the full model.run() producing float-second intervals + a segments CSV. The pose
feature stream and the ONNX runner are faked so this needs no video or model
artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rallyclip_core.contracts import RunRequest, RuntimeDeps
from rallyclip_core.pipelines import (
    FRAME_STARTEND_HEATMAP,
    resolve_pipeline_spec,
)
from rallyclip_engine import models as engine_models
from rallyclip_engine.models import HeatmapHybridModel, build_analysis_model

FPS = 5.0


# --------------------------------------------------------- pipeline resolution


def _write_heatmap_manifest(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"not a real onnx")  # only the suffix matters here
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "pipeline": {"id": "frame_startend_heatmap"},
                "feature_pipeline": {"feature_set": "v1", "target_fps": FPS},
                "inference": {"seq_len_frames": 100, "overlap_frames": 50},
                "model": {"outputs": ["pointness_logit", "start_heatmap_logit", "end_heatmap_logit"]},
                "postprocess": {
                    "method": "heatmap_hybrid",
                    "params": {
                        "mode": "hybrid",
                        "threshold": 0.5,
                        "sigma_frames": 2.5,
                        "min_duration_sec": 0.3,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path, manifest_path


def test_resolve_pipeline_spec_from_heatmap_manifest(tmp_path):
    model_path, manifest_path = _write_heatmap_manifest(tmp_path)
    spec = resolve_pipeline_spec(model_path, manifest_path)
    assert spec.pipeline_id == FRAME_STARTEND_HEATMAP
    assert spec.model_output == "pointness_start_end_heatmap"
    assert spec.decode_method == "heatmap_hybrid"
    assert spec.feature_set == "v1"
    # decode knobs flow through from postprocess.params
    assert spec.params["mode"] == "hybrid"
    assert spec.params["sigma_frames"] == 2.5
    assert spec.params["min_duration_sec"] == 0.3


def test_resolve_pipeline_spec_via_method_only(tmp_path):
    # No explicit pipeline.id -> the postprocess.method selects the heatmap pipeline.
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_pipeline": {"feature_set": "v1", "target_fps": FPS},
                "inference": {"seq_len_frames": 100, "overlap_frames": 50},
                "postprocess": {"method": "heatmap_hybrid", "params": {"threshold": 0.4}},
            }
        ),
        encoding="utf-8",
    )
    spec = resolve_pipeline_spec(model_path, manifest_path)
    assert spec.pipeline_id == FRAME_STARTEND_HEATMAP
    assert spec.params["threshold"] == 0.4


def test_heatmap_override_incompatible_with_classic_manifest(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_pipeline": {"feature_set": "v1", "target_fps": FPS},
                "inference": {"seq_len_frames": 100, "overlap_frames": 50},
                "postprocess": {"method": "hysteresis", "params": {"sigma": 1.5, "low": 0.5, "high": 0.65}},
            }
        ),
        encoding="utf-8",
    )
    from rallyclip_core.contracts import UnsupportedPipelineError

    with pytest.raises(UnsupportedPipelineError):
        resolve_pipeline_spec(model_path, manifest_path, override_pipeline_id=FRAME_STARTEND_HEATMAP)


# ----------------------------------------------------------- full model.run()


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


def _make_request(tmp_path: Path, *, write_csv=True, segment_video=False, start_time: int = 0) -> RunRequest:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    return RunRequest(
        video_path=tmp_path / "match.mp4",
        output_dir=tmp_path / "out",
        output_name="match",
        csv_output_dir=tmp_path / "out",
        write_csv=write_csv,
        segment_video=segment_video,
        yolo_weights="yolov8n-pose-960-dynamic.onnx",
        yolo_device=None,
        model_path=model_path,
        scaler_path=tmp_path / "scaler.json",
        fps=FPS,
        seq_len=10,
        imgsz=960,
        conf=0.25,
        feature_set="v1",
        screen_width=1280,
        screen_height=720,
        overlap=5,
        sigma=0.0,
        low=0.0,
        high=0.0,
        min_dur_sec=0.0,
        pipeline_id=FRAME_STARTEND_HEATMAP,
        start_time=start_time,
    )


def _fake_deps(segment_calls=None) -> RuntimeDeps:
    def fake_segment_video(video, intervals, out):
        if segment_calls is not None:
            segment_calls.append((video, intervals, out))

    return RuntimeDeps(
        np=np,
        PoseExtractor=None,
        DataPreprocessor=None,
        FeatureEngineer=None,
        load_scaler_asset=lambda path: _IdentityScaler(),
        load_model_from_checkpoint=None,
        run_windowed_inference_average_onnx_stream=None,
        run_windowed_inference_average_torch_stream=None,
        gaussian_filter1d=None,
        hysteresis_threshold=None,
        extract_segments_from_binary=None,
        write_segments_csv=lambda *a, **k: None,
        segment_video=fake_segment_video,
    )


def _spec(tmp_path, params=None):
    _model, manifest = _write_heatmap_manifest(tmp_path)
    spec = resolve_pipeline_spec(tmp_path / "model.onnx", manifest)
    if params is not None:
        # rebuild with overridden decode params
        from dataclasses import replace

        spec = replace(spec, params=params)
    return spec


def _install_fakes(monkeypatch, tracks):
    # bypass the heavy pose->feature stream
    monkeypatch.setattr(
        engine_models, "build_feature_stream",
        lambda request, deps, pcb, ccb: [(np.zeros(4, dtype=np.float32), 0) for _ in range(len(tracks[0]))],
    )
    # the ONNX runner returns our synthetic (3, N) tracks regardless of inputs
    import infer

    monkeypatch.setattr(
        infer, "run_multitrack_windowed_inference_onnx_stream",
        lambda *a, **k: np.asarray(tracks, dtype=np.float32),
    )


def test_heatmap_model_run_produces_intervals_and_csv(tmp_path, monkeypatch):
    n = 20
    point = np.full(n, 0.1, dtype=np.float32)
    point[5:11] = 0.9  # run frames 5..10
    start = np.zeros(n, dtype=np.float32)
    start[5] = 0.9
    end = np.zeros(n, dtype=np.float32)
    end[10] = 0.9
    _install_fakes(monkeypatch, np.stack([point, start, end], axis=0))

    request = _make_request(tmp_path)
    spec = _spec(tmp_path)
    model = build_analysis_model(request, spec, _fake_deps())
    assert isinstance(model, HeatmapHybridModel)

    result = model.run()

    assert result.intervals_sec == [pytest.approx((5 / FPS, 10 / FPS))]
    assert result.frame_segments == [(5, 10)]  # rounded, for RunResult consumers
    assert result.diagnostics["pipeline_id"] == FRAME_STARTEND_HEATMAP
    # CSV written in the segments.csv contract, sub-frame precision preserved
    csv_text = (tmp_path / "out" / "match_segments.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "start_time,end_time"
    assert csv_text.splitlines()[1] == "1.000,2.000"


def test_heatmap_model_offsets_timestamps_by_request_start_time(tmp_path, monkeypatch):
    n = 20
    point = np.full(n, 0.1, dtype=np.float32)
    point[5:11] = 0.9
    start = np.zeros(n, dtype=np.float32)
    start[5] = 0.9
    end = np.zeros(n, dtype=np.float32)
    end[10] = 0.9
    _install_fakes(monkeypatch, np.stack([point, start, end], axis=0))

    request = _make_request(tmp_path, start_time=12)
    result = build_analysis_model(request, _spec(tmp_path), _fake_deps()).run()

    assert result.intervals_sec == [pytest.approx((12 + 5 / FPS, 12 + 10 / FPS))]
    csv_text = (tmp_path / "out" / "match_segments.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[1] == "13.000,14.000"


def test_heatmap_model_segments_video_from_intervals(tmp_path, monkeypatch):
    n = 16
    point = np.full(n, 0.1, dtype=np.float32)
    point[4:9] = 0.9
    start = np.zeros(n, dtype=np.float32)
    start[4] = 0.9
    end = np.zeros(n, dtype=np.float32)
    end[8] = 0.9
    _install_fakes(monkeypatch, np.stack([point, start, end], axis=0))

    calls = []
    request = _make_request(tmp_path, write_csv=False, segment_video=True)
    model = build_analysis_model(request, _spec(tmp_path), _fake_deps(segment_calls=calls))
    result = model.run()

    assert len(calls) == 1
    _video, intervals, _out = calls[0]
    assert intervals == result.intervals_sec == [pytest.approx((4 / FPS, 8 / FPS))]


def test_heatmap_model_uses_injected_deps_over_module_import(tmp_path, monkeypatch):
    # When RuntimeDeps carries the heatmap slots (as the engine's default loader
    # wires them), the model must use them -- not the bare `infer` module import.
    n = 20
    point = np.full(n, 0.1, dtype=np.float32)
    point[5:11] = 0.9
    start = np.zeros(n, dtype=np.float32)
    start[5] = 0.9
    end = np.zeros(n, dtype=np.float32)
    end[10] = 0.9
    tracks = np.stack([point, start, end], axis=0)

    monkeypatch.setattr(
        engine_models, "build_feature_stream",
        lambda request, deps, pcb, ccb: [(np.zeros(4, dtype=np.float32), 0) for _ in range(n)],
    )
    # Deliberately do NOT patch the infer module: if the model bypassed deps and
    # imported the real runner, it would choke on the fake model bytes.
    from dataclasses import replace as dc_replace

    from infer import HeatmapDecodeConfig, decode_heatmap_segments

    ran_via_deps = []

    def fake_runner(*a, **k):
        ran_via_deps.append(True)
        return tracks

    deps = _fake_deps()
    deps.run_multitrack_windowed_inference_onnx_stream = fake_runner
    deps.decode_heatmap_segments = decode_heatmap_segments
    deps.heatmap_decode_config_cls = HeatmapDecodeConfig

    request = _make_request(tmp_path, write_csv=False)
    model = build_analysis_model(request, _spec(tmp_path), deps)
    result = model.run()

    assert ran_via_deps == [True]
    assert result.intervals_sec == [pytest.approx((5 / FPS, 10 / FPS))]


def test_heatmap_model_rejects_non_onnx(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, np.zeros((3, 12), dtype=np.float32))
    from rallyclip_core.contracts import UnsupportedPipelineError

    request = _make_request(tmp_path)
    object.__setattr__(request, "model_path", tmp_path / "model.pth")  # frozen dataclass
    model = build_analysis_model(request, _spec(tmp_path), _fake_deps())
    with pytest.raises(UnsupportedPipelineError, match="ONNX models only"):
        model.run()


# ------------------------------------------- per-run decode overrides (request > manifest)


def _decode_cfg_for(tmp_path, decode_params, *, spec_params=None):
    from dataclasses import replace as dc_replace

    from infer import HeatmapDecodeConfig

    request = dc_replace(_make_request(tmp_path), decode_params=decode_params)
    deps = _fake_deps()
    deps.heatmap_decode_config_cls = HeatmapDecodeConfig
    model = build_analysis_model(request, _spec(tmp_path, params=spec_params), deps)
    return model._decode_config()


def test_decode_config_without_overrides_is_manifest_only(tmp_path):
    cfg = _decode_cfg_for(tmp_path, None)
    assert cfg.threshold == 0.5
    assert cfg.sigma_frames == 2.5
    assert cfg.min_duration_sec == 0.3
    assert cfg.end_pad_sec == 0.0
    assert cfg.merge_gap_sec == 0.0


def test_decode_config_request_overrides_win_over_manifest(tmp_path):
    cfg = _decode_cfg_for(
        tmp_path,
        {"threshold": 0.7, "end_pad_sec": 1.5, "merge_gap_sec": 1.0},
        spec_params={"mode": "hybrid", "threshold": 0.5, "merge_gap_sec": 0.5, "sigma_frames": 2.5},
    )
    assert cfg.threshold == 0.7  # request wins
    assert cfg.end_pad_sec == 1.5  # absent from manifest, set by request
    assert cfg.merge_gap_sec == 1.0
    assert cfg.sigma_frames == 2.5  # untouched manifest value survives


def test_decode_config_ignores_none_and_unknown_override_keys(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        cfg = _decode_cfg_for(tmp_path, {"threshold": None, "not_a_field": 3, "end_pad_sec": 1.0})
    assert cfg.threshold == 0.5  # None does not clobber the manifest value
    assert cfg.end_pad_sec == 1.0
    assert any("not_a_field" in r.getMessage() for r in caplog.records)


def test_decode_overrides_change_run_output(tmp_path, monkeypatch):
    # End-to-end through model.run(): threshold override drops the weak run,
    # end_pad extends the strong one.
    n = 40
    point = np.full(n, 0.1, dtype=np.float32)
    point[5:11] = 0.9
    point[20:26] = 0.6
    start = np.zeros(n, dtype=np.float32)
    start[[5, 20]] = 0.9
    end = np.zeros(n, dtype=np.float32)
    end[[10, 25]] = 0.9
    _install_fakes(monkeypatch, np.stack([point, start, end], axis=0))
    from dataclasses import replace as dc_replace

    base = build_analysis_model(_make_request(tmp_path, write_csv=False), _spec(tmp_path), _fake_deps()).run()
    assert len(base.intervals_sec) == 2

    request = dc_replace(
        _make_request(tmp_path, write_csv=False),
        decode_params={"threshold": 0.7, "end_pad_sec": 1.0},
    )
    result = build_analysis_model(request, _spec(tmp_path), _fake_deps()).run()
    assert result.intervals_sec == [pytest.approx((5 / FPS, 10 / FPS + 1.0))]


def test_classic_pipeline_ignores_decode_overrides(tmp_path):
    # The hysteresis spec has no heatmap decode; decode_params must not reach it.
    from dataclasses import replace as dc_replace

    from rallyclip_core.contracts import PipelineSpec
    from rallyclip_core.pipelines import FRAME_PROBABILITY_HYSTERESIS

    request = dc_replace(
        _make_request(tmp_path), pipeline_id=FRAME_PROBABILITY_HYSTERESIS,
        sigma=1.0, low=0.45, high=0.7, min_dur_sec=1.0,
        decode_params={"threshold": 0.7, "end_pad_sec": 1.0},
    )
    spec = PipelineSpec(
        pipeline_id=FRAME_PROBABILITY_HYSTERESIS, feature_set="v1",
        model_output="frame_probability", decode_method="gaussian_hysteresis",
        params={"sigma": 1.0, "low": 0.45, "high": 0.7, "min_dur_sec": 1.0},
    )
    from infer import extract_segments_from_binary, gaussian_filter1d, hysteresis_threshold

    deps = _fake_deps()
    deps.gaussian_filter1d = gaussian_filter1d
    deps.hysteresis_threshold = hysteresis_threshold
    deps.extract_segments_from_binary = extract_segments_from_binary
    model = build_analysis_model(request, spec, deps)
    assert not isinstance(model, HeatmapHybridModel)
    probs = np.zeros(30, dtype=np.float32)
    probs[5:15] = 0.9
    plain = build_analysis_model(dc_replace(request, decode_params=None), spec, deps)
    # Runs without error and the overrides change nothing on the classic decode.
    segs = model.postprocess(probs)
    assert segs and segs == plain.postprocess(probs)


def test_coerce_decode_overrides_validates_and_filters():
    from rallyclip_core.contracts import coerce_decode_overrides

    assert coerce_decode_overrides(None) is None
    assert coerce_decode_overrides({}) is None
    assert coerce_decode_overrides({"threshold": None, "sigma": 1.0}) is None  # unknown/None dropped
    assert coerce_decode_overrides({"threshold": "0.65", "end_pad_sec": 1}) == {
        "threshold": 0.65, "end_pad_sec": 1.0,
    }
    for bad in ({"threshold": 1.5}, {"threshold": "abc"}, {"end_pad_sec": -1.0},
                {"merge_gap_sec": float("nan")}, {"threshold": True}):
        with pytest.raises(ValueError):
            coerce_decode_overrides(bad)
