from __future__ import annotations

import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from rallyclip_core.contracts import (
    CancelCheck,
    FrameSegment,
    Interval,
    PipelineSpec,
    ProgressCallback,
    ProgressEvent,
    RunRequest,
    RunResult,
    RuntimeDeps,
    UnsupportedPipelineError,
)
from rallyclip_core.intervals import frame_segments_to_intervals, write_point_intervals
from rallyclip_core.pipelines import (
    FRAME_PROBABILITY_HYSTERESIS,
    FRAME_STARTEND_HEATMAP,
    START_END_ATTENTION_VOTING,
)


def _emit(
    callback: Optional[ProgressCallback],
    stage: str,
    progress: int,
    status: str = "in_progress",
    metadata: Optional[dict] = None,
) -> None:
    if callback is not None:
        callback(ProgressEvent(stage=stage, progress=progress, status=status, metadata=metadata or {}))


def _check(cancel_check: Optional[CancelCheck]) -> None:
    if cancel_check is not None:
        cancel_check()


def _estimate_stream_window_count(num_frames: int, sequence_length: int, overlap: int) -> Optional[int]:
    if num_frames < sequence_length or sequence_length <= 0 or overlap < 0 or overlap >= sequence_length:
        return None
    step = sequence_length - overlap
    count = 1 + max(0, (num_frames - sequence_length) // step)
    last_start = (count - 1) * step
    if last_start + sequence_length < num_frames:
        count += 1
    return count


_RAW_POSE_KEYS = ("boxes", "keypoints", "conf", "box_conf")


class PlayerRecordSink(list):
    """Opt-in player tee target: one (near, far, raw_pose_frame) per kept frame.
    ``raw_scale`` maps the raw (source-pixel) detections to the reference frame;
    ``geometry`` is the run's court geometry (net-line slots), None without one."""

    raw_scale = (1.0, 1.0)
    geometry = None


def _tee_pose_frames(pose_stream, holder: dict):
    """Remember the pose frame the preprocessor is currently on (it consumes one pose
    frame per yielded record, no lookahead), so the record tee can attach the raw,
    pre-court-filter detections to it."""
    for frame_data in pose_stream:
        holder["last"] = frame_data
        yield frame_data


def _raw_pose_frame(holder: Optional[dict]):
    raw = holder.get("last") if holder is not None else None
    if isinstance(raw, dict):
        return {k: raw[k] for k in _RAW_POSE_KEYS if k in raw}
    return None


def _tee_player_records(records, sink: list, holder: Optional[dict] = None):
    """v1 path: pass preprocessed records through unchanged, appending (near, far, raw)
    for every kept frame (target >= 0) -- exactly the frames FeatureEngineer turns into rows."""
    for rec in records:
        if rec[1] >= 0:
            sink.append((rec[2], rec[3], _raw_pose_frame(holder)))
        yield rec


def _tee_slot_players(slot_stream, sink: list, holder: Optional[dict] = None):
    """v2 path: pass ``(status, {slot: player})`` through unchanged, appending
    (near, far, raw) for every kept frame (status >= 0) -- exactly the frames
    _iter_v2_features turns into rows. Players are in reference pixels (screen size);
    with net-line slots the far player comes from the crop pass. raw is the
    full-frame, pre-court-filter detections (the crop detections are not kept)."""
    for status, players in slot_stream:
        if status >= 0:
            players = players or {}
            sink.append((players.get("near"), players.get("far"), _raw_pose_frame(holder)))
        yield status, players


def build_feature_stream(
    request: RunRequest,
    deps: RuntimeDeps,
    progress_callback: Optional[ProgressCallback],
    cancel_check: Optional[CancelCheck],
    player_sink: Optional[list] = None,
):
    """Shared source-video -> feature-row stream.

    pose (YOLO) -> court mask -> per-frame preprocess -> feature engineering,
    yielding the v1 362-dim feature rows. Head-agnostic: the frame-probability
    and boundary-heatmap pipelines consume identical features, differing only in
    the model head + decode, so this lives at module scope rather than in any one
    AnalysisModel.

    ``player_sink`` (opt-in, RunRequest.save_player_tracks): a list that receives one
    ``(near_player, far_player, raw_pose_frame)`` per kept frame, 1:1 with the feature
    rows; raw_pose_frame is the pre-court-filter detections in source pixels (the
    court filter drops the far player), scaled via ``PlayerRecordSink.raw_scale``.
    """
    feature_set = str(request.feature_set)
    feature_cfg = _manifest_feature_pipeline(request)
    if feature_set != "v1":
        from features.v2 import FEATURE_SETS

        if feature_set not in FEATURE_SETS:
            raise UnsupportedPipelineError(
                f"This model declares feature_set='{feature_set}', which the runtime "
                f"feature pipeline does not implement (have: v1, {', '.join(sorted(FEATURE_SETS))})."
            )
    slot_mode = str(feature_cfg.get("slot_mode") or "classic")
    if slot_mode not in {"classic", "netline"} or (slot_mode == "netline" and feature_set == "v1"):
        raise UnsupportedPipelineError(f"Unsupported slot_mode='{slot_mode}' for feature_set='{feature_set}'.")

    yolo_weights = request.yolo_weights
    if not Path(yolo_weights).is_absolute():
        # Resolve bare weight names (from the manifest) against the models
        # directory and the manifest's own directory, so every consumer —
        # court detection included — gets a concrete file path. Names that
        # don't resolve stay bare and fall through to ultralytics' own
        # download/resolution, as before.
        search_dirs = [Path(request.models_dir or (Path.cwd() / "models"))]
        if request.manifest_path is not None:
            search_dirs.append(Path(request.manifest_path).parent)
        for base in search_dirs:
            candidate = base / yolo_weights
            if candidate.exists():
                yolo_weights = str(candidate)
                break
    pose_device = None
    if request.yolo_device and deps.apply_pose_device is not None:
        pose_device = deps.apply_pose_device(str(request.yolo_device), model_path=yolo_weights, set_env=False)
    elif deps.apply_pose_device is not None:
        pose_device = deps.apply_pose_device(None, model_path=yolo_weights, set_env=False)

    # Court detection runs a handful of frames; CoreML only accelerates the
    # pose loop, so keep the preprocessor's YOLO on CPU for that device.
    court_device = "cpu" if pose_device == "coreml" else pose_device
    pre = deps.DataPreprocessor(
        screen_width=int(request.screen_width),
        screen_height=int(request.screen_height),
        save_court_masks=False,
        yolo_model_path=yolo_weights,
        conf=float(request.conf),
        **({"yolo_device": court_device} if court_device is not None else {}),
    )
    _check(cancel_check)
    _emit(progress_callback, "pose", 1)
    court_mask, _ = pre.compute_court_mask(str(request.video_path))
    _emit(progress_callback, "pose", 3)

    _check(cancel_check)
    extractor_kwargs = {
        "model_path": yolo_weights,
        "model_dir": str(request.models_dir or (Path.cwd() / "models")),
    }
    if pose_device is not None:
        extractor_kwargs["device"] = pose_device
    extractor = deps.PoseExtractor(**extractor_kwargs)

    def pose_progress(frac: float, meta=None) -> None:
        _check(cancel_check)
        # meta (frames_seen/frames_total/smoothed fps) feeds client-side
        # ETA display; forward it rather than dropping it here.
        _emit(progress_callback, "pose", int(3 + max(0.0, min(1.0, frac)) * 96), metadata=meta or {})

    src_height, src_width, _ = pre._source_frame_shape(str(request.video_path))
    geometry = None
    if slot_mode == "netline":
        geometry = _court_geometry(request, feature_cfg, (int(request.screen_width), int(request.screen_height)))
        _emit(progress_callback, "pose", 3)
    pose_kwargs = {}
    if geometry is not None:
        pose_kwargs.update(_far_crop_kwargs(geometry, feature_cfg, (int(request.screen_width), int(request.screen_height))))
    pose_frames = extractor.iter_pose_frames(
        video_path=str(request.video_path),
        confidence_threshold=float(request.conf),
        start_time_seconds=int(request.start_time),
        duration_seconds=int(request.duration),
        target_fps=int(request.fps),
        imgsz=int(request.imgsz),
        annotations_csv=None,
        progress_callback=pose_progress,
        **pose_kwargs,
    )
    holder: Optional[dict] = None
    pose_stream = pose_frames
    if player_sink is not None:
        holder = {}
        pose_stream = _tee_pose_frames(pose_frames, holder)
        if isinstance(player_sink, PlayerRecordSink):
            player_sink.geometry = geometry  # serve side (deuce/ad) needs the court homography
            if src_width and src_height:
                player_sink.raw_scale = (
                    float(request.screen_width) / float(src_width),
                    float(request.screen_height) / float(src_height),
                )
    if feature_set != "v1":
        slot_stream = pre.iter_slot_players(pose_stream, court_mask, src_width, src_height, geometry)
        if player_sink is not None:
            slot_stream = _tee_slot_players(slot_stream, player_sink, holder)
        feature_stream = _iter_v2_features(
            slot_stream,
            feature_set, feature_cfg, int(request.screen_width), int(request.screen_height), float(request.fps),
        )
        _emit(progress_callback, "preprocess", 1)
        _emit(progress_callback, "feature", 1)
        return feature_stream
    preprocessed_stream = pre.iter_preprocess_frames(pose_stream, court_mask, src_width, src_height)
    if player_sink is not None:
        preprocessed_stream = _tee_player_records(preprocessed_stream, player_sink, holder)
    fe = deps.FeatureEngineer(
        screen_width=int(request.screen_width),
        screen_height=int(request.screen_height),
        target_fps=float(request.fps),
    )
    feature_stream = fe.iter_build_features(preprocessed_stream)
    _emit(progress_callback, "preprocess", 1)
    _emit(progress_callback, "feature", 1)
    return feature_stream


def _manifest_path(request: RunRequest) -> Optional[Path]:
    # The GUI does not pass manifest_path; the manifest sits next to the model.
    if request.manifest_path is not None:
        return Path(request.manifest_path)
    candidate = Path(request.model_path).parent / "manifest.json"
    return candidate if candidate.exists() else None


def _manifest_feature_pipeline(request: RunRequest) -> dict:
    import json

    path = _manifest_path(request)
    if path is None or not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")).get("feature_pipeline") or {})
    except Exception:
        return {}


def _far_crop_kwargs(geometry, feature_cfg: dict, reference_size) -> dict:
    """iter_pose_frames crop-pass kwargs: the geometry-driven, upscaled far-court
    window when the court model found the far baseline, else the fixed window.

    manifest feature_pipeline.far_crop: {"mode": "adaptive" (default) | "fixed",
    plus any extraction.crop_pass.ADAPTIVE_DEFAULTS key}. The fixed fallback
    passes only crop_window, so it is exactly the pre-adaptive crop pass.
    """
    import logging

    from extraction.crop_pass import CROP_WINDOW, adaptive_crop

    fixed = {"crop_window": tuple(feature_cfg.get("crop_window") or CROP_WINDOW)}
    cfg = dict(feature_cfg.get("far_crop") or {})
    mode = str(cfg.pop("mode", "adaptive"))
    if mode == "fixed":
        return fixed
    if mode != "adaptive":
        raise UnsupportedPipelineError(f"Unsupported far_crop mode='{mode}'.")
    spec = adaptive_crop(geometry.far_left, geometry.far_right, geometry.net_a, geometry.net_b, reference_size, **cfg)
    if spec is None:
        logging.warning("Far-court crop: no far baseline in the court geometry; using the fixed window.")
        return fixed
    logging.info(
        "Far-court crop: window %s (reference px), %.2fx -> %dx%d, imgsz %d",
        [round(v * s) for v, s in zip(spec.window, reference_size * 2)],
        spec.scale, spec.out_size[0], spec.out_size[1], spec.imgsz,
    )
    return {"crop_window": spec.window, "crop_out_size": spec.out_size, "crop_imgsz": spec.imgsz}


def _court_geometry(request: RunRequest, feature_cfg: dict, reference_size):
    """Net line + homography in reference pixels, or None (classic slots fallback)."""
    import logging

    from preprocessing.court_lines import compute_court_geometry
    from preprocessing.netline_slots import CourtGeometry

    name = feature_cfg.get("court_model_file")
    manifest = _manifest_path(request)
    model = (manifest.parent if manifest else Path(request.model_path).parent) / str(name) if name else None
    if model is None or not model.exists():
        raise UnsupportedPipelineError(f"slot_mode=netline needs the court model; not found: {model}")
    data = compute_court_geometry(str(request.video_path), str(model), out_size=reference_size)
    if data is None:
        logging.warning("No court geometry; falling back to classic near/far player slots.")
        return None
    return CourtGeometry.from_dict(data)


def _iter_v2_features(slot_stream, feature_set: str, feature_cfg: dict, width: int, height: int, fps: float):
    """(annotation_status, {slot: player}) records -> (feature_vector, target) for kept frames."""
    from features.v2 import FEATURE_SETS, V2FeatureStream, kept_columns, pack_player

    builder = FEATURE_SETS[feature_set](screen_width=width, screen_height=height)
    keep = None
    drop_groups = tuple(feature_cfg.get("drop_feature_groups") or ())
    drop_slots = tuple(feature_cfg.get("drop_slots") or ())
    if drop_groups or drop_slots:
        keep = kept_columns(builder.slots, drop_groups, drop_slots)
    stream = V2FeatureStream(builder, 1.0 / fps, keep)
    for status, players in slot_stream:
        if status < 0:
            continue
        current = {}
        for slot in builder.slots:
            player = players.get(slot)
            # Same float32 arrays + existence rule the training features read back
            # from the preprocessed h5.
            current[slot] = None if player is None else pack_player(
                np.asarray(player["keypoints"], dtype=np.float32),
                np.asarray(player["conf"], dtype=np.float32),
                np.asarray(player["box"], dtype=np.float32),
                np.float32(player.get("box_conf", -1.0)),
            )
        yield stream.step(current), status


class AnalysisModel(ABC):
    """Artifact-bound analysis model.

    A model owns the full path from source video to point intervals: preprocessing,
    inference, and postprocessing. Different future artifacts can provide different
    implementations behind this same output contract.
    """

    pipeline_id: str

    def __init__(self, request: RunRequest, spec: PipelineSpec, deps: RuntimeDeps) -> None:
        self.request = request
        self.spec = spec
        self.deps = deps

    @abstractmethod
    def preprocess(self, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        features,
        progress_callback: Optional[ProgressCallback],
        cancel_check: Optional[CancelCheck],
    ):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_output) -> List[FrameSegment]:
        raise NotImplementedError

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> RunResult:
        features = self.preprocess(progress_callback, cancel_check)
        model_output = self.infer(features, progress_callback, cancel_check)
        frame_segments = self.postprocess(model_output)
        _emit(progress_callback, "inference", 100, "completed")
        return self._write_outputs(frame_segments)

    def _write_outputs(self, frame_segments: List[FrameSegment]) -> RunResult:
        request = self.request
        intervals_sec = frame_segments_to_intervals(frame_segments, float(request.fps))
        csv_path = None
        video_path = None
        base_name = request.output_name or request.video_path.stem
        if request.write_csv:
            request.csv_output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = request.csv_output_dir / f"{base_name}_segments.csv"
            self.deps.write_segments_csv(frame_segments, str(csv_path), fps=float(request.fps), overwrite=True)
        if request.segment_video:
            request.output_dir.mkdir(parents=True, exist_ok=True)
            video_path = request.output_dir / f"{base_name}_segmented.mp4"
            if intervals_sec:
                self.deps.segment_video(str(request.video_path), intervals_sec, str(video_path))
        return RunResult(
            frame_segments=frame_segments,
            intervals_sec=intervals_sec,
            csv_path=csv_path,
            video_path=video_path,
            diagnostics={"pipeline_id": self.pipeline_id},
        )

    def _write_interval_result(self, intervals_sec: List[Interval]) -> RunResult:
        """Finalize a model that yields floating-point second intervals directly
        (e.g. the heatmap head's sub-frame boundaries), preserving precision the
        frame-segment path would quantize away. RunResult.frame_segments is still
        populated (rounded) for consumers that count/scan segments."""
        request = self.request
        csv_path = None
        video_path = None
        base_name = request.output_name or request.video_path.stem
        if request.write_csv:
            request.csv_output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = request.csv_output_dir / f"{base_name}_segments.csv"
            write_point_intervals(csv_path, intervals_sec)
        if request.segment_video:
            request.output_dir.mkdir(parents=True, exist_ok=True)
            video_path = request.output_dir / f"{base_name}_segmented.mp4"
            if intervals_sec:
                self.deps.segment_video(str(request.video_path), intervals_sec, str(video_path))
        fps = float(request.fps)
        frame_segments: List[FrameSegment] = [
            (int(round(s * fps)), int(round(e * fps))) for s, e in intervals_sec
        ]
        return RunResult(
            frame_segments=frame_segments,
            intervals_sec=intervals_sec,
            csv_path=csv_path,
            video_path=video_path,
            diagnostics={"pipeline_id": self.pipeline_id},
        )


class FrameProbabilityHysteresisModel(AnalysisModel):
    pipeline_id = FRAME_PROBABILITY_HYSTERESIS

    def preprocess(self, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        return build_feature_stream(self.request, self.deps, progress_callback, cancel_check)

    def infer(
        self,
        features,
        progress_callback: Optional[ProgressCallback],
        cancel_check: Optional[CancelCheck],
    ):
        request = self.request
        deps = self.deps
        _emit(progress_callback, "inference", 5)
        scaler = deps.load_scaler_asset(str(request.scaler_path))

        duration_s = request.estimated_duration_s
        if duration_s is None:
            duration_s = float(request.duration) if 0 < int(request.duration) < 999999 else 0.0
        estimated_feature_rows = max(0, int(round(float(duration_s) * float(request.fps))) - 1)
        estimated_windows = _estimate_stream_window_count(
            estimated_feature_rows,
            int(request.seq_len),
            int(request.overlap),
        )
        feature_rows_seen = 0

        def scaled_feature_rows():
            nonlocal feature_rows_seen
            for feature_vector, _target in features:
                _check(cancel_check)
                feature_rows_seen += 1
                if estimated_feature_rows > 0:
                    frac = min(1.0, feature_rows_seen / float(estimated_feature_rows))
                    stage_progress = int(1 + frac * 94)
                    _emit(progress_callback, "preprocess", stage_progress)
                    _emit(progress_callback, "feature", stage_progress)
                    if estimated_windows is None:
                        _emit(progress_callback, "inference", int(5 + frac * 80))
                row = deps.np.asarray(feature_vector, dtype=deps.np.float32).reshape(1, -1)
                yield scaler.transform(row)[0].astype(deps.np.float32)

        def infer_progress(frac: float) -> None:
            _emit(progress_callback, "inference", int(5 + max(0.0, min(1.0, frac)) * 90))

        if request.model_path.suffix.lower() == ".onnx":
            output = deps.run_windowed_inference_average_onnx_stream(
                str(request.model_path),
                scaled_feature_rows(),
                sequence_length=int(request.seq_len),
                overlap=int(request.overlap),
                progress_callback=infer_progress,
                total_windows=estimated_windows,
            )
        else:
            model, device = deps.load_model_from_checkpoint(str(request.model_path), return_logits=False)
            output = deps.run_windowed_inference_average_torch_stream(
                model,
                device,
                scaled_feature_rows(),
                sequence_length=int(request.seq_len),
                overlap=int(request.overlap),
                progress_callback=infer_progress,
                total_windows=estimated_windows,
            )

        _emit(progress_callback, "pose", 100, "completed")
        _emit(progress_callback, "preprocess", 100, "completed")
        _emit(progress_callback, "feature", 100, "completed")
        return output

    def postprocess(self, model_output) -> List[FrameSegment]:
        deps = self.deps
        request = self.request
        smoothed_probs = deps.gaussian_filter1d(model_output.astype(deps.np.float32), sigma=float(request.sigma))
        min_duration_frames = int(round(max(0.0, float(request.min_dur_sec)) * float(request.fps)))
        binary_pred = deps.hysteresis_threshold(
            smoothed_probs,
            low=float(request.low),
            high=float(request.high),
            min_duration=min_duration_frames,
        )
        return list(deps.extract_segments_from_binary(binary_pred))


class HeatmapHybridModel(AnalysisModel):
    """Boundary-heatmap head. The model emits three per-frame tracks (pointness,
    startness, endness); pointness runs detect points (robust recall, same as the
    classic head) and the start/end heatmaps refine each edge to sub-frame
    precision via soft-argmax. Produces floating-point second intervals directly.
    ONNX-only in the shipped runtime."""

    pipeline_id = FRAME_STARTEND_HEATMAP

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> RunResult:
        features = self.preprocess(progress_callback, cancel_check)
        tracks = self.infer(features, progress_callback, cancel_check)
        intervals_sec = self.postprocess(tracks)
        _emit(progress_callback, "inference", 100, "completed")
        result = self._write_interval_result(intervals_sec)
        if getattr(self.request, "save_player_tracks", False):
            result = self._write_player_outputs(result, tracks)
        return result

    def preprocess(self, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        if not getattr(self.request, "save_player_tracks", False):
            return build_feature_stream(self.request, self.deps, progress_callback, cancel_check)
        self._player_records = PlayerRecordSink()
        return build_feature_stream(
            self.request, self.deps, progress_callback, cancel_check, player_sink=self._player_records
        )

    def _write_player_outputs(self, result: RunResult, tracks) -> RunResult:
        """Opt-in Phase 1 outputs next to the segments CSV: ``<name>_players.npz``
        (dump-compatible: pointness/start/end tracks + near/far players) and
        ``<name>_points.json`` (analysis.players.points_document; serve side from the
        sink's court geometry, saved as ``geometry_json``). Never fails the run:
        on error the diagnostics carry ``player_analysis_error`` instead."""
        request = self.request
        diagnostics = dict(result.diagnostics)
        try:
            from analysis.players import PlayerTracks, RawDetections, analyze_points, points_document
            from analysis.serve_side import geometry_to_dict

            np = self.deps.np
            records = getattr(self, "_player_records", None) or []
            pointness = np.asarray(tracks[0], dtype=np.float64)
            n = int(pointness.shape[0])
            if len(records) != n:
                logging.warning("player records (%d) != track frames (%d); truncating", len(records), n)
            n = min(n, len(records))
            raw_frames = [r[2] if len(r) > 2 else None for r in records[:n]]
            raw = None
            if any(f is not None for f in raw_frames):
                raw = RawDetections.from_frames(raw_frames, getattr(records, "raw_scale", (1.0, 1.0)))
            players = PlayerTracks.from_records(
                [r[0] for r in records[:n]],
                [r[1] for r in records[:n]],
                fps=float(request.fps),
                start_time=float(request.start_time),
                ref_size=(int(request.screen_width), int(request.screen_height)),
                raw=raw,
                geometry=geometry_to_dict(getattr(records, "geometry", None)),
            )
            base_name = request.output_name or request.video_path.stem
            out_dir = Path(request.csv_output_dir)
            npz_path = players.save_npz(
                out_dir / f"{base_name}_players.npz",
                pointness=pointness[:n],
                start_prob=np.asarray(tracks[1], dtype=np.float64)[:n],
                end_prob=np.asarray(tracks[2], dtype=np.float64)[:n],
                default_segments=np.asarray(result.intervals_sec, dtype=np.float64).reshape(-1, 2),
                params=str(dict(self.spec.params or {})),
            )
            rows = analyze_points(result.intervals_sec, players)
            points_path = out_dir / f"{base_name}_points.json"
            points_path.write_text(json.dumps(points_document(rows), indent=1), encoding="utf-8")
            diagnostics["players_npz"] = str(npz_path)
            diagnostics["points_json"] = str(points_path)
        except Exception as exc:  # pragma: no cover - defensive: analysis is an add-on
            logging.warning("Player analysis failed; segments are unaffected", exc_info=True)
            diagnostics["player_analysis_error"] = f"{type(exc).__name__}: {exc}"
        return dataclasses.replace(result, diagnostics=diagnostics)

    def infer(
        self,
        features,
        progress_callback: Optional[ProgressCallback],
        cancel_check: Optional[CancelCheck],
    ):
        request = self.request
        deps = self.deps
        run_multitrack = deps.run_multitrack_windowed_inference_onnx_stream
        if run_multitrack is None:  # pre-existing deps constructions (CLI/GUI/tests)
            from infer import run_multitrack_windowed_inference_onnx_stream as run_multitrack

        if request.model_path.suffix.lower() != ".onnx":
            raise UnsupportedPipelineError(
                "The 'frame_startend_heatmap' runtime supports ONNX models only (got "
                f"'{request.model_path.suffix or request.model_path.name}'); export the heatmap "
                "checkpoint to model.onnx with 3 outputs (pointness/start/end)."
            )
        _emit(progress_callback, "inference", 5)
        scaler = deps.load_scaler_asset(str(request.scaler_path))

        duration_s = request.estimated_duration_s
        if duration_s is None:
            duration_s = float(request.duration) if 0 < int(request.duration) < 999999 else 0.0
        estimated_feature_rows = max(0, int(round(float(duration_s) * float(request.fps))) - 1)
        estimated_windows = _estimate_stream_window_count(
            estimated_feature_rows, int(request.seq_len), int(request.overlap)
        )
        feature_rows_seen = 0

        def scaled_feature_rows():
            nonlocal feature_rows_seen
            for feature_vector, _target in features:
                _check(cancel_check)
                feature_rows_seen += 1
                if estimated_feature_rows > 0:
                    frac = min(1.0, feature_rows_seen / float(estimated_feature_rows))
                    stage_progress = int(1 + frac * 94)
                    _emit(progress_callback, "preprocess", stage_progress)
                    _emit(progress_callback, "feature", stage_progress)
                    if estimated_windows is None:
                        _emit(progress_callback, "inference", int(5 + frac * 80))
                row = deps.np.asarray(feature_vector, dtype=deps.np.float32).reshape(1, -1)
                yield scaler.transform(row)[0].astype(deps.np.float32)

        def infer_progress(frac: float) -> None:
            _emit(progress_callback, "inference", int(5 + max(0.0, min(1.0, frac)) * 90))

        tracks = run_multitrack(
            str(request.model_path),
            scaled_feature_rows(),
            sequence_length=int(request.seq_len),
            overlap=int(request.overlap),
            progress_callback=infer_progress,
            total_windows=estimated_windows,
        )
        _emit(progress_callback, "pose", 100, "completed")
        _emit(progress_callback, "preprocess", 100, "completed")
        _emit(progress_callback, "feature", 100, "completed")
        return tracks

    def _decode_config(self):
        config_cls = self.deps.heatmap_decode_config_cls
        if config_cls is None:  # pre-existing deps constructions (CLI/GUI/tests)
            from infer import HeatmapDecodeConfig as config_cls

        valid = set(config_cls.__dataclass_fields__)
        params = {k: v for k, v in (self.spec.params or {}).items() if k in valid and v is not None}
        # Per-run overrides (config.toml / CLI / API) win over the manifest.
        overrides = getattr(self.request, "decode_params", None) or {}
        unknown = sorted(k for k in overrides if k not in valid)
        if unknown:
            logging.warning("Ignoring unknown heatmap decode override(s): %s", ", ".join(unknown))
        params.update({k: v for k, v in overrides.items() if k in valid and v is not None})
        return config_cls(**params)

    def postprocess(self, model_output) -> List[Interval]:
        decode = self.deps.decode_heatmap_segments
        if decode is None:  # pre-existing deps constructions (CLI/GUI/tests)
            from infer import decode_heatmap_segments as decode

        np = self.deps.np
        pointness = np.asarray(model_output[0], dtype=np.float64)
        start_prob = np.asarray(model_output[1], dtype=np.float64)
        end_prob = np.asarray(model_output[2], dtype=np.float64)
        num_frames = int(pointness.shape[0])
        timestamps = (
            float(self.request.start_time)
            + np.arange(num_frames, dtype=np.float64) / float(self.request.fps)
        )
        return decode(pointness, start_prob, end_prob, timestamps, self._decode_config())


class StartEndAttentionVotingModel(AnalysisModel):
    pipeline_id = START_END_ATTENTION_VOTING

    def preprocess(self, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        raise UnsupportedPipelineError(
            "Pipeline 'start_end_attention_voting' is declared but no production runtime is implemented yet."
        )

    def infer(self, features, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        raise UnsupportedPipelineError(
            "Pipeline 'start_end_attention_voting' is declared but no production runtime is implemented yet."
        )

    def postprocess(self, model_output) -> List[FrameSegment]:
        return decode_start_end_votes(
            model_output["start_scores"],
            model_output["end_scores"],
            threshold=float(model_output.get("threshold", 0.5)),
            min_duration_frames=int(model_output.get("min_duration_frames", 1)),
        )


def decode_start_end_votes(
    start_scores,
    end_scores,
    *,
    threshold: float = 0.5,
    min_duration_frames: int = 1,
) -> List[FrameSegment]:
    """Pair start/end score peaks into intervals.

    This is intentionally small and deterministic for now; the real E2E model can
    replace the voting rule behind the same postprocess contract.
    """
    starts = [idx for idx, value in enumerate(start_scores) if float(value) >= threshold]
    ends = [idx for idx, value in enumerate(end_scores) if float(value) >= threshold]
    intervals: List[FrameSegment] = []
    end_cursor = 0
    for start in starts:
        while end_cursor < len(ends) and ends[end_cursor] <= start:
            end_cursor += 1
        if end_cursor >= len(ends):
            break
        end = ends[end_cursor]
        if end - start >= max(1, min_duration_frames):
            intervals.append((start, end))
        end_cursor += 1
    return intervals


def build_analysis_model(request: RunRequest, spec: PipelineSpec, deps: RuntimeDeps) -> AnalysisModel:
    if spec.pipeline_id == FRAME_PROBABILITY_HYSTERESIS:
        return FrameProbabilityHysteresisModel(request, spec, deps)
    if spec.pipeline_id == FRAME_STARTEND_HEATMAP:
        return HeatmapHybridModel(request, spec, deps)
    if spec.pipeline_id == START_END_ATTENTION_VOTING:
        return StartEndAttentionVotingModel(request, spec, deps)
    raise UnsupportedPipelineError(f"Unsupported pipeline_id '{spec.pipeline_id}'.")

