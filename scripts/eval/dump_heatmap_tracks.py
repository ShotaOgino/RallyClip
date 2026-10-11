"""Dump HeatmapHybridModel per-frame tracks (pointness/start/end) to .npz so decode
params can be swept offline without re-running pose extraction, plus the per-frame
near/far player records (reference pixels = the manifest's screen size: 1280x720 for
v1 artifacts, 1920x1080 for v0.6.0) and the raw pre-court-filter detections, so the
Phase 1-2 analysis (src/analysis/) can run offline on the same dump.
Run from repo root: PYTHONPATH=src $PY scripts/eval/dump_heatmap_tracks.py --video V --out X.npz
"""

import argparse
import time
from pathlib import Path
import numpy as np

from cli.main import build_run_config
from rallyclip_core.contracts import RunRequest
from rallyclip_core.pipelines import resolve_pipeline_spec
from rallyclip_engine.models import build_analysis_model
from rallyclip_engine.runtime import load_runtime_deps

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--yolo-device", default=None)
ap.add_argument("--start-time", type=int, default=None)
ap.add_argument("--duration", type=int, default=None)
a = ap.parse_args()

ns = argparse.Namespace(
    video=a.video,
    yolo_device=a.yolo_device,
    write_csv=False,
    segment_video=False,
    output_dir=str(Path(a.out).parent),
    config=None,
    start_time=a.start_time,
    duration=a.duration,
)
cfg = build_run_config(ns)
req = RunRequest(
    video_path=cfg.video_path,
    output_dir=cfg.output_dir,
    output_name=cfg.output_name,
    csv_output_dir=cfg.csv_output_dir,
    write_csv=False,
    segment_video=False,
    yolo_weights=cfg.yolo_weights,
    yolo_device=cfg.yolo_device,
    model_path=cfg.model_path,
    scaler_path=cfg.scaler_path,
    manifest_path=cfg.manifest_path,
    pipeline_id=cfg.pipeline_id,
    fps=cfg.fps,
    seq_len=cfg.seq_len,
    imgsz=cfg.imgsz,
    conf=cfg.conf,
    feature_set=cfg.feature_set,
    screen_width=cfg.screen_width,
    screen_height=cfg.screen_height,
    overlap=cfg.overlap,
    sigma=cfg.sigma,
    low=cfg.low,
    high=cfg.high,
    min_dur_sec=cfg.min_dur_sec,
    start_time=cfg.start_time,
    duration=cfg.duration,
    models_dir=Path.cwd() / "models",
)
if cfg.yolo_device:
    import os

    os.environ["POSE_DEVICE"] = cfg.yolo_device
deps = load_runtime_deps()

# Tee the per-frame near/far player records (reference pixels, kept frames only, 1:1
# with the feature rows / model tracks). The feature stream itself is untouched.
#   v2 feature sets (v0.6.0): wrap DataPreprocessor.iter_slot_players -> (status, {near, far});
#     with net-line slots the far player comes from the far-court crop pass.
#   v1: wrap FeatureEngineer.iter_build_features -> (frame, target, near, far) records.
# Raw detections: the pose frame each preprocessor is currently on (one pose frame per
# yielded record, no lookahead), full-frame and BEFORE the near-court filter (which drops
# the far player on v1), native source pixels; scaled to the reference in _pack_raw.
_players = {"near": [], "far": [], "raw": []}
_src = {"w": None, "h": None, "last_raw": None}
_DP = deps.DataPreprocessor


def _tee_pose(pose_data):
    for frame_data in pose_data:
        _src["last_raw"] = frame_data
        yield frame_data


def _keep(near, far):
    _players["near"].append(near)
    _players["far"].append(far)
    _players["raw"].append(_src["last_raw"])


_orig_slots = _DP.iter_slot_players


def _tee_slots(self, pose_data, court_mask, src_width, src_height, geometry=None):
    _src["w"], _src["h"] = int(src_width), int(src_height)
    for status, players in _orig_slots(
        self, _tee_pose(pose_data), court_mask, src_width, src_height, geometry
    ):
        if status >= 0:
            players = players or {}
            _keep(players.get("near"), players.get("far"))
        yield status, players


_DP.iter_slot_players = _tee_slots

_orig_pre = _DP.iter_preprocess_frames


def _tee_pre(self, pose_data, court_mask, src_width, src_height):
    _src["w"], _src["h"] = int(src_width), int(src_height)
    return _orig_pre(self, _tee_pose(pose_data), court_mask, src_width, src_height)


_DP.iter_preprocess_frames = _tee_pre
_FE = deps.FeatureEngineer
_orig_iter = _FE.iter_build_features


def _tee_iter(self, records):
    def gen():
        for rec in records:
            frame_data, target, near, far = rec
            if target >= 0:
                _keep(near, far)
            yield rec

    return _orig_iter(self, gen())


_FE.iter_build_features = _tee_iter

RAW_MAX = 8


def _pack_raw(raws, ref_w, ref_h):
    n = len(raws)
    boxes = np.full((n, RAW_MAX, 4), np.nan, dtype=np.float32)
    kp = np.full((n, RAW_MAX, 17, 2), np.nan, dtype=np.float32)
    kpc = np.full((n, RAW_MAX, 17), np.nan, dtype=np.float32)
    bconf = np.full((n, RAW_MAX), np.nan, dtype=np.float32)
    count = np.zeros(n, dtype=np.int16)
    sx = ref_w / float(_src["w"] or ref_w)
    sy = ref_h / float(_src["h"] or ref_h)
    for i, fd in enumerate(raws):
        if not fd:
            continue
        b = np.asarray(fd.get("boxes", np.empty((0, 4))), dtype=np.float32).reshape(
            -1, 4
        )
        k = np.asarray(
            fd.get("keypoints", np.empty((0, 17, 2))), dtype=np.float32
        ).reshape(-1, 17, 2)
        c = np.asarray(fd.get("conf", np.empty((0, 17))), dtype=np.float32).reshape(
            -1, 17
        )
        bc = np.asarray(
            fd.get("box_conf", np.full((len(b),), -1.0)), dtype=np.float32
        ).reshape(-1)
        order = np.argsort(-bc) if len(bc) == len(b) else np.arange(len(b))
        order = order[:RAW_MAX]
        m = len(order)
        count[i] = m
        if m:
            boxes[i, :m] = b[order] * np.array([sx, sy, sx, sy], dtype=np.float32)
            kp[i, :m] = k[order] * np.array([sx, sy], dtype=np.float32)
            kpc[i, :m] = c[order]
            if len(bc) == len(b):
                bconf[i, :m] = bc[order]
    return boxes, kp, kpc, bconf, count


def _pack(players):
    n = len(players)
    box = np.full((n, 4), np.nan, dtype=np.float32)
    kp = np.full((n, 17, 2), np.nan, dtype=np.float32)
    kpc = np.full((n, 17), np.nan, dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    for i, p in enumerate(players):
        if p is None:
            continue
        present[i] = True
        box[i] = np.asarray(p["box"], dtype=np.float32)[:4]
        kp[i] = np.asarray(p["keypoints"], dtype=np.float32).reshape(17, 2)
        c = p.get("conf")
        if c is not None:
            kpc[i] = np.asarray(c, dtype=np.float32).reshape(17)
    return box, kp, kpc, present


spec = resolve_pipeline_spec(
    req.model_path, req.manifest_path, override_pipeline_id=req.pipeline_id
)
print(
    "pipeline:",
    spec.pipeline_id,
    "feature_set:",
    req.feature_set,
    "params:",
    spec.params,
    flush=True,
)
model = build_analysis_model(req, spec, deps)
last = [0.0]


def prog(ev):
    if ev.stage == "pose" and time.time() - last[0] > 30:
        last[0] = time.time()
        print(f"  pose {ev.progress}%", flush=True)


t0 = time.time()
feats = model.preprocess(prog, None)
tracks = model.infer(feats, prog, None)
pointness, start_prob, end_prob = (np.asarray(t, dtype=np.float64) for t in tracks[:3])
n = pointness.shape[0]
timestamps = float(req.start_time) + np.arange(n, dtype=np.float64) / float(req.fps)
default_segments = np.asarray(model.postprocess(tracks), dtype=np.float64)
nb, nk, nkc, npres = _pack(_players["near"])
fb, fk, fkc, fpres = _pack(_players["far"])
assert len(npres) == n, f"player records {len(npres)} != track frames {n}"
rb, rk, rkc, rbc, rcount = _pack_raw(
    _players["raw"], req.screen_width, req.screen_height
)
np.savez(
    a.out,
    pointness=pointness,
    start_prob=start_prob,
    end_prob=end_prob,
    timestamps=timestamps,
    default_segments=default_segments,
    fps=float(req.fps),
    params=str(spec.params),
    near_box=nb,
    near_kp=nk,
    near_kpconf=nkc,
    near_present=npres,
    far_box=fb,
    far_kp=fk,
    far_kpconf=fkc,
    far_present=fpres,
    ref_size=np.array([req.screen_width, req.screen_height], dtype=np.int32),
    raw_boxes=rb,
    raw_kp=rk,
    raw_kpconf=rkc,
    raw_box_conf=rbc,
    raw_count=rcount,
    src_size=np.array([_src["w"] or 0, _src["h"] or 0], dtype=np.int32),
)
print(
    f"players: near present {int(npres.sum())}/{n}, far present {int(fpres.sum())}/{n}, raw detections/frame mean {rcount.mean():.2f}, frames with >=2 raw {(rcount >= 2).sum()}",
    flush=True,
)
print(
    f"saved {a.out}: {n} frames, {len(default_segments)} default segments, {time.time() - t0:.0f}s",
    flush=True,
)
