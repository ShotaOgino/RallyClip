"""Far-court crop pass: geometry-driven upscaled window, mapping back, far-slot guards."""

import math

import numpy as np
import pytest

from extraction.crop_pass import (
    ADAPTIVE_DEFAULTS,
    CROP_OUT_SIZE,
    CROP_WINDOW,
    adaptive_crop,
    crop_frame,
    crop_pixels,
    crop_to_frame,
)
from extraction.yolo_onnx_runner import Result
from helpers.module_stubs import import_pose_extractor_with_stubs
from preprocessing.netline_slots import CourtGeometry, StaticDetections, new_far_stats, pick_far
from rallyclip_engine.models import _far_crop_kwargs

REF = (1920, 1080)
# A rear-camera court like the customer footage: far baseline 350 px wide at y~400,
# net at y~470 (reference pixels).
FAR_L, FAR_R = (760.0, 400.0), (1110.0, 400.0)
NET_A, NET_B = (600.0, 470.0), (1270.0, 470.0)


# --- adaptive window -------------------------------------------------------------
def test_adaptive_window_margins_and_upscale():
    spec = adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF)
    x1, y1, x2, y2 = (round(v * s) for v, s in zip(spec.window, REF * 2))
    base_w, depth = 350.0, 70.0
    assert (x1, x2) == (math.floor(760 - 0.35 * base_w), math.ceil(1110 + 0.35 * base_w))
    # top: max(1.2 * depth = 84, 0.3 * base_w = 105) above the far baseline
    assert y1 == 400 - 105
    assert y2 == math.ceil(470 + 0.15 * depth)
    assert spec.source == "geometry"
    # a 1.6 m player at the far baseline is 1.6 * 350 / 10.97 = 51 px -> ~2.15x for 110 px
    assert 2.0 < spec.scale < 2.4
    w, h = x2 - x1, y2 - y1
    assert spec.imgsz == spec.out_size[0] and spec.imgsz % 32 == 0
    assert spec.out_size[0] == pytest.approx(w * spec.scale, abs=1)
    assert spec.out_size[1] == pytest.approx(h * spec.scale, abs=1)
    # the estimated far player ends up >= 90 px tall in the crop
    assert 1.6 * base_w / 10.97 * spec.scale >= 90


def test_adaptive_window_is_clamped_to_the_frame():
    # A far baseline hugging the top-left corner: margins would leave the frame.
    spec = adaptive_crop((40.0, 30.0), (400.0, 30.0), (0.0, 90.0), (500.0, 90.0), REF)
    x1, y1, x2, y2 = spec.window
    assert x1 == 0.0 and y1 == 0.0
    assert 0 < x2 <= 1 and 0 < y2 <= 1


def test_adaptive_window_scale_respects_bounds_and_cost_cap():
    # Huge far player (broadcast close-up) -> never downscaled below native.
    big = adaptive_crop((300.0, 300.0), (1600.0, 300.0), (100.0, 500.0), (1800.0, 500.0), REF)
    assert big.scale >= 1.0 - 1e-6
    # Tiny far court -> capped at max_scale.
    tiny = adaptive_crop((900.0, 400.0), (1000.0, 400.0), (880.0, 420.0), (1020.0, 420.0), REF)
    assert tiny.scale <= ADAPTIVE_DEFAULTS["max_scale"] + 1e-6
    capped = adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF, max_long_side=640)
    assert max(capped.out_size) == 640 == capped.imgsz


@pytest.mark.parametrize("args", [
    (None, FAR_R, NET_A, NET_B),
    (FAR_L, None, NET_A, NET_B),
    (FAR_R, FAR_L, NET_A, NET_B),             # corners swapped -> no width
    (FAR_L, FAR_R, (600.0, 390.0), (1270.0, 390.0)),  # net above the far baseline
])
def test_adaptive_window_degenerate_geometry_is_none(args):
    assert adaptive_crop(*args, REF) is None


def test_adaptive_window_rejects_unknown_params():
    with pytest.raises(ValueError):
        adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF, side_margn=0.2)


# --- mapping back at scale -------------------------------------------------------
@pytest.mark.parametrize("frame_size", [(1920, 1080), (3840, 2160), (1280, 720)])
def test_upscaled_crop_maps_back_to_frame_pixels(frame_size):
    fw, fh = frame_size
    spec = adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF)
    frame = np.zeros((fh, fw, 3), np.uint8)
    crop = crop_frame(frame, spec.window, spec.out_size)
    assert (crop.shape[1], crop.shape[0]) == spec.out_size
    x1, y1, x2, y2 = crop_pixels(fw, fh, spec.window)
    # A frame point -> crop pixels (forward model of crop+resize) -> back.
    px, py = x1 + 0.4 * (x2 - x1), y1 + 0.7 * (y2 - y1)
    cx, cy = (px - x1) * spec.out_size[0] / (x2 - x1), (py - y1) * spec.out_size[1] / (y2 - y1)
    boxes, kps = crop_to_frame(np.array([[cx, cy, cx, cy]], np.float32), np.array([[[cx, cy]]], np.float32),
                               fw, fh, spec.window, spec.out_size)
    np.testing.assert_allclose(boxes[0], [px, py, px, py], atol=1e-2)
    np.testing.assert_allclose(kps[0, 0], [px, py], atol=1e-2)


def test_upscaled_crop_content_lands_where_mapped():
    """A bright square in the frame shows up in the resized crop where crop_to_frame inverts it."""
    spec = adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[380:400, 900:910] = 255  # feet at the far baseline, near the center
    crop = crop_frame(frame, spec.window, spec.out_size)
    ys, xs = np.nonzero(crop[..., 0] > 127)
    box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)
    back, _ = crop_to_frame(box, np.zeros((1, 1, 2), np.float32), 1920, 1080, spec.window, spec.out_size)
    np.testing.assert_allclose(back[0], [900, 380, 910, 400], atol=1.0)


# --- fixed-window fallback is unchanged ------------------------------------------
def _geom(far_left=FAR_L, far_right=FAR_R):
    px_to_m = np.linalg.inv(np.array([[50.0, 0.0, 960.0], [0.0, -20.0, 600.0], [0.0, 0.0, 1.0]]))
    data = {
        "net_line_px": [list(NET_A), list(NET_B)],
        "homography_px_to_m": px_to_m.tolist(),
        "points_px": {"far_left": None if far_left is None else list(far_left),
                      "far_right": None if far_right is None else list(far_right)},
    }
    return CourtGeometry.from_dict(data)


def test_far_crop_kwargs_fall_back_to_the_fixed_window():
    fixed = {"crop_window": CROP_WINDOW}
    assert _far_crop_kwargs(_geom(far_left=None), {}, REF) == fixed
    assert _far_crop_kwargs(_geom(), {"far_crop": {"mode": "fixed"}}, REF) == fixed
    manifest_window = {"crop_window": [0.2, 0.0, 0.8, 0.5], "far_crop": {"mode": "fixed"}}
    assert _far_crop_kwargs(_geom(), manifest_window, REF) == {"crop_window": (0.2, 0.0, 0.8, 0.5)}


def test_far_crop_kwargs_adaptive_by_default_with_manifest_overrides():
    kw = _far_crop_kwargs(_geom(), {"far_crop": {"target_player_px": 90}}, REF)
    spec = adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF, target_player_px=90)
    assert kw == {"crop_window": spec.window, "crop_out_size": spec.out_size, "crop_imgsz": spec.imgsz}


class _RecordingModel:
    def __init__(self, crop_box=None):
        self.calls = []
        self.crop_box = crop_box

    def predict(self, source, imgsz, **_kw):
        self.calls.append(([s.shape for s in source], imgsz))
        out = []
        for s in source:
            if self.crop_box is not None and s.shape[:2] != (1080, 1920):
                out.append(Result(np.array([self.crop_box], np.float32), np.array([0.9], np.float32),
                                  np.zeros((1, 17, 2), np.float32), np.ones((1, 17), np.float32)))
            else:
                out.append(None)
        return out


def _run_extractor(monkeypatch, crop_box=None, **crop_kwargs):
    pose_module = import_pose_extractor_with_stubs(monkeypatch)
    monkeypatch.setattr(pose_module.av, "open", lambda _p: (_ for _ in ()).throw(OSError()))
    ex = pose_module.PoseExtractor.__new__(pose_module.PoseExtractor)
    ex.device, ex.batch_size, ex.imgsz = "cpu", 1, 960
    ex.model = _RecordingModel(crop_box)
    ex.frame_iterator_pyav = lambda _p: iter([(np.zeros((1080, 1920, 3), np.uint8), 0.0)])
    monkeypatch.setenv("RALLYCLIP_NO_TQDM", "1")
    frames = list(ex.iter_pose_frames("x.mp4", 0.25, duration_seconds=1, target_fps=5, imgsz=960, **crop_kwargs))
    return ex.model.calls, frames


def test_fixed_crop_pass_is_native_scale_at_the_full_frame_imgsz(monkeypatch):
    calls, _ = _run_extractor(monkeypatch, crop_window=CROP_WINDOW)
    assert calls == [([(1080, 1920, 3)], 960), ([(CROP_OUT_SIZE[1], CROP_OUT_SIZE[0], 3)], 960)]


def test_no_crop_window_runs_no_crop_pass(monkeypatch):
    calls, frames = _run_extractor(monkeypatch)
    assert calls == [([(1080, 1920, 3)], 960)] and "crop" not in frames[0]


def test_adaptive_crop_pass_feeds_upscaled_crop_and_maps_back(monkeypatch):
    spec = adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF)
    x1, y1, x2, y2 = crop_pixels(1920, 1080, spec.window)
    s = spec.out_size[0] / (x2 - x1)
    crop_box = [(900 - x1) * s, (340 - y1) * s, (920 - x1) * s, (400 - y1) * s]
    calls, frames = _run_extractor(monkeypatch, crop_box=crop_box, crop_window=spec.window,
                                   crop_out_size=spec.out_size, crop_imgsz=spec.imgsz)
    assert calls[1] == ([(spec.out_size[1], spec.out_size[0], 3)], spec.imgsz)
    np.testing.assert_allclose(frames[0]["crop"]["boxes"][0], [900, 340, 920, 400], atol=0.5)


# --- far-slot guards -------------------------------------------------------------
# Straight-on synthetic court: x = 960 + 50u, y = 600 - 20v (net at y = 600).
PX_TO_M = np.linalg.inv(np.array([[50.0, 0.0, 960.0], [0.0, -20.0, 600.0], [0.0, 0.0, 1.0]]))
GEOM = CourtGeometry(np.array([686.0, 600.0]), np.array([1234.0, 600.0]), PX_TO_M, "test")


def box_at(u, v, h=50.0):
    x, y = 960 + 50 * u, 600 - 20 * v
    return np.array([x - 12, y - h, x + 12, y], dtype=np.float32)


def test_pick_far_prefers_in_court_over_gate_only_candidates():
    # Ball kid deep behind the baseline at the center line vs the player 2 m off it.
    boxes = np.stack([box_at(0.2, 16.5), box_at(2.0, 11.0)])
    stats = new_far_stats()
    assert pick_far(boxes, None, GEOM, stats=stats) == 1
    assert stats == {"frames": 1, "picked": 1, "outside_court": 0, "static_picked": 0, "static_rejected": 0}


def test_pick_far_counts_frames_that_pick_outside_the_court():
    stats = new_far_stats()
    assert pick_far(np.stack([box_at(6.9, 8.0)]), None, GEOM, stats=stats) == 0  # wide, inside the gate
    assert stats["outside_court"] == 1 and stats["picked"] == 1


def test_pick_far_never_picks_a_static_bystander_off_court():
    bystander = box_at(-6.9, 6.0)  # beside the far court, inside the wide gate
    stats = new_far_stats()
    assert pick_far(np.stack([bystander]), None, GEOM, np.array([True]), stats) is None
    assert stats["static_rejected"] == 1 and stats["picked"] == 0
    # Not static (yet): still a fallback candidate.
    assert pick_far(np.stack([bystander]), None, GEOM, np.array([False])) == 0


def test_pick_far_prefers_a_moving_player_over_a_static_one_in_court():
    # Static figure on the center line (e.g. a coach on court) vs the moving player.
    boxes = np.stack([box_at(0.0, 13.0), box_at(3.0, 10.0)])
    stats = new_far_stats()
    assert pick_far(boxes, None, GEOM, np.array([True, False]), stats) == 1
    assert pick_far(boxes[:1], None, GEOM, np.array([True]), stats) == 0  # only one left: still picked
    assert stats["static_picked"] == 1


def test_static_detections_flag_a_stationary_box_but_not_a_mover():
    tracker = StaticDetections(min_frames=20, forget_frames=5)
    bystander = box_at(-6.9, 6.0)
    flags = []
    for t in range(40):
        player = box_at(-3.0 + 0.15 * t, 10.0 + 0.1 * (t % 7))
        flags.append(tracker.update(np.stack([bystander + np.float32(t % 2), player])))
    flags = np.array(flags)
    assert not flags[:19, 0].any() and flags[19:, 0].all()
    assert not flags[:, 1].any()


def test_static_detections_forget_after_a_gap():
    tracker = StaticDetections(min_frames=5, forget_frames=3)
    b = np.stack([box_at(0.0, 10.0)])
    for _ in range(6):
        last = tracker.update(b)
    assert last[0]
    for _ in range(5):
        tracker.update(np.empty((0, 4), np.float32))
    assert not tracker.update(b)[0]


def test_slots_far_pick_stats_on_the_preprocessor():
    from preprocessing.data_preprocessor import DataPreprocessor

    def dets(boxes):
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        n = len(boxes)
        return {"boxes": boxes, "box_conf": np.full(n, 0.9, np.float32),
                "keypoints": np.zeros((n, 17, 2), np.float32), "conf": np.ones((n, 17), np.float32)}

    frame = dets([box_at(0, -9, h=120)])
    frame["crop"] = dets([box_at(6.9, 8.0)])
    empty = dets([box_at(0, -9, h=120)])
    pre = DataPreprocessor(screen_width=1920, screen_height=1080)
    out = list(pre.iter_slot_players([frame, empty], np.zeros((1080, 1920), np.uint8), 1920, 1080, GEOM))
    assert out[0][1]["far"] is not None and out[1][1]["far"] is None
    assert pre.far_pick_stats["frames"] == 2 and pre.far_pick_stats["outside_court"] == 1


def test_static_canvas_shrinks_an_upscaled_crop_once(monkeypatch):
    """CoreML static export ignores imgsz: the crop is resized straight to fit its canvas."""
    spec = adaptive_crop(FAR_L, FAR_R, NET_A, NET_B, REF)
    pose_module = import_pose_extractor_with_stubs(monkeypatch)
    monkeypatch.setattr(pose_module.av, "open", lambda _p: (_ for _ in ()).throw(OSError()))
    ex = pose_module.PoseExtractor.__new__(pose_module.PoseExtractor)
    ex.device, ex.batch_size, ex.imgsz = "coreml", 1, 960
    ex.model = _RecordingModel()
    ex.model.static_input_hw = (544, 960)
    ex.frame_iterator_pyav = lambda _p: iter([(np.zeros((1080, 1920, 3), np.uint8), 0.0)])
    monkeypatch.setenv("RALLYCLIP_NO_TQDM", "1")
    list(ex.iter_pose_frames("x.mp4", 0.25, duration_seconds=1, target_fps=5, imgsz=960, crop_window=spec.window,
                             crop_out_size=spec.out_size, crop_imgsz=spec.imgsz))
    (h, w, _), = ex.model.calls[1][0]
    assert w == 960 and h == round(spec.out_size[1] * 960 / spec.out_size[0])
    # The fixed crop already fits the canvas: unchanged.
    ex.model.calls.clear()
    ex.frame_iterator_pyav = lambda _p: iter([(np.zeros((1080, 1920, 3), np.uint8), 0.0)])
    list(ex.iter_pose_frames("x.mp4", 0.25, duration_seconds=1, target_fps=5, imgsz=960, crop_window=CROP_WINDOW))
    assert ex.model.calls[1] == ([(540, 960, 3)], 960)
