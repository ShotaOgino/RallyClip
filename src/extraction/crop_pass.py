"""Second pose pass on a far-court crop.

A 1920-wide frame letterboxes to imgsz=960 at scale 0.5, which leaves the far
player at ~80px, near YOLO's floor for pose. Running the same model on the
top-centre 960x540 window of the 1920x1080 frame feeds them in at native scale.

Shared by scripts/extract_crop_poses.py (training side-car) and the runtime pose
extractor. The window is fractional so it applies to any source resolution;
on 1920x1080 it is exactly the training crop (480, 0, 1440, 540).

Adaptive window (runtime, when the court model found the far baseline): on a
high rear camera the far player can be only ~40-55px tall even at native scale,
where YOLO misses most of them. `adaptive_crop` derives a tight window around the
far half-court from the court geometry and upscales it so the far player reaches
~`target_player_px` -- at about the same pixel count (cost) as the fixed crop.
The fixed window stays the fallback whenever the far corners are unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

CROP_WINDOW = (0.25, 0.0, 0.75, 0.5)  # x1, y1, x2, y2 as fractions of the frame
CROP_OUT_SIZE = (960, 540)            # crop resized to this (w, h): half of 1920x1080
MERGE_IOU = 0.6                       # matches PlayerAssigner.merge_iou_thresh

# Adaptive (geometry-driven) window defaults; overridable per model via
# manifest feature_pipeline.far_crop (see adaptive_crop for the meaning of each).
DOUBLES_WIDTH_M = 10.97
STRIDE = 32
ADAPTIVE_DEFAULTS = {
    "side_margin": 0.35,         # x margin each side, fraction of the far-baseline width
    "top_ratio": 1.2,            # window top: this x the far-baseline-to-net distance above the baseline ...
    "top_min_width_ratio": 0.3,  # ... but at least this x the far-baseline width (players run back)
    "bottom_ratio": 0.15,        # below the net line, x the far-baseline-to-net distance (feet at the net)
    "target_player_px": 110.0,   # upscale until a standing far player is about this tall
    "player_height_m": 1.6,      # assumed far-player height for that estimate (juniors, ready stance)
    "min_scale": 1.0,
    "max_scale": 3.0,
    "max_long_side": 1920,       # cost cap on the upscaled crop's long side (px)
}


@dataclass(frozen=True)
class CropSpec:
    """Where to crop, what size to feed YOLO, and at which imgsz.

    window: fractional (x1, y1, x2, y2) of the frame. out_size: (w, h) the crop is
    resized to, in reference-scale pixels. imgsz: predict size for the crop pass,
    None = the caller's full-frame imgsz (the fixed-window contract).
    """

    window: tuple[float, float, float, float] = CROP_WINDOW
    out_size: tuple[int, int] = CROP_OUT_SIZE
    imgsz: Optional[int] = None
    scale: float = 1.0
    source: str = "fixed"


FIXED_CROP = CropSpec()


def _ceil_stride(v: float) -> int:
    return int(math.ceil(v / STRIDE - 1e-9) * STRIDE)


def adaptive_crop(
    far_left: Optional[Sequence[float]],
    far_right: Optional[Sequence[float]],
    net_a: Optional[Sequence[float]],
    net_b: Optional[Sequence[float]],
    ref_size: tuple[int, int],
    **params,
) -> Optional[CropSpec]:
    """Far-court window + upscale from the court geometry, or None (use the fixed crop).

    All inputs are reference pixels (ref_size = (w, h), the geometry's frame).
    x: the far-baseline corners widened by side_margin x baseline width each side.
    y: from max(top_ratio x baseline-to-net distance, top_min_width_ratio x
    baseline width) above the far baseline down to bottom_ratio x that distance
    below the net line. Clamped to the frame. Scale: a player_height_m player at
    the far baseline (baseline width = 10.97 m doubles) upscaled to
    target_player_px, clamped to [min_scale, max_scale] and the long-side cap;
    the long side is rounded up to a multiple of 32 so YOLO's letterbox at
    imgsz = long side is a pure pad (no second resize).
    """
    p = {**ADAPTIVE_DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    unknown = set(p) - set(ADAPTIVE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown far_crop parameter(s): {sorted(unknown)}")
    if far_left is None or far_right is None or net_a is None or net_b is None:
        return None
    ref_w, ref_h = int(ref_size[0]), int(ref_size[1])
    (lx, ly), (rx, ry) = (float(far_left[0]), float(far_left[1])), (float(far_right[0]), float(far_right[1]))
    base_w = rx - lx
    far_y = (ly + ry) / 2
    net_y = max(float(net_a[1]), float(net_b[1]))
    depth = net_y - far_y
    if not all(map(math.isfinite, (lx, ly, rx, ry, net_y))) or base_w <= 1 or depth <= 1:
        return None
    x1 = max(0.0, lx - p["side_margin"] * base_w)
    x2 = min(float(ref_w), rx + p["side_margin"] * base_w)
    top = max(p["top_ratio"] * depth, p["top_min_width_ratio"] * base_w)
    y1 = max(0.0, min(ly, ry) - top)
    y2 = min(float(ref_h), net_y + p["bottom_ratio"] * depth)
    x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    w, h = x2 - x1, y2 - y1
    if w < STRIDE or h < STRIDE:
        return None
    player_px = p["player_height_m"] * base_w / DOUBLES_WIDTH_M
    scale = min(max(p["target_player_px"] / player_px, p["min_scale"]), p["max_scale"])
    target = max(w, h) * scale
    long_side = _ceil_stride(target)
    if long_side > max(w, h) * p["max_scale"]:  # rounding up would overshoot the cap
        long_side = max(STRIDE, int(target // STRIDE) * STRIDE)
    long_side = min(long_side, max(STRIDE, int(p["max_long_side"]) // STRIDE * STRIDE))
    scale = long_side / max(w, h)
    out_size = (long_side, max(1, int(round(h * scale)))) if w >= h else (max(1, int(round(w * scale))), long_side)
    return CropSpec(
        window=(x1 / ref_w, y1 / ref_h, x2 / ref_w, y2 / ref_h),
        out_size=out_size,
        imgsz=long_side,
        scale=float(scale),
        source="geometry",
    )


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_detections(boxes, box_conf, kps, kp_conf, thresh=MERGE_IOU):
    """Union detections that overlap above `thresh` (single-linkage).

    A player clipped by the crop boundary routinely yields several partial boxes
    that survive YOLO's own NMS; without this they consume both crop slots and
    evict the far player. The union box is kept, along with the keypoints of the
    highest-confidence member of the cluster.
    """
    n = len(boxes)
    if n <= 1:
        return boxes, box_conf, kps, kp_conf
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(boxes[i], boxes[j]) > thresh:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    ob, oc, ok, okc = [], [], [], []
    for members in groups.values():
        bs = boxes[members]
        best = members[int(np.argmax(box_conf[members]))]
        ob.append([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()])
        oc.append(float(box_conf[best]))
        ok.append(kps[best])
        okc.append(kp_conf[best])
    order = np.argsort(-np.asarray(oc))
    return (np.asarray(ob, np.float32)[order], np.asarray(oc, np.float32)[order],
            np.asarray(ok, np.float32)[order], np.asarray(okc, np.float32)[order])


def crop_pixels(width: int, height: int, window: Sequence[float] = CROP_WINDOW) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = window
    return (int(round(x1 * width)), int(round(y1 * height)), int(round(x2 * width)), int(round(y2 * height)))


def crop_frame(frame_bgr: np.ndarray, window: Sequence[float] = CROP_WINDOW,
               out_size: tuple[int, int] = CROP_OUT_SIZE) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = crop_pixels(w, h, window)
    crop = frame_bgr[y1:y2, x1:x2]
    if (crop.shape[1], crop.shape[0]) == tuple(out_size):
        return np.ascontiguousarray(crop)
    interp = cv2.INTER_AREA if crop.shape[1] > out_size[0] else cv2.INTER_LINEAR
    return cv2.resize(crop, out_size, interpolation=interp)


def crop_to_frame(boxes: np.ndarray, kps: np.ndarray, width: int, height: int,
                  window: Sequence[float] = CROP_WINDOW,
                  out_size: tuple[int, int] = CROP_OUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Map crop-pixel boxes [N, 4] and keypoints [N, K, 2] back to frame pixels."""
    x1, y1, x2, y2 = crop_pixels(width, height, window)
    sx, sy = (x2 - x1) / out_size[0], (y2 - y1) / out_size[1]
    boxes = boxes * np.array([sx, sy, sx, sy], np.float32) + np.array([x1, y1, x1, y1], np.float32)
    kps = kps * np.array([sx, sy], np.float32) + np.array([x1, y1], np.float32)
    return boxes.astype(np.float32), kps.astype(np.float32)
