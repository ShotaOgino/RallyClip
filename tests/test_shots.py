"""Phase 2 shot events + rule-based stroke type (src/analysis/shots.py).

Synthetic 5 fps COCO-17 skeletons in reference 1280x720 pixels: a near player seen from
behind (right shoulder at larger x) and a smaller far player facing the camera. Swings
are scripted wrist paths over 3-4 samples, the same coarseness the real dumps have.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis.shots import (
    PoseWindow,
    ShotParams,
    SwingEvent,
    analyze_point_shots,
    arm_speeds,
    classify_stroke,
    detect_swings,
    estimate_handedness,
)

ROOT = Path(__file__).resolve().parents[1]
FPS = 5.0
GOLDEN_TRACKS = ROOT / "tests" / "fixtures" / "golden_cli" / "clip.tracks.npz"

# wrist offsets (x, y) in units of 1/40 shoulder width, relative to the rest pose; x is
# multiplied by `facing` (+1 back to camera, -1 facing it) so it means "toward the
# player's right".
FOREHAND = {-1: (55, -10), 0: (40, -20), 1: (-70, -40), 2: (-30, -10)}
BACKHAND = {-1: (-75, -10), 0: (-60, -20), 1: (50, -45), 2: (20, -10)}
PUNCH = {0: (50, -10), 1: (-25, -30), 2: (-12, -15)}


def _skeleton(cx, foot_y, h, sw, facing):
    top = foot_y - h
    sh_y = top + 0.22 * h
    k = np.zeros((17, 2))
    r = facing  # +1: person's right at larger image x
    k[0] = (cx, top + 0.1 * h)
    k[1], k[2] = (cx - 0.1 * sw * r, top + 0.08 * h), (cx + 0.1 * sw * r, top + 0.08 * h)
    k[3], k[4] = (cx - 0.2 * sw * r, top + 0.1 * h), (cx + 0.2 * sw * r, top + 0.1 * h)
    k[5], k[6] = (cx - 0.5 * sw * r, sh_y), (cx + 0.5 * sw * r, sh_y)
    k[7], k[8] = (cx - 0.6 * sw * r, sh_y + 0.15 * h), (cx + 0.6 * sw * r, sh_y + 0.15 * h)
    k[9], k[10] = (cx - 0.55 * sw * r, sh_y + 0.3 * h), (cx + 0.55 * sw * r, sh_y + 0.3 * h)
    k[11], k[12] = (cx - 0.33 * sw * r, top + 0.5 * h), (cx + 0.33 * sw * r, top + 0.5 * h)
    k[13], k[14] = (cx - 0.3 * sw * r, top + 0.72 * h), (cx + 0.3 * sw * r, top + 0.72 * h)
    k[15], k[16] = (cx - 0.3 * sw * r, top + 0.95 * h), (cx + 0.3 * sw * r, top + 0.95 * h)
    box = np.array([cx - 0.35 * h, top, cx + 0.35 * h, foot_y])
    return k, box


def make_side(n, *, cx=640.0, foot_y=480.0, h=130.0, sw=40.0, facing=1, foot_path=None, seed=0):
    """One player's arrays at 5 fps: kp (n,17,2), kpconf, present, box."""
    rng = np.random.default_rng(seed)
    kp = np.zeros((n, 17, 2))
    box = np.zeros((n, 4))
    for i in range(n):
        fy = foot_path[i] if foot_path is not None else foot_y
        kp[i], box[i] = _skeleton(cx, fy, h, sw, facing)
    kp += rng.normal(0, 0.3, kp.shape)
    return {"kp": kp, "kpconf": np.full((n, 17), 0.9), "present": np.ones(n, bool), "box": box,
            "sw": sw, "facing": facing}


def add_swing(side, k, pattern, arm="right", both=False):
    """Overlay a scripted wrist path centred on sample k. Patterns are written for a
    right-hander; a left arm mirrors x (a lefty's forehand is on the player's left)."""
    s = side["sw"] / 40.0
    mirror = -1.0 if arm == "left" else 1.0
    arms = ("left", "right") if both else (arm,)
    for a in arms:
        j = 10 if a == "right" else 9
        for d, (dx, dy) in pattern.items():
            if 0 <= k + d < len(side["kp"]):
                side["kp"][k + d, j] += (dx * s * side["facing"] * mirror, dy * s)
    return side


def add_serve(side, k, arm="right"):
    """Toss arm up at k-2, hitting wrist behind the head at k-1, contact above the head at
    k, fast pull-down at k+1."""
    kp, box = side["kp"], side["box"]
    top = box[k, 1]
    f = side["facing"]
    hit, toss = (10, 9) if arm == "right" else (9, 10)
    cx = (box[k, 0] + box[k, 2]) / 2
    sw = side["sw"]
    kp[k - 2, toss] = (cx - 0.5 * sw * f, top - 25 * sw / 40)
    kp[k - 1, hit] = (cx + 0.6 * sw * f, top - 5 * sw / 40)
    kp[k, hit] = (cx + 0.4 * sw * f, top - 30 * sw / 40)
    kp[k + 1, hit] = (cx - 0.75 * sw * f, top + (0.22 * 130 + 50) * sw / 40)
    return side


def ts(n, t0=0.0):
    return t0 + np.arange(n) / FPS


def window(side, t, name="near"):
    return PoseWindow(kp=side["kp"], kpconf=side["kpconf"], present=side["present"], timestamps=t,
                      box=side["box"], side=name)


def detect(side, t, name="near", **kw):
    return detect_swings(side["kp"], side["kpconf"], side["present"], t, side=name, box=side["box"],
                         handedness=kw.pop("handedness", "right"), **kw)


def tracks_from(near, far, t):
    d = {"timestamps": t}
    for name, s in (("near", near), ("far", far)):
        d[f"{name}_kp"], d[f"{name}_kpconf"] = s["kp"], s["kpconf"]
        d[f"{name}_present"], d[f"{name}_box"] = s["present"], s["box"]
    return d


def absent(n):
    return {"kp": np.full((n, 17, 2), np.nan), "kpconf": np.full((n, 17), np.nan), "present": np.zeros(n, bool),
            "box": np.full((n, 4), np.nan), "sw": 18.0, "facing": -1}


# ------------------------------------------------------------------ detection


def test_forehand_arc_gives_one_event_near_contact():
    n = 30
    t = ts(n)
    side = add_swing(make_side(n), 12, FOREHAND)
    ev = detect(side, t)
    assert len(ev) == 1
    e = ev[0]
    assert e.source == "motion" and e.player_side == "near" and e.arm == "right"
    # biggest step is contact (k=12) -> follow-through (k=13); sub-sample peak in between
    assert t[12] <= e.t <= t[13]
    assert e.peak_speed > ShotParams().abs_floor
    assert 0.0 < e.confidence <= 1.0


def test_quiet_player_has_no_events():
    n = 40
    assert detect(make_side(n), ts(n)) == []


def test_speed_is_normalised_by_body_scale():
    """The same swing drawn at half size (far-player scale) has the same peak speed."""
    n = 30
    t = ts(n)
    big = detect(add_swing(make_side(n, sw=40, h=130), 12, FOREHAND), t)
    small = detect(add_swing(make_side(n, sw=20, h=65, cx=600, foot_y=300), 12, FOREHAND), t)
    assert len(big) == len(small) == 1
    assert small[0].peak_speed == pytest.approx(big[0].peak_speed, rel=0.15)


def test_two_swings_closer_than_refractory_collapse_to_one():
    n = 30
    t = ts(n)
    side = add_swing(add_swing(make_side(n), 10, FOREHAND), 13, FOREHAND)
    ev = detect(side, t)
    assert len(ev) == 1


def test_swings_further_apart_than_refractory_are_both_kept():
    n = 40
    t = ts(n)
    side = add_swing(add_swing(make_side(n), 10, FOREHAND), 18, FOREHAND)
    ev = detect(side, t)
    assert len(ev) == 2
    assert ev[1].t - ev[0].t >= ShotParams().refractory_sec


def test_nan_gap_and_identity_jump_do_not_create_events():
    n = 50
    t = ts(n)
    side = add_swing(make_side(n), 10, FOREHAND)
    # player missing for 1.2 s, then the detector locks onto someone 400 px away
    side["present"][20:26] = False
    side["kp"][20:26] = np.nan
    side["box"][20:26] = np.nan
    other = make_side(n, cx=1040.0, seed=1)
    for key in ("kp", "box"):
        side[key][26:] = other[key][26:]
    side["kp"][30, 10] = (np.nan, np.nan)  # one missing wrist keypoint
    side["kpconf"][31, 9] = 0.05           # one low-confidence wrist
    ev = detect(side, t)
    assert len(ev) == 1 and t[10] <= ev[0].t <= t[11]


def test_single_frame_left_right_swap_is_repaired():
    n = 30
    t = ts(n)
    side = make_side(n)
    side["kp"][15, [9, 10]] = side["kp"][15, [10, 9]]  # YOLO swaps the wrist labels once
    sp = arm_speeds(side["kp"], side["kpconf"], side["present"], t, side["box"])
    assert sp["swapped"][14] and sp["swapped"][15]
    assert np.nanmax(sp["right"]) < 2.0
    assert detect(side, t) == []


def test_handedness_from_fastest_wrist():
    n = 80
    t = ts(n)
    side = make_side(n)
    for k in (10, 25, 40, 55, 70):
        add_swing(side, k, FOREHAND, arm="left")
    arm, ratio = estimate_handedness(side["kp"], side["kpconf"], side["present"], t, side["box"])
    assert arm == "left" and ratio > 1.5


# ------------------------------------------------------------- classification


def _classify(side, t, k_event=None, name="near", **kw):
    ev = detect(side, t, name=name)
    assert len(ev) == 1, ev
    return ev[0], classify_stroke(ev[0], window(side, t, name), handedness="right", **kw)


def test_classify_forehand_and_backhand_near():
    n = 30
    t = ts(n)
    _, fh = _classify(add_swing(make_side(n), 12, FOREHAND), t)
    _, bh = _classify(add_swing(make_side(n), 12, BACKHAND, both=True), t)
    assert fh["stroke"] == "forehand" and fh["confidence"] > 0.3
    assert bh["stroke"] == "backhand" and bh["confidence"] > 0.3


def test_classify_forehand_far_player_facing_camera():
    n = 30
    t = ts(n)
    far = add_swing(make_side(n, cx=640, foot_y=250, h=60, sw=18, facing=-1), 12, FOREHAND)
    _, fh = _classify(far, t, name="far")
    assert fh["stroke"] == "forehand"


def test_classify_left_hander_flips_the_side_rule():
    n = 30
    t = ts(n)
    side = add_swing(make_side(n), 12, FOREHAND, arm="left")
    ev = detect(side, t, handedness="left")
    assert len(ev) == 1
    assert ev[0].arm == "left"
    assert classify_stroke(ev[0], window(side, t), handedness="left")["stroke"] == "forehand"
    bh = add_swing(make_side(n), 12, BACKHAND, arm="left")
    ev = detect(bh, t, handedness="left")
    assert classify_stroke(ev[0], window(bh, t), handedness="left")["stroke"] == "backhand"


def test_classify_serve_vs_overhead():
    n = 30
    t = ts(n)
    side = add_serve(make_side(n), 8)
    e, serve = _classify(side, t, point_start=0.2, is_first_of_side=True)
    assert serve["stroke"] == "serve" and serve["confidence"] >= 0.6
    # same posture, but not the player's first swing -> overhead
    oh = classify_stroke(e, window(side, t), handedness="right", point_start=0.2, is_first_of_side=False)
    assert oh["stroke"] == "overhead"
    # first swing but 4 s after the point start -> overhead
    late = classify_stroke(e, window(side, t), handedness="right", point_start=e.t - 4.0, is_first_of_side=True)
    assert late["stroke"] == "overhead"
    # the other side is the known server -> not a serve
    other = classify_stroke(e, window(side, t), handedness="right", point_start=0.2, is_first_of_side=True,
                            server="far")
    assert other["stroke"] == "overhead"


def test_known_server_first_swing_is_a_serve_without_posture():
    n = 30
    t = ts(n)
    side = add_swing(make_side(n), 8, FOREHAND)
    _, st = _classify(side, t, point_start=0.5, is_first_of_side=True, server="near")
    assert st == {"stroke": "serve", "confidence": 0.5}


def test_classify_volley_inside_court_short_swing():
    n = 40
    t = ts(n)
    feet = np.concatenate([np.full(15, 480.0), np.linspace(480, 380, 6), np.full(n - 21, 380.0)])
    side = add_swing(make_side(n, foot_path=feet), 25, PUNCH)
    e, st = _classify(side, t, baseline_foot_y=480.0)
    assert e.width_sec <= ShotParams().volley_short_sec + 1e-9
    assert st["stroke"] == "volley"
    # same punch from the baseline is a groundstroke
    _, st2 = _classify(add_swing(make_side(n), 25, PUNCH), t)
    assert st2["stroke"] == "forehand"


def test_far_player_is_never_volley():
    n = 40
    t = ts(n)
    feet = np.concatenate([np.full(15, 300.0), np.linspace(300, 340, 6), np.full(n - 21, 340.0)])
    far = add_swing(make_side(n, h=60, sw=18, facing=-1, foot_path=feet), 25, PUNCH)
    _, st = _classify(far, t, name="far", baseline_foot_y=300.0)
    assert st["stroke"] != "volley"


def test_prior_event_and_missing_pose_are_unknown():
    n = 20
    t = ts(n)
    side = make_side(n)
    prior = SwingEvent(t=1.0, player_side="near", peak_speed=0.0, arm="right", confidence=0.2, source="prior")
    assert classify_stroke(prior, window(side, t))["stroke"] == "unknown"
    side["present"][:] = False
    ev = SwingEvent(t=1.0, player_side="near", peak_speed=9.0, arm="right", confidence=0.5)
    assert classify_stroke(ev, window(side, t)) == {"stroke": "unknown", "confidence": 0.0}


# --------------------------------------------------------------- per point


def test_alternation_prior_fills_missing_far_shots_and_serve():
    n = 60
    t = ts(n)
    near = make_side(n)
    for k in (10, 25, 40):  # forehands at 2.0, 5.0, 8.0 s
        add_swing(near, k, FOREHAND)
    out = analyze_point_shots((0.0, 11.0), tracks_from(near, absent(n), t),
                              handedness={"near": "right", "far": "right"})
    sides = [s["player_side"] for s in out["shots"]]
    assert sides == ["far", "near", "far", "near", "far", "near"]
    assert out["rally_length"] == 6 and out["prior_filled"] == 3 and out["motion_events"] == 3
    assert [s["source"] for s in out["shots"]].count("prior") == 3
    assert out["shots"][0]["stroke"] == "serve" and out["shots"][0]["source"] == "prior"
    assert all(s["stroke"] == "forehand" for s in out["shots"] if s["source"] == "motion")
    times = [s["t"] for s in out["shots"]]
    assert times == sorted(times)


def test_far_player_absent_point_runs_and_reports_no_far_motion():
    n = 30
    t = ts(n)
    out = analyze_point_shots((0.0, 6.0), tracks_from(make_side(n), absent(n), t))
    assert out["rally_length"] == 0 and out["prior_filled"] == 0
    assert out["shots"] == []


def test_opponent_overlap_is_deduped_and_same_side_double_suppressed():
    n = 40
    t = ts(n)
    near = add_swing(make_side(n), 10, FOREHAND)
    far = add_swing(make_side(n, foot_y=250, h=60, sw=18, facing=-1, cx=600), 11, FOREHAND)
    far["kpconf"][:] = 0.4  # weaker evidence than the near player's
    out = analyze_point_shots((0.0, 7.0), tracks_from(near, far, t), handedness={"near": "right", "far": "right"})
    assert out["deduped"] == 1
    assert [s["player_side"] for s in out["shots"] if s["source"] == "motion"] == ["near"]

    near2 = add_swing(add_swing(make_side(n), 10, FOREHAND), 16, BACKHAND, both=True)  # 1.2 s apart
    out2 = analyze_point_shots((0.0, 7.0), tracks_from(near2, absent(n), t), handedness={"near": "right", "far": "right"})
    assert out2["suppressed"] == 1
    assert [s["source"] for s in out2["shots"]].count("motion") == 1


def test_known_server_prior_serve_goes_to_server():
    n = 40
    t = ts(n)
    near = add_swing(make_side(n), 15, FOREHAND)  # near's first swing 3 s in, not a serve
    out = analyze_point_shots({"start": 0.0, "end": 7.0}, tracks_from(near, absent(n), t), server="far",
                              handedness={"near": "right", "far": "right"})
    assert out["shots"][0] == {**out["shots"][0], "player_side": "far", "stroke": "serve", "source": "prior"}


# --------------------------------------------------------- scripts + hygiene


def _load_script(name):
    path = ROOT / "scripts" / "eval" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


@pytest.mark.skipif(not GOLDEN_TRACKS.exists(), reason="golden tracks fixture missing")
def test_score_shots_on_golden_tracks(tmp_path, capsys):
    mod = _load_script("score_shots")
    out_json = tmp_path / "shots.json"
    results = mod.main(["--tracks", str(GOLDEN_TRACKS), "--json-out", str(out_json)])
    text = capsys.readouterr().out
    assert text.isascii()
    assert "inter-shot interval" in text and "rally length hist" in text
    summary = results[0]["summary"]
    # Fixture = the v0.6.0 runtime's dump of the golden clip (1920x1080 reference): one
    # default segment, and the net-line slots' far-court crop pass sees the far player
    # (assigned far present in ~30 % of frames), so both sides contribute motion shots.
    assert summary["points"] == 1 and summary["points_source"] == "default_segments"
    assert summary["present_frac"]["far"] > 0.2
    assert summary["far_source"] == "assigned"
    assert sum(summary["rally_length_hist"].values()) == 1
    saved = json.loads(out_json.read_text())
    assert saved[0]["summary"]["shots_total"] == summary["shots_total"]
    motion_sides = set()
    for p in saved[0]["points"]:
        assert p["rally_length"] == len(p["shots"])
        motion_sides |= {s["player_side"] for s in p["shots"] if s["source"] == "motion"}
    assert motion_sides == {"near", "far"}


@pytest.mark.skipif(not GOLDEN_TRACKS.exists(), reason="golden tracks fixture missing")
def test_export_shot_clips_collects_events_and_writes_template(tmp_path, monkeypatch):
    mod = _load_script("export_shot_clips")
    tracks = dict(np.load(GOLDEN_TRACKS, allow_pickle=True))
    events = mod.collect_events(tracks, [tuple(s) for s in tracks["default_segments"]], include_prior=False)
    assert events and all(e["source"] == "motion" for e in events)
    # cut step stubbed (PyAV re-encode is exercised by the manual run, not the unit gate)
    cuts = []
    import segmentation.segment as seg
    monkeypatch.setattr(seg, "segment_video", lambda src, iv, out: cuts.append((src, iv, out)))
    monkeypatch.setattr(mod, "video_info", lambda path: (0.0, 24.0))
    rows = mod.main(["--tracks", str(GOLDEN_TRACKS), "--video", "clip.mp4", "--out-dir", str(tmp_path), "--n", "60"])
    assert len(rows) == len(cuts) == len(events)
    for (_, iv, _), r in zip(cuts, rows):
        (lo, hi), = iv
        assert hi - lo == pytest.approx(1.2) and lo <= r["t"] <= hi
    with open(tmp_path / "labels_template.csv") as f:
        got = list(csv.DictReader(f))
    assert [r["event_id"] for r in got] == [r["event_id"] for r in rows]
    assert all(r["label"] == "" for r in got)


def test_score_shots_rebuilds_far_track_from_raw_detections():
    mod = _load_script("score_shots")
    n = 6
    near_box = np.tile([600.0, 350.0, 690.0, 480.0], (n, 1))
    raw = np.full((n, 8, 4), np.nan)
    raw[:, 0] = near_box                       # the near player again
    raw[:, 1] = [1180.0, 150.0, 1210.0, 230.0]  # ball kid at the side
    raw[:, 2] = [620.0, 160.0, 650.0, 230.0]    # far player, centre court
    raw[3, 2] = np.nan                         # missed once
    raw[4:, 2] += [60.0, 0.0, 60.0, 0.0]       # moves, still nearest the previous pick
    raw[:, 3] = [400.0, 600.0, 430.0, 700.0]   # someone low in the frame
    kp = np.zeros((n, 8, 17, 2))
    kp[:, 2] = 7.0
    d = {"near_box": near_box, "near_present": np.ones(n, bool), "raw_boxes": raw, "raw_kp": kp,
         "raw_kpconf": np.ones((n, 8, 17)), "raw_count": np.full(n, 4), "ref_size": np.array([1280, 720]),
         "far_present": np.zeros(n, bool)}
    box, fkp, _, pres = mod._far_from_raw(d)
    assert pres.tolist() == [True, True, True, False, True, True]
    assert np.allclose(box[pres][:, 0], [620, 620, 620, 680, 680])
    assert np.all(fkp[pres] == 7.0)
    d2, used, frac = mod.prepare_far_track(d, "auto")
    assert used == "raw" and frac == 0.0 and d2["far_present"].sum() == 5
    assert mod.prepare_far_track(d, "assigned")[1] == "assigned"


def test_shots_module_stays_free_of_heavy_imports():
    code = ("import sys; import analysis.shots; "
            "bad = [m for m in ('torch', 'cv2', 'av', 'ultralytics', 'onnxruntime') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")})
