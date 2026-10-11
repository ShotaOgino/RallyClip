"""Stroke-label tooling: frame-strip exporter geometry (no video) and the label scorer."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture(scope="module")
def exporter():
    return _load_script("export_shot_frames")


@pytest.fixture(scope="module")
def scorer():
    return _load_script("score_stroke_labels")


# ------------------------------------------------------------------ exporter geometry


def test_crop_rect_expands_box_and_widens_to_min_aspect(exporter):
    # 100 x 200 box centred at (1000, 700): +60 % -> 160 x 320, widened to 0.75 * 320 = 240
    x1, y1, x2, y2 = exporter.crop_rect((1920, 1080), [950, 600, 1050, 800])
    assert (x2 - x1, y2 - y1) == (240, 320)
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == (1000, 700)


def test_crop_rect_clips_to_frame_and_falls_back(exporter):
    x1, y1, x2, y2 = exporter.crop_rect((1920, 1080), [1850, 900, 1950, 1100])
    assert x1 >= 0 and y1 >= 0 and x2 == 1920 and y2 == 1080
    lower_centre = (480, 540, 1440, 1080)
    assert exporter.crop_rect((1920, 1080), None) == lower_centre
    assert exporter.crop_rect((1920, 1080), [np.nan, 0, 10, 10]) == lower_centre
    assert exporter.crop_rect((1920, 1080), [10, 10, 10, 50]) == lower_centre  # degenerate


def test_tile_crops_on_synthetic_frame(exporter):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[540:, 480:1440] = (200, 30, 30)        # lower-centre half is red
    rect = exporter.crop_rect((1920, 1080), None)
    img = exporter.tile_crops([frame, None, frame], rect, tile_w=400, gap=4)
    tile_h = round(400 * 540 / 960)
    assert img.shape == (tile_h, 3 * 400 + 2 * 4, 3) and img.dtype == np.uint8
    assert tuple(img[tile_h // 2, 200]) == (200, 30, 30)       # tile 1: the crop
    assert tuple(img[tile_h // 2, 404 + 200]) == (128, 128, 128)  # tile 2: missing frame
    assert tuple(img[tile_h // 2, 402]) == (255, 255, 255)     # separator
    assert tuple(img[tile_h // 2, 2 * 404 + 200]) == (200, 30, 30)
    cap = exporter.tile_crops([frame], rect, tile_w=400, captions=["0"])
    assert not np.array_equal(cap[:30, :40], img[:30, :40])    # caption drawn top-left


def test_event_box_unions_present_boxes_and_scales(exporter):
    tracks = {"timestamps": np.array([0.0, 0.2, 0.4, 0.6]),
              "near_present": np.array([True, True, False, True]),
              "near_box": np.array([[10, 10, 20, 40], [12, 8, 22, 42], [0, 0, 999, 999], [50, 50, 60, 60]], float)}
    b = exporter.event_box(tracks, "near", 0.2, (2.0, 1.5), window=0.3)
    np.testing.assert_allclose(b, [20, 12, 44, 63])
    assert exporter.event_box(tracks, "near", 5.0, (1.0, 1.0)) is None


def test_stratified_pick_balances_strokes_and_spreads_time(exporter):
    ev = ([{"t": float(i), "predicted_stroke": "forehand"} for i in range(40)]
          + [{"t": 100.0 + i, "predicted_stroke": "serve"} for i in range(5)]
          + [{"t": 200.0 + i, "predicted_stroke": "volley"} for i in range(15)])
    pick = exporter.stratified_pick(ev, 21, seed=0)
    assert len(pick) == 21 and pick == sorted(set(pick))
    kinds = [ev[i]["predicted_stroke"] for i in pick]
    assert kinds.count("serve") == 5 and kinds.count("forehand") == 8 and kinds.count("volley") == 8
    fh_t = [ev[i]["t"] for i in pick if ev[i]["predicted_stroke"] == "forehand"]
    assert min(fh_t) < 6 and max(fh_t) > 33               # spread over the stratum, not clumped
    assert exporter.stratified_pick(ev, 100) == list(range(len(ev)))


# ------------------------------------------------------------------ scorer


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture()
def csv_pair(tmp_path):
    ev_fields = ["event_id", "t", "point_index", "in_point", "merged", "predicted_stroke", "confidence",
                 "event_confidence", "arm", "assumed_hand", "crop", "png"]
    preds = [("e1", "serve", "1", "1", "right"), ("e2", "backhand", "1", "1", "left"),
             ("e3", "forehand", "1", "1", "right"), ("e4", "volley", "1", "0", "right"),
             ("e5", "forehand", "1", "1", "left"), ("e6", "unknown", "1", "1", "right"),
             ("e7", "backhand", "0", "0", "right")]
    _write(tmp_path / "events.csv", ev_fields,
           [{"event_id": e, "t": i, "point_index": 0 if ip == "1" else -1, "in_point": ip, "merged": m,
             "predicted_stroke": p, "arm": arm, "assumed_hand": "left"} for i, (e, p, ip, m, arm) in enumerate(preds)])
    lab_fields = ["event_id", "t", "point_index", "label", "handedness", "label_confidence", "labeller", "dup_of", "notes"]
    labels = [("e1", "serve", "high", ""), ("e2", "forehand", "high", ""), ("e3", "forehand", "low", ""),
              ("e4", "serve", "high", "e1"), ("e5", "no_swing", "high", ""), ("e6", "backhand", "medium", ""),
              ("e7", "no_swing", "high", "")]
    _write(tmp_path / "labels.csv", lab_fields,
           [{"event_id": e, "label": lab, "handedness": "" if lab == "no_swing" else "right", "label_confidence": c,
             "labeller": "test", "dup_of": d} for e, lab, c, d in labels]
           + [{"event_id": "e8", "label": ""}])  # unlabelled + not in events: skipped
    return tmp_path / "labels.csv", tmp_path / "events.csv"


def test_scorer_accuracy_confusion_and_false_events(scorer, csv_pair, tmp_path, capsys):
    labels, events = csv_pair
    out = scorer.main(["--labels", str(labels), "--events", str(events), "--json-out", str(tmp_path / "s.json")])
    assert capsys.readouterr().out.isascii()
    a = out["all"]
    assert a["labelled"] == 7
    assert a["false_event_fraction"] == round(2 / 7, 3)
    assert a["duplicate_fraction"] == round(1 / 7, 3)
    acc = a["accuracy"]                                  # gradeable: e1 e2 e3 e4 e6
    assert (acc["n"], acc["correct"], acc["coverage"]) == (5, 2, 0.8)
    assert a["accuracy_excluding_low_conf_labels"]["n"] == 4   # e3 is low
    c = a["confusion"]
    assert c["serve"]["serve"] == 1 and c["serve"]["volley"] == 1
    assert c["forehand"]["backhand"] == 1 and c["forehand"]["forehand"] == 1
    assert c["no_swing"]["forehand"] == 1 and c["no_swing"]["backhand"] == 1
    pc = a["per_class"]
    assert pc["serve"] == {"support": 2, "predicted": 1, "precision": 1.0, "recall": 0.5}
    assert pc["forehand"]["precision"] == 1.0 and pc["forehand"]["recall"] == 0.5
    fb = a["forehand_backhand"]
    assert (fb["correct"], fb["swapped"], fb["n_labelled"]) == (1, 1, 3)
    assert fb["by_handedness"] == {"observed=right assumed=left": {"correct": 1, "swapped": 1}}
    assert fb["by_event_arm"] == {"observed=right event_arm=left": {"swapped": 1},
                                  "observed=right event_arm=right": {"correct": 1}}
    assert out["between"]["false_event_fraction"] == 1.0
    assert out["merged"]["labelled"] == 5
    assert (tmp_path / "s.json").exists()


def test_scorer_rejects_unknown_label(scorer, csv_pair):
    labels, events = csv_pair
    text = labels.read_text().replace(",forehand,right,low", ",smash,right,low")
    labels.write_text(text)
    with pytest.raises(SystemExit):
        scorer.load_rows(str(labels), str(events))


def test_committed_pseudo_labels_are_well_formed():
    path = ROOT / "labels" / "2026_08_24_2R_Tsukamoto_Takeda.shot_labels.csv"
    if not path.exists():
        pytest.skip("pseudo-label file not present")
    rows = list(csv.DictReader(open(path, newline="")))
    vocab = {"serve", "forehand", "backhand", "volley", "overhead", "unknown", "no_swing"}
    ids = {r["event_id"] for r in rows}
    assert len(rows) >= 150 and len(ids) == len(rows)
    for r in rows:
        assert r["label"] in vocab and r["label_confidence"] in {"high", "medium", "low"}
        assert r["handedness"] in {"", "left", "right"} and r["labeller"] == "claude-opus-pseudo"
        assert not r["dup_of"] or r["dup_of"] in ids


# ------------------------------------------------------------------ relabel + detection


def test_relabel_events_matches_by_time_and_marks_undetected(exporter):
    old = [{"event_id": "e1", "t": "10.0", "point_index": "0", "in_point": "1", "png": "e1.png"},
           {"event_id": "e2", "t": "12.0", "point_index": "0", "in_point": "1"},
           {"event_id": "e3", "t": "30.0", "point_index": "-1", "in_point": "0"}]
    cur = [{"t": 10.1, "merged": 1, "predicted_stroke": "serve", "confidence": 0.8, "event_confidence": 0.7,
            "arm": "right", "assumed_hand": "right"},
           {"t": 10.05, "merged": 0, "predicted_stroke": "forehand", "confidence": 0.5, "event_confidence": 0.4,
            "arm": "left", "assumed_hand": "right"},
           {"t": 20.0, "merged": 1, "predicted_stroke": "backhand", "confidence": 0.5, "event_confidence": 0.5,
            "arm": "left", "assumed_hand": "right"}]
    rows, extra = exporter.relabel_events(old, cur)
    assert [r["event_id"] for r in rows] == ["e1", "e2", "e3"]
    assert rows[0]["predicted_stroke"] == "forehand" and rows[0]["detected"] == 1  # nearest wins (0.05 s)
    assert rows[0]["png"] == "e1.png" and rows[0]["t"] == "10.0"                   # old identity kept
    assert rows[1]["predicted_stroke"] == "none" and rows[1]["detected"] == 0       # 10.1 is 1.9 s away
    assert rows[2]["detected"] == 0 and rows[2]["in_point"] == "0"
    assert extra == 2


def test_scorer_detection_section(scorer, csv_pair):
    labels, events = csv_pair
    rows = list(csv.DictReader(open(events, newline="")))
    for r in rows:  # e5 (no_swing) and e6 (backhand) no longer detected
        gone = r["event_id"] in ("e5", "e6")
        r["detected"] = "0" if gone else "1"
        if gone:
            r["predicted_stroke"] = "none"
    _write(events, list(rows[0].keys()), rows)
    out = scorer.main(["--labels", str(labels), "--events", str(events)])
    d = out["all"]["detection"]
    assert (d["labelled"], d["detected"], d["no_swing_rejected"], d["no_swing_labelled"]) == (7, 5, 1, 2)
    assert d["false_event_rate_detected"] == round(1 / 5, 3)
    assert d["swing_recall"] == round(4 / 5, 3) and d["stroke_recall"] == round(4 / 5, 3)
    acc = out["all"]["accuracy"]
    assert (acc["n"], acc["correct"], acc["coverage"]) == (5, 2, 0.8)  # e6: "none" = wrong, not covered (was "unknown")
    assert out["all"]["confusion"]["backhand"]["none"] == 1


def test_scorer_without_detected_column_has_no_detection_section(scorer, csv_pair):
    labels, events = csv_pair
    assert scorer.main(["--labels", str(labels), "--events", str(events)])["all"]["detection"] is None
