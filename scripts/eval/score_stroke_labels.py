"""Score the rule-based stroke type (analysis.shots.classify_stroke) against stroke labels.

PYTHONPATH=src $PY scripts/eval/score_stroke_labels.py --labels L.shot_labels.csv --events E.events.csv
    [--subset in_point|merged|between|all] [--json-out out.json]

``--events`` is the ``events.csv`` written by ``scripts/eval/export_shot_frames.py``
(event_id, predicted_stroke, in_point, merged, arm, assumed_hand, ...); ``--labels`` has
event_id, label, handedness, label_confidence (+ optional dup_of, notes). Joined on
event_id; events without a label row (or with an empty label) are skipped.

Label vocabulary: serve, forehand, backhand, volley, overhead, unknown (the frames do not
show which stroke), no_swing (the frames show no stroke at all: walking, ball bouncing,
towel; a false event). ``dup_of`` names the event holding the same stroke's contact
(e.g. a serve detected at the toss and again at contact).

Reported per subset (default: every subset that has rows):

* event sanity: false-event fraction (no_swing / labelled), duplicate fraction, and the
  label distribution;
* gradeable events = label in the five stroke classes: accuracy of the prediction
  (a predicted "unknown" counts as wrong; coverage = share not predicted unknown),
  overall and excluding low-confidence labels; confusion matrix (rows = label incl.
  unknown / no_swing, columns = prediction); per-class precision / recall;
* forehand/backhand: among events labelled forehand or backhand and predicted forehand
  or backhand, correct vs swapped, broken down by (observed handedness, the detector's
  assumed hand) and by the event's arm (the wrist that carried the swing).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional

STROKES = ["serve", "forehand", "backhand", "volley", "overhead"]
PRED_COLS = STROKES + ["unknown"]
LABEL_ROWS = STROKES + ["unknown", "no_swing"]
SUBSETS = {
    "all": lambda r: True,
    "in_point": lambda r: str(r.get("in_point", "1")) == "1",
    "merged": lambda r: str(r.get("merged", "0")) == "1",
    "between": lambda r: str(r.get("in_point", "1")) == "0",
}


def load_rows(labels_path: str, events_path: str) -> List[dict]:
    with open(events_path, newline="") as f:
        events = {r["event_id"]: r for r in csv.DictReader(f)}
    rows = []
    with open(labels_path, newline="") as f:
        for lab in csv.DictReader(f):
            ev = events.get(lab["event_id"])
            label = (lab.get("label") or "").strip().lower()
            if ev is None or not label:
                continue
            if label not in LABEL_ROWS:
                raise SystemExit(f"{lab['event_id']}: unknown label {label!r} (vocabulary: {', '.join(LABEL_ROWS)})")
            rows.append({**ev, "label": label, "handedness": (lab.get("handedness") or "").strip().lower(),
                         "label_confidence": (lab.get("label_confidence") or "").strip().lower(),
                         "dup_of": (lab.get("dup_of") or "").strip(),
                         "predicted_stroke": (ev.get("predicted_stroke") or "unknown").strip().lower()})
    return rows


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 3) if b else None


def accuracy(rows: List[dict]) -> Dict[str, object]:
    g = [r for r in rows if r["label"] in STROKES]
    correct = sum(r["predicted_stroke"] == r["label"] for r in g)
    covered = [r for r in g if r["predicted_stroke"] != "unknown"]
    return {"n": len(g), "correct": correct, "accuracy": _ratio(correct, len(g)),
            "coverage": _ratio(len(covered), len(g)),
            "accuracy_when_predicted": _ratio(sum(r["predicted_stroke"] == r["label"] for r in covered), len(covered))}


def confusion(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    cols = PRED_COLS + sorted({r["predicted_stroke"] for r in rows} - set(PRED_COLS))
    m = {lab: {c: 0 for c in cols} for lab in LABEL_ROWS}
    for r in rows:
        m[r["label"]][r["predicted_stroke"]] += 1
    return m


def per_class(rows: List[dict]) -> Dict[str, Dict[str, object]]:
    g = [r for r in rows if r["label"] in STROKES]
    out = {}
    for c in STROKES:
        tp = sum(r["label"] == c and r["predicted_stroke"] == c for r in g)
        n_pred = sum(r["predicted_stroke"] == c for r in g)
        n_true = sum(r["label"] == c for r in g)
        out[c] = {"support": n_true, "predicted": n_pred, "precision": _ratio(tp, n_pred), "recall": _ratio(tp, n_true)}
    return out


def fh_bh_breakdown(rows: List[dict]) -> Dict[str, object]:
    sides = ("forehand", "backhand")
    g = [r for r in rows if r["label"] in sides and r["predicted_stroke"] in sides]
    by_hand: Dict[str, Counter] = defaultdict(Counter)
    by_arm: Dict[str, Counter] = defaultdict(Counter)
    for r in g:
        ok = "correct" if r["predicted_stroke"] == r["label"] else "swapped"
        by_hand[f"observed={r['handedness'] or '?'} assumed={r.get('assumed_hand') or '?'}"][ok] += 1
        by_arm[f"observed={r['handedness'] or '?'} event_arm={r.get('arm') or '?'}"][ok] += 1
    labelled_fb = [r for r in rows if r["label"] in sides]
    return {"n_labelled": len(labelled_fb),
            "n_predicted_fh_or_bh": len(g),
            "correct": sum(r["predicted_stroke"] == r["label"] for r in g),
            "swapped": sum(r["predicted_stroke"] != r["label"] for r in g),
            "by_handedness": {k: dict(v) for k, v in sorted(by_hand.items())},
            "by_event_arm": {k: dict(v) for k, v in sorted(by_arm.items())},
            "observed_handedness": dict(Counter(r["handedness"] or "?" for r in rows if r["label"] in STROKES)),
            "assumed_hand": dict(Counter(r.get("assumed_hand") or "?" for r in rows))}


def score(rows: List[dict]) -> Dict[str, object]:
    n = len(rows)
    not_low = [r for r in rows if r["label_confidence"] != "low"]
    return {
        "labelled": n,
        "label_distribution": dict(Counter(r["label"] for r in rows).most_common()),
        "prediction_distribution": dict(Counter(r["predicted_stroke"] for r in rows).most_common()),
        "false_event_fraction": _ratio(sum(r["label"] == "no_swing" for r in rows), n),
        "unclear_fraction": _ratio(sum(r["label"] == "unknown" for r in rows), n),
        "duplicate_fraction": _ratio(sum(bool(r["dup_of"]) for r in rows), n),
        "accuracy": accuracy(rows),
        "accuracy_excluding_low_conf_labels": accuracy(not_low),
        "confusion": confusion(rows),
        "per_class": per_class(rows),
        "forehand_backhand": fh_bh_breakdown(rows),
    }


def print_report(name: str, s: Dict[str, object]) -> None:
    p = print
    p(f"== {name}: {s['labelled']} labelled events ==")
    p(f"labels: {s['label_distribution']}")
    p(f"predictions: {s['prediction_distribution']}")
    p(f"false events (no_swing): {s['false_event_fraction']}  unclear (unknown): {s['unclear_fraction']}  "
      f"duplicates: {s['duplicate_fraction']}")
    for key in ("accuracy", "accuracy_excluding_low_conf_labels"):
        a = s[key]
        p(f"{key}: {a['correct']}/{a['n']} = {a['accuracy']}  coverage {a['coverage']}  "
          f"accuracy when predicted {a['accuracy_when_predicted']}")
    conf = s["confusion"]
    cols = list(next(iter(conf.values())).keys())
    p("confusion (rows = label, cols = predicted):")
    p("  " + "label".ljust(10) + "".join(c[:8].rjust(9) for c in cols))
    for lab, row in conf.items():
        if sum(row.values()):
            p("  " + lab.ljust(10) + "".join(str(row[c]).rjust(9) for c in cols))
    p("per class: " + "; ".join(f"{c} P={v['precision']} R={v['recall']} (n={v['support']})"
                                 for c, v in s["per_class"].items()))
    fb = s["forehand_backhand"]
    p(f"forehand/backhand: {fb['correct']} correct, {fb['swapped']} swapped of {fb['n_predicted_fh_or_bh']} "
      f"predicted FH/BH ({fb['n_labelled']} labelled FH/BH)")
    for k, v in fb["by_handedness"].items():
        p(f"  {k}: {v}")
    for k, v in fb["by_event_arm"].items():
        p(f"  {k}: {v}")
    p(f"  observed handedness (stroke-labelled events): {fb['observed_handedness']}; assumed: {fb['assumed_hand']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--subset", choices=sorted(SUBSETS), default=None, help="only this subset (default: all with rows)")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    rows = load_rows(a.labels, a.events)
    names = [a.subset] if a.subset else ["all", "in_point", "merged", "between"]
    out = {}
    for name in names:
        sub = [r for r in rows if SUBSETS[name](r)]
        if not sub:
            continue
        out[name] = score(sub)
        print_report(name, out[name])
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
