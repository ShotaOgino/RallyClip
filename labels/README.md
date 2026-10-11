# Stroke labels (roadmap Phase 2)

**Status: model-generated pseudo-labels. Human verification pending.** Do not treat these
files as ground truth. Use them to find large, obvious failures; any number derived from
them needs a human pass before it goes into a decision.

## Files

| File | What |
|---|---|
| `2026_08_24_2R_Tsukamoto_Takeda.events.csv` | 186 near-player swing events on beta-tennis match 2R, with the rule-based prediction (`analysis.shots`). Written by `scripts/eval/export_shot_frames.py` (v0.6.0 track dump, GT points from the `.tdproj`, `--n 300 --include-between --seed 0`; there were only 186 swings, so all of them were exported). `png` is relative to the exporter's `--out-dir`. |
| `2026_08_24_2R_Tsukamoto_Takeda.shot_labels.csv` | One label per event: `event_id, t, point_index, label, handedness, label_confidence, labeller, dup_of, notes`. |

Score them with:

```bash
PYTHONPATH=src $PY scripts/eval/score_stroke_labels.py \
  --labels labels/2026_08_24_2R_Tsukamoto_Takeda.shot_labels.csv \
  --events labels/2026_08_24_2R_Tsukamoto_Takeda.events.csv
```

## Provenance

- **Labeller:** `claude-opus-pseudo` (Claude Opus 5.5). It looked at one PNG per event:
  three crops of the near player at t - 0.25 s, t and t + 0.25 s, decoded from the 1080p59.94
  source and cropped to the player's box expanded by 60 %. About 10 events were also checked
  on denser extra frames. The labeller did not see `events.csv` (prediction, arm, assumed
  hand) until every event had a label. The PNG file names carry only the event id. The PNGs
  are not in git. To regenerate them, run the exporter with the same arguments
  (deterministic, about 30 s).
- **Vocabulary:** `serve, forehand, backhand, volley, overhead` (stroke classes), `unknown`
  (the frames show a stroke-like motion, but not enough to tell which stroke), `no_swing`
  (the frames show no stroke at all, such as walking, bouncing the ball or using the towel;
  a false event). `handedness` is the hitting hand when the frames show it.
  `label_confidence` is `high` (contact or an unambiguous phase with the ball visible),
  `medium` (only preparation or follow-through, but unambiguous) or `low`.
- **`dup_of`:** the event that holds the same stroke's contact. This is mostly a serve
  detected once at the toss and again about 1 s later at contact. The duplicate keeps the
  stroke label, because the motion is part of that serve.
- **Players:** the near player changes at change-overs. Both players were seen hitting
  right-handed: racket in the right hand on serves, free arm on the left, two-handed
  backhands on the left side. One player wears a red waistband and white wristbands. The
  other wears a grey top or patterned shorts and a black wristband on the free (left) arm.

## Known limits

- One labeller, which is the same model family as the code under test. Nobody has checked
  the labels independently.
- Three frames 0.25 s apart often miss the contact. Forehand vs backhand is then judged
  from preparation and follow-through, and low and some medium labels rest on that.
- One match, two junior players, one venue and one camera. No volleys turned up among the
  labelled events (one overhead), so the volley and overhead rules are not measured, apart
  from their false positives.

## To verify (human)

Open each PNG with its row and correct `label`, `handedness` and `dup_of`. Set `labeller`
to a human id on every row you check. Start with `label_confidence == low` and with
`unknown`.
