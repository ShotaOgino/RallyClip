# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-12 (session: merge the upscaled far crop into the port; re-dump and
re-evaluate all five beta-tennis matches)._

## Repo state

- Fork `ShotaOgino/RallyClip` (`origin`; `upstream` = `iroblesrazzaq/RallyClip`, read-only).
  Branch **`feat/port-v050-analysis`** (worktree `~/Documents/RallyClip-v060-port`), based on
  `feat/runtime-v060` (9dd5ad1) + `origin/feat/far-crop-upscale` (be1627d, merged 9a964ce).
  No PR opened (user decides). Another agent works on `feat/stroke-fixes`
  (`~/Documents/RallyClip-v060-strokefix`: `src/analysis/shots.py`, stroke-label scripts).
- Commits this session: `9a964ce` merge far crop; `0bc2ca5` far_reach server detection +
  `server_method` + `score_shots.py --server phase1`; `5607c17` jp-junior 0.7 / 2.0;
  docs commit (this file, DECISIONS, testing, features, roadmap).
- Env: worktree uv `.venv` (py3.12.7, `--extra cpu --extra dev --extra train`),
  `PY=.venv/bin/python`. Artifact `models/rallyclip_v0.6.0/` (gitignored).
- Gate (idle machine): **610 passed, 9 skipped, 38 deselected**; compileall 0; golden parity
  1 passed (no regeneration). Under load (avg ~80-97, dumps + other jobs) the two GUI preview
  latency benchmarks fail; they pass alone / idle.

## Dumps (merged code, default device)

`~/Documents/RallyClip-0.5.0/logs/eval/beta-tennis-v060/<match>.farcrop.tracks.npz` for
1R / 2R / 3R / SF / P1R (all with `geometry_json`; the 2R one is the far-crop branch's dump
with geometry added from `compute_court_geometry`, it has no `raw_*` keys). Logs next to them
(`<match>.farcrop.log`). 6-65 min per match depending on machine load.

## Five-match tables (tdproj GT; P1R's tdproj is the `AB選` file in the SF folder)

(a) Point detection (IoU >= 0.5; FP = predictions overlapping no GT point; time-F1 =
beta-tennis protocol):

| match | decode | GT | pred | F1 | merged | FP | missed | time-F1 |
|---|---|---|---|---|---|---|---|---|
| 1R | manifest default | 71 | 56 | 0.740 | 15 | 2 | 2 | 0.875 |
| 2R | manifest default | 55 | 49 | 0.827 | 7 | 2 | 1 | 0.872 |
| 3R | manifest default | 76 | 66 | 0.831 | 8 | 1 | 3 | 0.862 |
| SF | manifest default | 59 | 64 | 0.878 | 0 | 5 | 2 | 0.842 |
| P1R | manifest default | 95 | 90 | 0.876 | 6 | 1 | 2 | 0.876 |
| 1R | jp-junior old 0.5/1.0 | 71 | 58 | 0.760 | 16 | 4 | 1 | 0.895 |
| 2R | jp-junior old 0.5/1.0 | 55 | 50 | 0.819 | 8 | 4 | 1 | 0.911 |
| 3R | jp-junior old 0.5/1.0 | 76 | 69 | 0.869 | 8 | 2 | 1 | 0.900 |
| SF | jp-junior old 0.5/1.0 | 59 | 63 | 0.869 | 0 | 5 | 1 | 0.869 |
| P1R | jp-junior old 0.5/1.0 | 95 | 98 | 0.819 | 6 | 8 | 2 | 0.905 |
| 1R | **jp-junior new 0.7/2.0** | 71 | 67 | 0.913 | 5 | 2 | 1 | 0.917 |
| 2R | **jp-junior new 0.7/2.0** | 55 | 57 | 0.946 | 1 | 2 | 1 | 0.919 |
| 3R | **jp-junior new 0.7/2.0** | 76 | 76 | 0.947 | 1 | 0 | 3 | 0.912 |
| SF | **jp-junior new 0.7/2.0** | 59 | 74 | 0.827 | 0 | 14 | 3 | 0.878 |
| P1R | **jp-junior new 0.7/2.0** | 95 | 98 | 0.881 | 4 | 2 | 2 | 0.912 |

Means: default 0.830, old jp-junior 0.827, new jp-junior **0.903** (grid threshold 0.4-0.7 x
floor {0,1,2}, smooth 1.5, end pad 1.6; 0.75 / 0.8 checked: 0.895 / 0.879). Profile updated
(DECISIONS 2026-10-12). SF's 14 FPs at 0.7 are the open item.

(b) Phase 1 (scoreboard-derived reference; acc over detected / coverage):

| match | server far_reach (new) | near + elimination (far hidden) | raise only | identity | side changes P/R |
|---|---|---|---|---|---|
| 1R | 1.0 / 0.986 | 1.0 / 0.986 | 1.0 / 0.704 | 1.0 | 1.0/1.0 (5) |
| 2R | 1.0 / 0.964 | 0.982 / 1.0 | 1.0 / 0.636 | 1.0 | 1.0/1.0 (4) |
| 3R | 1.0 / 0.974 | 0.987 / 1.0 | 1.0 / 0.658 | 1.0 | 1.0/1.0 (6) |
| SF | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 0.61 | 1.0 | 1.0/1.0 (4) |
| P1R | 0.978 / 0.979 | 0.968 / 1.0 | 0.987 / 0.80 | 0.968 | 1.0/0.875 (8; tiebreak) |

Identity / side changes are for far_reach (raise only: identity 0.527 on 2R, 0.453 on P1R).
`detect_server` already scored far raises symmetrically, but the strict raise test catches
only 10-60 % of far serves, and elimination switches off when the far player is visible;
hence the new `far_reach` method.

Serve side, visual vs scoreboard parity (far_reach run): 1R 39/71 = 0.55, 2R 51/55 = 0.93
(far serves 21/21, previously 22/23 via the receiver), 3R 49/76 = 0.65, SF 57/59 = 0.97
(far 28/30), P1R 66/95 = 0.69. Far serves: server-feet visual == receiver-feet visual on
164/164 points (two independent observations), so the 1R/3R/P1R disagreements are the parity
reference (separately clipped first-serve faults with gaps > 5 s, repeated clips), not the
positions.

(c) Phase 2 (`score_shots.py --server phase1`; in brackets: far track hidden, v0.5.0-like):

| match | shots/point mean (median) | motion near/far | prior-filled | inter-shot median s (p10-p90) | serve side per game |
|---|---|---|---|---|---|
| 1R | 4.21 (4) [3.42] | 135/146 [136/0] | 0.060 [0.44] | 1.60 (0.90-2.56) | 1.0 |
| 2R | 4.46 (4) [3.75] | 117/109 [118/0] | 0.078 [0.43] | 1.61 (0.91-2.60) | 0.981 |
| 3R | 3.61 (3) [2.91] | 130/123 [132/0] | 0.077 [0.40] | 1.50 (0.80-2.78) | 0.987 |
| SF | 7.09 (6) [6.05] | 188/196 [191/0] | 0.081 [0.47] | 1.61 (0.92-2.48) | 1.0 |
| P1R | 4.70 (4) [3.58] | 189/211 [191/0] | 0.103 [0.44] | 1.59 (0.95-2.71) | 0.926 |

Without `--server phase1` the shot pass's own serve inference drops serve-side-per-game to
0.75-0.91 now that far swings exist. Inter-shot p10 fell from 1.2 s to ~0.9 s (a far swing
and the next near swing can now both be motion events); no shot labels yet.

(d) Far presence (assigned far slot): all frames / inside tdproj points / points with far >= 50 %:
1R 0.765 / 0.985 / 71/71; 2R 0.798 / 0.972 / 55/55; 3R 0.743 / 0.916 / 75/76;
SF 0.862 / 0.989 / 59/59; P1R 0.816 / 0.967 / 95/95 (v0.6.0 native crop on 2R: 0.064 / 0.02).

## Next steps

1. Retrain with the adaptive crop in `scripts/extract_crop_poses.py` so train and serve far
   features match; then re-tune jp-junior (today's 0.7 / 2.0 compensates for the mismatch).
2. SF at 0.7: 14 pure FPs (74 predictions vs 59 points) — inspect them (warm-up / breaks?).
3. Serve-side parity reference: second-serve merge by gap alone fails on 1R/3R/P1R; use the
   visual side (now two independent observations) to flag fault clips, or label a few games.
4. Phase 2: far strokes now observed — label ~300 swings (`export_shot_clips.py`), measure
   event F1 / stroke accuracy; retire `score_shots.py::_far_from_raw` once no dump needs it.
   Coordinate with `feat/stroke-fixes` (shots.py lives there).
5. Phase 1: tiebreak side changes (P1R's missed change); identity confidence still relies on
   the gap cue.
6. Wire `--server phase1` style joining into the engine outputs (shots are not in the runtime yet).
7. Widen the eval set beyond one player / venue / day.
