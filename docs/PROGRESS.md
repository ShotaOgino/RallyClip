# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-12 (session: Phase 2 stroke pseudo-labels on beta-tennis 2R, branch
`feat/shot-labels-2r` off `feat/port-v050-analysis`, worktree `~/Documents/RallyClip-v060-labels`).
The port state below is unchanged._

## This session: stroke labels (see DECISIONS 2026-10-12)

- New: `scripts/eval/export_shot_frames.py` (3-crop PNG strip per near swing, PyAV, one
  seek per point; `--hand` counterfactual), `scripts/eval/score_stroke_labels.py`,
  `tests/test_stroke_labels.py` (+ `test_shots.py`: 32 passed), `labels/` (README,
  2R `events.csv` + `shot_labels.csv`). Full default gate NOT run (machine loaded by other
  jobs); targeted tests + compileall only.
- **Labels are model pseudo-labels (`claude-opus-pseudo`), human verification pending.**
  186 events: serve 54, forehand 52, no_swing 38, backhand 26, unknown 15, overhead 1.
- Rule classifier on 2R: in-point accuracy 0.361 (merged shots 0.402); serve P 1.0 / R 0.66;
  FH/BH swapped 43/56 because `estimate_handedness` says `left` while both players are
  right-handed; with `--hand right`: in-point 0.566, merged 0.637. Volley rule: 23
  in-point predictions, 0 real. False events 7 % in points, 65 % between points.
- PNGs (not in git) were in the session scratchpad; regenerate with the exporter (README).

## Repo state

- Fork `ShotaOgino/RallyClip` (`origin`; `upstream` = `iroblesrazzaq/RallyClip`, read-only).
  Branch **`feat/port-v050-analysis`**, branched from `origin/feat/runtime-v060` (9dd5ad1,
  same as upstream). Worktree `~/Documents/RallyClip-v060-port`; the clone
  `~/Documents/RallyClip-v060` holds a separate local branch `feat/far-crop-upscale`
  (geometry-driven upscaled far crop, another agent, not pushed).
- Commits on this branch:
  - `08f2fe8` decode overrides + `end_pad_sec` + GUI decode presets (jp-junior) + 32 GB upload cap
  - `7dc3723` `scripts/eval/` decode tooling (`dump_heatmap_tracks.py`, `score_heatmap_decode.py`)
  - `fbee612` `src/analysis/` (players / shots / score) + opt-in player outputs (engine, CLI
    `--players`, GUI always on, `/api/library/<id>/points`) + Phase 1/2/4 eval scripts + tests
  - docs/features commit (this file, DECISIONS, REPO_MAP, testing, features.json, roadmap)
- No PR opened (user decides). PR review would be Greptile only.
- Env: worktree uv `.venv` (py3.12.7) from `uv sync --extra cpu --extra dev --extra train`
  (the `[train]` extra is needed for the training test modules); `PY=.venv/bin/python`
  (`uv run` rewrites `uv.lock`'s version line, don't commit that). Artifact
  `models/rallyclip_v0.6.0/` copied from the clone (gitignored).
- Default gate: **558 passed, 9 skipped, 38 deselected** (baseline before the port: 440
  passed + 1 wall-clock preview-benchmark failure that only appears under machine load).
  Compile gate exit 0. Golden parity 1 passed; golden clip CSV byte-identical to unmodified
  `feat/runtime-v060` on this machine.

## What was ported (see DECISIONS 2026-10-11 "Port the v0.5.0 snapshot work")

1. **Decode overrides**: CLI `--threshold --merge-gap-sec --smooth-sigma-frames
   --hybrid-min-duration-sec --end-pad-sec`, config.toml `[run]`, `/api/upload-and-start`;
   `RunRequest.decode_params` merged over the manifest in `HeatmapHybridModel._decode_config`.
   `end_pad_sec` pads after merge + floor, re-merges touch/overlap, clamps.
2. **Decode presets**: `DECODE_PRESETS` (default, jp-junior), `/api/config/defaults
   .decode_presets`, `decode_preset` in upload config, frontend select (hidden if absent),
   `configs/decode/jp-junior.toml`. **jp-junior on v0.6.0 = threshold 0.5,
   smooth_sigma_frames 1.5, merge_gap_sec 0.0, hybrid_min_duration_sec 1.0, end_pad_sec 1.6.**
3. **Upload cap 32 GB** (`MAX_UPLOAD_BYTES`, `RALLYCLIP_MAX_UPLOAD_GB`, JSON 413,
   `max_upload_bytes`). `feat/runtime-v060` still had 2 GB; upstream PR #60 (open, vs
   `main`) removes the cap entirely, so expect a conflict there.
4. **`scripts/eval/`**: dump (v2: tees `iter_slot_players`; v1: `iter_build_features`; raw
   pre-filter detections kept), decode scorer (base = dump's manifest params),
   `analyze_players.py`, `score_shots.py`, `export_shot_clips.py`, `reconstruct_score.py`.
   All run on the golden clip on this branch.
5. **`src/analysis/`** + engine/CLI/GUI wiring. The player sink tees the v2 slot stream (near
   from below the net, far from the crop pass, 1920x1080 reference). `PlayerTracks` keeps
   the shipped far record per frame and fills gaps from the raw pick. Fixture
   `tests/fixtures/golden_cli/clip.tracks.npz` regenerated on this branch.
6. Docs: DECISIONS (snapshot entries carried over + port entry), REPO_MAP routes,
   testing counts, features.json (`heatmap-decode-overrides`, `upload-cap-32gb`,
   `phase1-player-identity`, `phase2-shot-events`, `phase4-score-reconstruction`),
   `docs/service-roadmap-plan.md`.

Not ported: the snapshot's `logs/eval/` outputs (gitignored), its `.gstack/` ignore line,
and its README / pyproject edits (none were part of the work).

## v0.6.0 numbers so far

- Golden clip (this branch): far player present in 36/120 kept frames (v0.5.0: 0); shots
  report 2 near + 2 far motion swings in the one point.
- beta-tennis 2R (measured in the v0.5.0 snapshot session with the v0.6.0 clone + a scratch
  dump): court geometry OK (far baseline ~y400, net ~y465 in 1080p); point F1 (IoU>=0.5)
  default 0.83 (v0.5.0 default 0.79); with threshold 0.5 / hybrid_min_duration_sec 1.0 /
  smooth 1.5 / end pad 1.6: **0.925** (v0.5.0 jp-junior profile: 0.933). Far slot present in
  664/10300 frames but only ~2% inside points; frame test: native-scale crop finds the far
  player 1/6, 2x upscaled crop 4/6.
- Phase 1/2/4 real-match numbers (server 0.97-1.0, identity 0.97-1.0 on 1R/2R/3R/SF/P1R) are
  from v0.5.0 dumps; not re-run on v0.6.0 dumps yet.

## Next steps

1. User: review the branch / decide on a PR to upstream (`feat/runtime-v060` or `main`).
2. Re-dump the five beta-tennis matches with `scripts/eval/dump_heatmap_tracks.py` on this
   branch (logs/eval/beta-tennis/v060/), confirm jp-junior on 1R/3R/SF/P1R (only 2R was
   measured), and re-run `analyze_players.py` / `score_shots.py`.
3. Far player: take the `feat/far-crop-upscale` result (upscaled, geometry-driven crop);
   if far presence inside points >= 70%, re-run Phase 1/2 with far evidence (server by
   observation, far strokes) and retire `score_shots.py::_far_from_raw`.
4. Phase 2: human pass over `labels/2026_08_24_2R_*.shot_labels.csv` (low / unknown
   first); then fix handedness (`estimate_handedness` says `left` on 2R: free arm or a pose L/R swap, not diagnosed; per player
   via Phase 1 identity, or from the serve arm), tighten the volley rule, and keep the
   contact (not the toss) when the same-side merge collapses a serve pair. Label a second
   match.
5. Phase 1: tiebreak side changes (every 6 points, no gap).
6. Widen the eval set beyond one player / venue.
