# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-11 (session: port of the v0.5.0 snapshot work onto the v0.6.0
runtime branch)._

## Merged from feat/far-crop-upscale (2026-10-12) — far-court crop pass

- `extraction.crop_pass.adaptive_crop`: far-court window from the court geometry's far-baseline
  corners + net line, upscaled so the far player is ~110 px; fixed window stays the fallback
  (no far corners / manifest `far_crop.mode: fixed`). Threaded through
  `rallyclip_engine.models._far_crop_kwargs` -> `PoseExtractor.iter_pose_frames(crop_out_size, crop_imgsz)`;
  static CoreML canvas fit in the extractor.
- `netline_slots.pick_far`: in-far-court > static-in-court > off-court tiers; static off-court
  candidates never picked (`StaticDetections`); counters on `DataPreprocessor.far_pick_stats`.
- 2R first 10 min: far present inside true points 0.8% -> 97.8% (CoreML) / 99.9% (CPU) at
  ~1.0x (CoreML) / ~1.02-1.06x (CPU) pose cost. Details: docs/DECISIONS.md 2026-10-11.
- Gate: 407 passed, 12 skipped, 27 deselected (torch-only modules ignored, see testing.md);
  golden parity passes without regeneration.

### Far-crop next steps

1. Re-extract training crop poses with the adaptive window (`scripts/extract_crop_poses.py` still
   uses the fixed window) and retrain, so train/serve far features match.
2. Evaluate segment accuracy on all labeled beta-tennis matches with the new crop (only a
   10-min segment sanity check so far: unchanged 12/17 matches).


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

## Serve side (2026-10-11, added after the port)

`src/analysis/serve_side.py` + rows `serve_side*` in `analyze_points` / `_points.json`
(`serve_side_check`), geometry via `PlayerRecordSink.geometry` and npz `geometry_json`
(engine sidecar + `dump_heatmap_tracks.py`), eval table in `analyze_players.py --geometry`.
2R (v0.6.0 dump, geometry recomputed from the video): visual vs scoreboard parity 50/55 =
0.909 (near serves 28/32, far serves via the near receiver 22/23); confidence median 0.975.
See DECISIONS 2026-10-11 "Serve side". Gate 580 passed + 2 wall-clock benchmark failures
under load avg ~80; golden parity 1 passed. The existing 2R dump has no `geometry_json`
(it predates the key): pass `--geometry` or re-dump.

## Next steps

1. User: review the branch / decide on a PR to upstream (`feat/runtime-v060` or `main`).
2. Re-dump the five beta-tennis matches with `scripts/eval/dump_heatmap_tracks.py` on this
   branch (logs/eval/beta-tennis/v060/), confirm jp-junior on 1R/3R/SF/P1R (only 2R was
   measured), and re-run `analyze_players.py` / `score_shots.py`.
3. Far player: take the `feat/far-crop-upscale` result (upscaled, geometry-driven crop);
   if far presence inside points >= 70%, re-run Phase 1/2 with far evidence (server by
   observation, far strokes) and retire `score_shots.py::_far_from_raw`.
4. Phase 2 labels: `export_shot_clips.py` on one match, label ~300 swings.
5. Phase 1: tiebreak side changes (every 6 points, no gap).
5b. Serve side on the other four matches (re-dump with `geometry_json`); use it to
    flag tdproj clip-parity errors and as a serve-side input to identity / side changes.
6. Widen the eval set beyond one player / venue.
