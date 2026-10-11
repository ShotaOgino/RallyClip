# REPO_MAP — RallyClip (worktree: RallyClip-perf; branch: see docs/PROGRESS.md)

Built at commit `e1c32a7` (2026-07-19). Staleness check:
`git diff --stat e1c32a7.. -- src tests scripts configs .github pyproject.toml`
If that shows changes not reflected here, update this map in the same commit as your work.

Cross-chat handoff (container): `../CHAT_HANDOFF.md`.

## Container context (one level up)

This checkout lives in `rallyclip_container/` (the container dir is **not** a git
repo). Sibling layout:

| Sibling | What | Touch it? |
|---|---|---|
| **This dir (`RallyClip-perf/`)** | Linked worktree of the same `.git` as `../RallyClip/`. Active feature + training work. Branch usually tracks `origin/main` (see PROGRESS). | **Yes — default place to code.** |
| `../RallyClip/` | Primary clone (shares `.git`), often parked on `docs`. Holds local `data/`, `models/`, weights. | No feature work; read data/weights only. |
| `../YOLO-ONNX/` | Separate repo: YOLO26 C++ ORT lab + Ultralytics/ONNX parity (`scripts/parity_v8n_960.py`). Desktop v8@960 ONNX came from here. | Export + parity; then copy contract into this repo's `models/pose/`. |
| `../rallyclip-prod/` | **Different GitHub repo** (`rallyclip-prod`): Modal/Vercel cloud experiments. Not the desktop app. | Don't invest. |
| `../new_data/` | new_data sessions + deterministic labels (`labels/`, `SINGLES_MATCHES_LABELING.md`). | Data + labeling scripts only. |
| `../TODO.md` | Container data/training backlog (incl. YOLO26 §2b). | Update when backlog moves. |

## Top-level layout (this repo)

| Path | What |
|---|---|
| `src/` | The Python package (8 subpackages, see seams below). `package-dir = src`. |
| `tests/` | Pytest suites + `tests/fixtures/` (court goldens ~11MB, golden CLI clip, quality GT) + `tests/helpers/`. |
| `scripts/` | Training/data tooling + `scripts/export_heatmap_model.py` (TCN ONNX export) + `scripts/fetch_artifact.py` / `scripts/release/pack_artifact.sh` (GitHub artifact zip) + `scripts/perf/` benchmarks + `scripts/release/` (sign, DMG, notarize, CI cert import). Not shipped. |
| `configs/` | Training YAMLs (`configs/train/base.yaml`, `holdout.yaml`, `configs/extract/*`). |
| `configs/decode/` | Per-run decode profiles for `rallyclip --config` (`jp-junior.toml`: threshold 0.7 / smooth_sigma_frames 1.5 / merge_gap_sec 0.0 / hybrid_min_duration_sec 2.0 / end_pad_sec 1.6, re-measured 2026-10-12 on the v0.6.0 artifact + upscaled far crop, all five beta-tennis matches). |
| `scripts/eval/` | Real-match evaluation (not shipped): `dump_heatmap_tracks.py` (pose->TCN tracks + near/far players + raw detections to .npz once; v2: tees `DataPreprocessor.iter_slot_players`, v1: `FeatureEngineer.iter_build_features`) + `score_heatmap_decode.py` (offline decode sweep vs beta-tennis `.tdproj`/CSV GT; base = the dump's manifest params); `analyze_players.py` (Phase 1), `score_shots.py` / `export_shot_clips.py` (Phase 2), `reconstruct_score.py` (Phase 4). Dumps in gitignored `logs/eval/`. |
| `models/` | Tracked **manifests + `rallyclip_v0.5.0/SHA256SUMS`**. ONNX is GitHub Release `artifact-rallyclip_v0.5.0`; `python scripts/fetch_artifact.py` unpacks into `models/rallyclip_v0.5.0/` (gitignored). `RallyClip.spec` still packs that folder into the DMG. |
| `docs/` | This harness + plans (Tier 3) + `docs/perf/` (streaming-perf loop journal) + `docs/training.md` + `docs/artifact-registry-plan.md` (ONNX off git). |
| `packaging/`, `RallyClip.spec` | PyInstaller/macOS packaging. |
| `.github/workflows/` | `ci.yml` (3 OS × unit/e2e), `release.yml` (tag `v*` → signed/notarized DMG). |
| `train.py`, `visualize.py` | Training-pipeline entry points (developer workflow, not runtime). |
| `config.toml` | Local runtime config for CLI runs. |
| `build/`, `logs/`, `src/rallyclip.egg-info/`, `__pycache__` | SKIP: generated, gitignored. |
| `README.md`, `PROJECT_SUMMARY.md`, `REFACTOR.md`, `TODO.md` | Tier 4: human-oriented background. TODO.md is the long-horizon idea pile, not current state (that's docs/PROGRESS.md). |

## By task — read Y, nothing else

- **Analysis pipeline behavior / segments wrong** → `src/rallyclip_engine/models.py` (AnalysisModel ABC + pipeline impls), `src/rallyclip_core/pipelines.py`, `models/rallyclip_v0.5.0/manifest.json`, then `src/rallyclip_core/intervals.py`.
- **Heatmap TCN ship / decode** → `scripts/export_heatmap_model.py`, `src/training/models/heatmap_tcn.py`, `models/rallyclip_v0.5.0/manifest.json`, `src/infer/heatmap_decode.py`, `src/rallyclip_engine/models.py` (`HeatmapHybridModel`).
- **Decode params vs real labelled matches (merged points, end offset)** → `scripts/eval/score_heatmap_decode.py`, `src/infer/heatmap_decode.py` (`HeatmapDecodeConfig`, `end_pad_sec` / `_pad_ends`), the artifact manifest's `postprocess.params` (`models/rallyclip_v0.6.0/manifest.json`). Per-run overrides (`threshold`, `merge_gap_sec`, `smooth_sigma_frames`, `hybrid_min_duration_sec`, `end_pad_sec`): keys/validation `src/rallyclip_core/contracts.py` (`DECODE_OVERRIDE_KEYS`, `coerce_decode_overrides`) -> `RunRequest.decode_params`; CLI flags / config `[run]` in `src/cli/main.py::_resolve_decode_overrides`; `/api/upload-and-start` job config via `src/gui/app.py::_CLIENT_KEYS`; merged over the manifest in `src/rallyclip_engine/models.py::HeatmapHybridModel._decode_config`. Tests: `tests/test_heatmap_decode.py` (end pad), `tests/test_heatmap_runtime.py` (merge), `tests/test_cli_config_contract.py` (precedence).
- **CLI** → `src/cli/main.py` (single file; `main()` at :337), `tests/test_cli_config_contract.py`, `tests/test_cli_json_output.py`.
- **GUI/backend HTTP API** → `src/gui/app.py` (Flask, module-global config — see seams), `src/rallyclip_api/services.py`, `tests/test_api_route_contracts.py`, `tests/test_api_job_lifecycle.py`. Decode presets: `src/gui/app.py::DECODE_PRESETS` (data; jp-junior mirrors `configs/decode/jp-junior.toml`, test-enforced) -> `GET /api/config/defaults` `decode_presets` [{id,label,description,params}] -> `POST /api/upload-and-start` config `decode_preset` (resolved in `_normalize_config`; explicit per-key values win; unknown id 400) -> frontend "Decode preset" select (`script.js::populateDecodePresetSelect`, localStorage `rallyclip_decode_preset`, hidden if the key is absent). Upload cap: `MAX_UPLOAD_BYTES` (32 GB, `RALLYCLIP_MAX_UPLOAD_GB`), served as `max_upload_bytes`, JSON 413.
- **In-app update** → `src/gui/update_release.py` (newest published `v*` release, skip `artifact-rallyclip_*`), `src/gui/app.py` (`/api/update/status|open|download|cancel`), `src/gui/frontend/script.js` (`handleUpdateClick`), `tests/test_update_release.py`.
- **Desktop shell / playback** → `src/gui/desktop.py` (pywebview shell), `src/rallyclip_core/playback.py`, `tests/test_native_playback.py` (server-side proxy/descriptor only; the Qt native player was deleted 2026-07-04). Viewer streams `/api/library/<id>/source` directly (Range/206); WebM preview windows are the codec fallback.
- **Segment edit mode** → `src/gui/app.py` (`PUT .../segments`, `DELETE .../segments/edits`), `src/rallyclip_core/library.py` (`resolve_segments`: `segments_edited.csv` wins, original never written), `src/rallyclip_core/intervals.py` (`write_point_intervals`), frontend `src/gui/frontend/script.js` (edit-mode section), e2e `tests/test_gui_playwright.py::test_ui_segment_edit_mode_*`.
- **Pose extraction (ONNX runtime)** → `src/extraction/yolo_onnx_runner.py` + `src/extraction/pose_extractor.py` (dispatch on weights extension), `tests/test_yolo_onnx_runner.py`, `docs/onnx-pose-parity-plan.md`, `../YOLO-ONNX/scripts/parity_v8n_960.py`.
- **Court detection** → `src/preprocessing/court_detector_impl.py`, `tests/test_court_detection_deterministic.py`, `tests/helpers/court_fixtures.py`; regen fixtures with `scripts/court_fixtures_gen.py`.
- **v0.6.0 net-line slots / far-court crop pass** → `src/extraction/crop_pass.py` (fixed window + geometry-driven upscaled `adaptive_crop`), `src/rallyclip_engine/models.py` (`_far_crop_kwargs`, `_court_geometry`), `src/preprocessing/court_lines.py` (court-line model -> `points_px`/net line/homography), `src/preprocessing/netline_slots.py` (`CourtGeometry`, `pick_far` tiers, `StaticDetections`), `src/preprocessing/data_preprocessor.py` (`iter_slot_players`, `far_pick_stats`), tests `tests/test_crop_pass.py`, `tests/test_netline_slots.py`, `tests/test_runtime_v2_path.py`.
- **Features/preprocessing (runtime)** → `src/features/feature_engineer.py`, `src/preprocessing/data_preprocessor.py`, contract tests `tests/test_runtime_*_contract.py`.
- **Training pipeline** → `docs/training.md`, `src/training/pipeline.py`, `src/training/normalize/`, `configs/train/base.yaml` + `holdout.yaml`, `train.py`, `scripts/convert_new_data_labels.py`. Pose extract still `src/training/pose/yolo_hdf5.py` (moving to YOLO26 ONNX — `../TODO.md` §2b).
- **new_data labels (container)** → `../new_data/SINGLES_MATCHES_LABELING.md`, `../new_data/labels/`, `../new_data/generate_labels.py`.
- **Packaging/release** → `RallyClip.spec`, `packaging/macos/`, `scripts/release/`, `scripts/fetch_artifact.py`, `src/runtime/artifact.py`, `.github/workflows/release.yml`, `docs/cli-in-release-binary-plan.md`, `docs/artifact-registry-plan.md`.
- **Player identity / server (roadmap Phase 1)** → `src/analysis/players.py` (numpy only; `src/analysis/` = match analysis above point detection: players, shots, score): `detect_server` (methods `raise` both sides / `far_reach` = far wrist peak via `wrist_reach` when the far player is observed / `elimination` when it is not; rows carry `server_method`), `PlayerTracks` (`from_npz` / `from_arrays` on the dump keys `{near,far}_{box,kp,kpconf,present}` + `ref_size` + optional `raw_*`, `from_records` in-engine, `save_npz`; the shipped far record wins per frame, `pick_far_player` on `raw_*` fills its gaps), `detect_server`, `track_identities`, `serve_consistency`, `analyze_points` -> per-point rows, `points_document` (`<name>_points.json`). Engine opt-in: `RunRequest.save_player_tracks` -> `rallyclip_engine/models.py` (`build_feature_stream(player_sink=)`: `_tee_slot_players` on v2 / `_tee_player_records` on v1 + `_tee_pose_frames` for raw; `HeatmapHybridModel._write_player_outputs`); CLI `--players`; GUI jobs always on -> library item `points.json` + `players.npz`, `GET /api/library/<id>/points` (404 if absent). Eval: `scripts/eval/analyze_players.py`. Tests: `tests/test_players.py`, `tests/test_runtime_v2_path.py::test_v2_player_tee_*`, `tests/test_api_route_contracts.py::test_library_points_*`, `tests/test_cli_config_contract.py::test_players_flag_reaches_run_request`.
- **Serve side deuce/ad (roadmap Phase 1)** → `src/analysis/serve_side.py` (numpy only): visual = feet (ankles, else box bottom) through the court homography (`geometry` dict `homography_px_to_m` / `CourtGeometry.px_to_m`) in a -1.0/+0.5 s window around the server's first raise (`detect_server` evidence); near x>0 or far x<0 = deuce for server AND receiver (`side_for_x`); score = parity of the point index in its game (`score_serve_sides`, `points_in_game` merges a clip < `same_point_gap_s` after the previous = second serve); `combine` / `cross_check`; knobs `ServeSideConfig`. Wired via `analyze_points(geometry=, game_ids=)` (rows gain `ROW_KEYS`; default geometry `PlayerTracks.geometry` = npz `geometry_json`; default game ids = server-id segmentation from `analysis.score`), `points_document` `serve_side_check`. Engine: `build_feature_stream` sets `PlayerRecordSink.geometry` (v2 net-line runs) -> `_write_player_outputs`. Dumps: `scripts/eval/dump_heatmap_tracks.py` wraps `court_lines.compute_court_geometry` -> `geometry_json`. Eval table: `scripts/eval/analyze_players.py --geometry`. Tests: `tests/test_serve_side.py`, `tests/test_players.py::test_engine_player_outputs_carry_serve_side_and_geometry`, `::test_build_feature_stream_hands_netline_geometry_to_the_sink`.
- **Shot events / stroke type (roadmap Phase 2)** → `src/analysis/shots.py` (numpy only, not in the runtime path): `arm_speeds`, `detect_swings` -> `SwingEvent`, `classify_stroke`, `analyze_point_shots` (both sides + alternation prior), knobs in `ShotParams` (body scales / box heights, so reference-size independent). Report: `scripts/eval/score_shots.py` (`--server phase1` = per-point `detect_server`); labelling clips: `scripts/eval/export_shot_clips.py`. Tests: `tests/test_shots.py` (fixture `tests/fixtures/golden_cli/clip.tracks.npz` = this branch's v0.6.0 dump of the golden clip).
- **Score reconstruction (roadmap Phase 4)** → `src/analysis/score.py` (stdlib only): `MatchFormat`, immutable `ScoreState` + `advance`, `reconstruct` / `reconstruct_games`, `infer_first_server_and_check` (Phase 1 hook), beta-tennis winner-CSV readers. CLI: `scripts/eval/reconstruct_score.py` (`--winners-csv` / `--servers-json`). Tests: `tests/test_score.py`.
- **Service roadmap (points -> players -> shots -> ball -> score)** → `docs/service-roadmap-plan.md`.
- **Perf** → `docs/perf/PLAN.md` + `docs/perf/JOURNAL.md`, `scripts/perf/bench_*.py`, baselines in `docs/perf/baseline/`.

## Key seams & entry points

- `src/rallyclip_core/` — pure contracts (`RunRequest`→`RunResult`, ProgressEvent, SavedMatchStore, playback scheduling). **Rule: no heavy imports here** (torch/ultralytics/av/cv2/numpy); `tests/test_gui_startup_imports.py` enforces it.
- `src/rallyclip_engine/runtime.py:27` — `RuntimeDeps` default binds `PoseExtractor`; the dependency-injection seam (the ONNX pose swap was validated through it before landing as the default).
- `src/rallyclip_core/pipelines.py:14` — `pipeline_id_from_manifest_values`: the model artifact's manifest (not code) selects the pipeline. Shipped default: `frame_startend_heatmap` (TCN 3-head hybrid decode); `frame_probability_hysteresis` remains on `models/rallyclip_v0.4.0/`; `start_end_attention_voting` is a stub.
- `models/rallyclip_v0.5.0/manifest.json` — the model contract: imgsz 960, conf 0.25, fps 5, seq_len 100, pipeline `frame_startend_heatmap`. Don't hardcode these in code. `runtime.defaults.DEFAULT_ARTIFACT_DIR` is the packaging/runtime pointer.
- `src/rallyclip_api/services.py:10` — `RallyClipServices` facade; CLI and Flask GUI are both thin clients of it. Desktop app = pywebview (WKWebView/WebView2) → local Flask; **all behavior is `/api/*` HTTP**, no private channel.
- `src/extraction/pose_extractor.py` `_flush_batch` — the predict surface (4 arrays per result) that `yolo_onnx_runner.YOLO` replicates; `.pt` weights still route to lazily-imported ultralytics (`[train]` extra).
- `src/gui/app.py:86-226` — config is module globals (`PREFERENCES_PATH`, `JOBS_DIR`, …). E2E harness `tests/helpers/e2e_backend.py` must redirect ALL of them; config-object refactor planned (`docs/runtime-config-refactor-plan.md`).
- Video decode is **PyAV everywhere** in runtime; OpenCV is image-ops only (court detector classical CV + the parity-critical letterbox resize; its bundled FFmpeg is dead weight — removable only via a custom videoio-less wheel).

## Deliberately NOT worth reading

- `build/` — stale PyInstaller output (includes a full copy of src as `build/lib/`; grep hits there are noise).
- `models/*/model.onnx`, weights, `tests/fixtures/**` binaries — data, not code.
- `docs/perf/baseline/*.json`, `docs/perf/iterations.jsonl` — machine-written bench records.
- `docs/*.icns|svg|png` — icon assets.
- Root `README.md`/`PROJECT_SUMMARY.md`/`REFACTOR.md` — background prose; superseded for orientation by this map.
