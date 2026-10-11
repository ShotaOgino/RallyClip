# DECISIONS — append-only why-log

Format per entry: date — what / why / rejected alternative. Never rewrite old entries.

## 2026-07-03 — Doc harness created in this worktree (RallyClip-perf)

- **What:** AGENTS.md + features.json + docs/{REPO_MAP,PROGRESS,DECISIONS,ENVIRONMENT,testing}.md
  live at the root of this repo, committed on `refactor/runtime-api-engine`.
- **Why:** RallyClip-perf is where active work happens (latest commits, clean tree);
  the container dir (`rallyclip_container/`) is not a git repo, so the harness must
  live inside a repo to be versioned. Container-level context lives in
  docs/REPO_MAP.md's sibling table (no container-level doc file).
- **Rejected:** harness at container level (unversionable without a new git repo
  wrapping four checkouts — nested-repo mess); harness in `RallyClip/` (parked on
  `docs`, not where work happens; shares `.git` anyway).
- Structural choices made while scanning: replaced the old generic root `AGENTS.md`
  (repo-guidelines boilerplate, partly stale — e.g. claimed yolov8s default) and
  folded root `ENVIRONMENT.md` into `docs/ENVIRONMENT.md` so Tier-1 docs have one
  home. `TODO.md` kept as Tier-4 idea pile; current state lives in PROGRESS.md.
  Lint: no config exists and CI doesn't lint → documented scoped-lint convention
  (23 legacy ruff errors recorded in testing.md) instead of adding a lint config
  (that would be a behavior change beyond a docs harness).
- Test interpreter standardized on `../RallyClip/.venv-train/bin/python3` because it
  was verified in-session (212 passed); conda `tennis_env` kept as an alternative.

## 2026-07-04 — ONNX pose runner is the production path (PR #26)

- **What:** manifest points `feature_pipeline.yolo_model` at a bundled dynamic-axes
  960 ONNX; PoseExtractor and CourtDetector dispatch on the weights extension;
  torch/ultralytics moved to the `[train]` extra.
- **Why:** byte-equal segments on 17/17 sweep samples + golden clip; ~1.45× faster,
  ~40% less RSS; removes torch from install and bundle.
- **Rejected:** YOLO26 end-to-end export (NMS in graph — different output contract,
  raises typed error instead of silent mis-decode); keeping ultralytics as runtime
  fallback (would keep torch in the dependency closure).

## 2026-07-04 — System webview shell replaces QtWebEngine (PR #27)

- **What:** pywebview (WKWebView/WebView2) window over the unchanged Flask backend;
  deleted gui/native_player.py and the QWebChannel bridge.
- **Why:** the native Qt player existed only because Chromium-in-QtWebEngine ships no
  H.264/HEVC; the system webview plays both, and the frontend already had a complete
  HTML5 fallback. Bundle 765→266MB; ~1600 lines deleted; frontend unchanged.
- **Rejected:** pruning unused Qt modules only (~60-100MB, keeps Chromium + native
  player complexity); Tauri/Electron-style rewrite (new stack for no extra benefit).

## 2026-07-05 — CoreML EP + static-shape export wins the Apple-silicon spike (no MLX rewrite)

- **What:** benchmarked the shipped pose ONNX on this M-series Mac via onnxruntime
  execution providers (real frames from a saved match, 40-frame batches). Shipping
  config (dynamic-axes ONNX, CPU EP, rect 544x960): 15.6 fps. CoreML EP on the
  dynamic model: only ~1.2x — the Neural Engine rejects unbounded dims (E5RT
  "unbounded dimension"), so 110/380 nodes stay on CPU with 8+ partition round-trips.
  Re-exporting the same checkpoint with static shapes flips it: static rect
  544x960 + CoreML EP (MLProgram, MLComputeUnits=ALL) = **120.4 fps — ~7.7x the
  15.6 fps shipping path** (8.05x vs the same static model on CPU, 15.0 fps);
  max abs divergence on confident detections 1.2e-4. Static exports were made
  from models/yolov8n-pose.pt, the checkpoint the bundled dynamic ONNX came
  from; production must golden-verify the static export against the bundled
  ONNX before swapping. LSTM head: CoreML is *slower*
  (0.59s vs 0.48s/200 runs) — keep it on CPU. Scripts + JSON results committed in
  docs/perf/coreml-spike/.
- **Why it matters:** pose extraction is the pipeline bottleneck; ~7.7x there without
  new dependencies (CoreMLExecutionProvider ships in stock onnxruntime 1.24.4).
  Productionizing needs: (a) a static 544x960 export added to the model bundle,
  (b) an opt-in provider flag (CPU stays the parity default — 1e-4 divergence
  breaks byte-equal goldens), (c) a fallback for non-16:9 sources (letterbox pads
  to the static shape, as the spike did for square).
- **Rejected:** MLX rewrite (whole new inference stack for less gain than a
  re-export); NeuralNetwork-format CoreML (0.25 abs divergence — actually wrong);
  CPUAndNeuralEngine-only compute units (2x — ANE alone loses to ANE+GPU "ALL");
  accelerating the LSTM (measured slower on CoreML).

## 2026-07-05 — Frozen-app data lives in the OS app-data dir (PR #28)

- **What:** `gui.app._frozen_data_root()` puts packaged-build user data in
  `~/Library/Application Support/RallyClip` (macOS) / `%APPDATA%` (Windows) /
  XDG data home (Linux), with a one-time `shutil.move` migration from the
  v0.1.0 `~/RallyClip` location. Windows is selected via `sys.platform`.
- **Why:** dumping a data dir in `$HOME` violates platform conventions; the
  migration keeps v0.1.0 users' libraries. `sys.platform` (not `os.name`)
  because pathlib picks WindowsPath/PosixPath from `os.name` at instantiation —
  monkeypatching it breaks every `Path()` in tests on Windows.
- **Rejected:** `Path.rename` (EXDEV across filesystems — shutil.move falls
  back to copy+delete); zero-arg lru_cache memoization (leaks state across
  platform-monkeypatching tests for a handful of one-time stats).

## 2026-07-05 — Viewer streams the source file directly (PR #29)

- **What:** `/api/library/<id>/source` serves the saved match with
  `send_file(conditional=True)` (Range/206); the frontend models it as one
  full-length window so the existing source-time scheduler (seeks, point
  skips, timeline) is unchanged. WebM preview windows remain the automatic
  fallback (probe error or 10s timeout; probes are sequence-ticketed so stale
  callbacks are inert).
- **Why:** the stuck-at-first-8s bug: 8s VP8/WebM windows transcode at ~2.5×
  real time (17–22s per window, file-mtime evidence + live WebKit repro), so
  playback stalled at every boundary. The WebM pipeline only ever existed
  because QtWebEngine's Chromium lacked H.264 — the system webview (PR #27)
  decodes it natively.
- **Rejected:** speeding up the transcode (still a transcode; still burns CPU
  and disk); MSE path (permanently dormant, `canUseMsePreview()` false);
  deleting the window pipeline immediately (kept as codec-fallback until
  direct playback is QA-confirmed in the wild).

## 2026-07-05 — Segment edits are a shadow CSV, never the original (PR #31)

- **What:** viewer edit mode writes user point edits to `segments_edited.csv`;
  the model-produced `segments.csv` is never modified. The edited copy wins
  everywhere (`resolve_segments`: playback manifest, /segments, CSV download,
  lazy export — export.mp4 invalidated on edit/reset). Reset deletes the copy;
  it refuses if the original is missing (legacy items). Frontend autosaves are
  serialized and generation-guarded (a stale PUT can't resurrect a reset).
- **Why:** "Reset to original" must always be possible, so the original is a
  read-only contract; a shadow file with precedence is the smallest mechanism
  that gives every consumer the edited times without touching the analysis
  output. All behavior stays `/api/*` HTTP per the architecture invariant.
- **Rejected:** editing segments.csv in place with a backup copy (reversed
  precedence is easier to corrupt — an interrupted write loses the original);
  edits in meta.json (two sources of truth for point times); save-on-Done only
  (drag sessions lose work on crash; autosave matches the iPhone-Photos model).

## 2026-08-24 — Ship the champion TCN as default (v0.5.0); keep classic LSTM as fallback

- **What:** Export `TennisPointHeatmapTCN` (run `20260724_tcn64_cos1e4`) to
  `models/rallyclip_v0.5.0/` with three named logit outputs and
  `pipeline.id=frame_startend_heatmap`. Point CLI/GUI/packaging at that bundle.
  Keep `models/rallyclip_v0.4.0/` in-tree. Bake hybrid decode knobs into the
  ship manifest; dummy hysteresis keys (`sigma`/`low`/`high`/`min_dur_sec`) stay
  in `postprocess.params` so CLI/GUI `_resolve_mutable` / `build_gui_defaults`
  still resolve. GUI jobs resolve pipeline from the artifact unless the client
  explicitly sends `pipeline_id`.
- **Why:** Champion hybrid decode is ~43.9% test acceptable vs classic v0.4.0
  ~30.7% six-bin good. No extra training. Mac/CLI runtime already exists.
- **Rejected:** shipping heatmap LSTM (~31.5% ≈ classic); start/end-only or
  pair-DP as default decode; chasing bit-identical train-wt metrics (min-duration
  is applied at slightly different stages); `rallyclip serve` / Win-Linux freeze
  in the same change.

## 2026-09-07 — Desktop DMG is built/signed/notarized in GitHub Actions

- **What:** `release.yml` on `v*` tags runs tests, `pyinstaller RallyClip.spec`,
  Developer ID codesign (hardened runtime + timestamp), a drag-to-Applications
  DMG, `notarytool` submit/wait, stapler, and a draft GitHub Release asset
  named `RallyClip-<pyproject-version>-macOS-arm64.dmg`. The same scripts run
  locally (`scripts/release/package_macos.sh`). The spec bundles
  `runtime.defaults.DEFAULT_ARTIFACT_DIR` (currently `models/rallyclip_v0.5.0`)
  and sets `CFBundleIdentifier` `com.iroblesrazzaq.rallyclip`.
- **Why:** v0.1–v0.3 DMGs were assembled and notarized by hand; CI only uploaded
  an unsigned `.tar.gz`. Signing belongs in CI so a tag is the release
  artifact, not a leftover file on one Mac. GitHub-hosted `macos-latest`
  runners can codesign/notarize if the Developer ID `.p12` and App Store
  Connect API key are injected as secrets (no self-hosted Mac required).
- **Rejected:** keeping the ad-hoc `pyinstaller --hidden-import ...` CLI in
  `release.yml` (it had drifted from `RallyClip.spec`); `rcodesign` on Linux
  (extra toolchain, still need Apple notary); `apple-actions/import-codesign-certs`
  (another pin; a 40-line import script is enough); notarizing a zip of the
  `.app` then wrapping a DMG (one DMG submission matches the proven v0.1.0
  manual path); failing open on missing secrets for tags (that would re-ship
  unsigned builds). `workflow_dispatch` still wraps an unsigned DMG when
  secrets are absent.

## 2026-09-07 — PR review is Greptile only; Bugbot stays off

- **What:** Agents must not invoke Cursor Bugbot. PR review is Greptile
  (`@greptileai`, iterate to 5/5). Disable Bugbot for this repo in the Cursor
  dashboard so it does not spend tokens on every PR push.
- **Why:** Greptile is already the repo reviewer and is free here; Bugbot is a
  second paid review on the same diffs.
- **Rejected:** leaving Bugbot on "only when mentioned" (still easy to trip by
  commenting `cursor review`); uninstalling the whole Cursor GitHub App (that
  also breaks Cloud Agents / PR comments we still use).

## 2026-09-07 — ONNX off git; artifact zip; DMG still combines at build

- **What:** Plan (`docs/artifact-registry-plan.md`): git tracks manifests +
  `DEFAULT_ARTIFACT_DIR` pointer; weights live on GitHub Release
  `artifact-rallyclip_v0.5.0`. CI fetches the zip before PyInstaller. The
  notarized `.app` still contains `models/rallyclip_v0.5.0/` and does not
  download on launch. From-source clones run `scripts/fetch_artifact.py`.
- **Why:** Even `git clone --depth 1` currently pulls every ONNX on `main`
  (triplicated pose weights + old LSTM trees). Clone should be code; the DMG
  stays a single offline install.
- **Rejected:** Git LFS/DVC (extra quota/tooling); first-launch download inside
  the Mac app; putting weights in Application Support next to the library;
  rewriting git history in the same change.

## 2026-09-08 — Release probe boots the real GUI, Flask first

- **What:** Packaged release CI launches `"$BIN"` (no `--backend-only`). Flask
  starts and `/api/health` can succeed before pywebview/WKWebView is imported.
  After `create_window`, the binary prints `RallyClip desktop shell ready`; the
  probe requires that line plus health. `--backend-only` stays as a debug flag.
- **Why:** v0.5.0 release failed because importing WKWebView on the GitHub
  Mac runner stalled before Flask came up (30s). Switching the probe to
  `--backend-only` would not catch a broken pywebview bundle (Greptile P2;
  agreed).
- **Rejected:** Keep `--backend-only` as the release probe (misses desktop-shell
  regressions); importing webview before starting Flask (original hang).

## 2026-09-08 — Ready marker is `window.events.loaded`

- **What:** `RallyClip desktop shell ready` prints from `window.events.loaded`
  (DOM ready in WKWebView). Release probe also `kill -0`s the process after
  seeing the line. `create_window()` only allocates the Python Window object.
- **Why:** Greptile P1: a marker before `webview.start()` can pass while native
  GUI init still fails; CI then kills the process.
- **Rejected:** Printing after `create_window()` / before `webview.start()`.

## 2026-09-09 — Assisted DMG update in the packaged app (v0.5.1)

- **What:** Frozen Mac app downloads the Latest `v*` arm64 DMG into
  `~/Downloads`, checks SHA-256, and opens it. Localhost GUI opens that
  release page. App version is 0.5.1; artifact remains `rallyclip_v0.5.0`.
- **Why:** v0.5.0 only linked the Releases list. Users still replace the app
  in Applications themselves (running binary cannot overwrite itself).
- **Rejected:** Sparkle / in-app swap of `/Applications/RallyClip.app`;
  treating the ONNX zip as an app update; unifying web vs app library paths.

## 2026-09-09 — Update check skips artifact releases

- **What:** Status/download list GitHub `/releases` and pick the newest
  published `v*` tag. Checksum-verify a staging file before replacing any
  existing DMG in `~/Downloads`. Frozen update button becomes Cancel.
- **Why:** `/releases/latest` can be `artifact-rallyclip_*`; a failed repeat
  download must not delete a good installer; a 300s-per-file transfer must
  stay cancellable.
- **Rejected:** Trusting `/releases/latest` as the app channel; Sparkle;
  byte-progress UI (the server downloads, the browser waits on one POST).

## 2026-09-09 — Cancel stops the server-side DMG download

- **What:** `POST /api/update/cancel` sets an event the download loop checks
  between chunks. Sidecar checksums must name the selected DMG (no last-hash
  fallback). Staging files use a unique suffix so a retry cannot share paths.
- **Why:** Aborting the browser fetch left Flask downloading and `open`ing the
  DMG. An unnamed sidecar hash could verify the wrong bytes.
- **Rejected:** Client-only AbortController as the cancel mechanism.

## 2026-07-19 — Training pose switches to YOLO26n ONNX; desktop stays v8 until retrain

- **What:** New training extracts will use **YOLO26n-pose ONNX** under a versioned
  contract (`models/pose/yolo26n/` + shared ORT runner with e2e `[1,300,57]`
  decode). Desktop v0.3.1 remains on **YOLOv8n** ONNX@960 (+ CoreML static
  sibling). Export/parity lab stays in sibling `YOLO-ONNX/`; bundled contract
  and training wire live in this repo. Re-export YOLO26 at training imgsz
  (960 rect / static 544×960) — do not use the experimental 640×640 square
  export for corpus work. Fingerprint pose HDF5 with model sha so v8/v26
  caches never mix.
- **Why:** YOLO26n is cheaper FLOPs for labeling/extract even if accuracy is
  similar; training and desktop were already misaligned (train Ultralytics
  `.pt` @ large imgsz vs desktop ONNX@960). A named ONNX contract is the path
  to one extract path for train (and later desktop after retrain).
- **Rejected:** Keep training on Ultralytics `.pt` forever (diverges from
  runtime, heavier); switch desktop to YOLO26 before LSTM retrain (feature
  distribution shift → silent quality drop); MLX rewrite (already rejected
  2026-07-05 — CoreML EP is the Apple path).

## 2026-07-19 — Normalize-to-720p@5fps before training YOLO (robust pipeline)

- **What:** Training pipeline gains an explicit normalize stage (1280×720 @ 5
  fps) before court/pose extract; `ignore_before_s` from new_data labels becomes
  target `-1` (dropped from loss); holdout frozen in `configs/train/holdout.yaml`;
  `train.head: classic | e2e_seg`. Implementation is in this worktree
  (uncommitted as of this entry); details in `../CHAT_HANDOFF.md` and
  `../TODO.md`.
- **Why:** Court detector is pixel-absolute (tuned at 720p); 5 fps is known
  sufficient; new_data warm-up must not be treated as negative; fixed ~5 h test set
  preferred over 5-fold for sweep cost.
- **Rejected:** Making court detector resolution-relative before retrain
  (larger change, riskier for first corpus); treating warm-up as negative
  class; leaving extract on full-fps YOLO then subsample at dataset build.

## 2026-07-19 (session 2)

- **YOLO26 training extraction runs CPU-only ORT for now.** CoreML deferred: the
  e2e export carries NMS in-graph, which risks bad CoreML EP partitioning; not
  worth engineering before the corpus is extracted once. Revisit if extract wall
  time hurts.
- **Pose models are manifest-defined backends** (models/pose/<name>/manifest.json:
  head family, input/letterbox contract, COCO-17 keypoint order, model sha256).
  Dataset identity tag = name@sha8; execution provider is provenance, not identity.
- **Unified corpus data root = container-level ../training_data/**, decoupled from
  both repos. Old RallyClip/data stays as-is for the v0.3.1 lineage.
- **c31e6888 excluded from corpus** (video present, zero export JSONs).
- **Flip augmentation deferred** — regenerate from normalized 720p@5fps videos
  later instead of carrying old flip files forward.

## 2026-07-19 (session 2, later) — YOLO26 → YOLOv8 reversal

- **Pose model = YOLOv8n everywhere (training extract, desktop, iOS). YOLO26 ditched.**
  Rationale: deployment target is Apple/CoreML, so ANE speed beats FLOPs. Measured
  (M2 Pro): v8-static-CoreML 90 fps + sub-pixel parity vs CPU (≤0.9 px); v26-CoreML
  41 fps with 30–180 px decode errors (e2e head fp16 candidate-selection flips —
  inherent to the head, would recur in native iOS CoreML); v26-CPU ~10–12 fps.
  Static-vs-dynamic v26 exports were bit-identical on CPU, isolating the EP as the
  cause. Accuracy delta v8n↔v26n judged marginal for 2-players-at-720p.
- **Corpus extraction runs provider=coreml** (static 544×960 sibling): ~1.5 h vs
  ~10–27 h CPU, and features come from the same runtime the app executes
  (train/serve consistency). EP stays provenance, not identity; parity gate in
  tests/test_pose_backend.py. CoreML is hard-refused for yolo26-e2e head family.
- YOLO26 bundle deleted from models/pose/; exports parked in ../YOLO-ONNX/exports/;
  e2e decoder + dispatch kept in the runner (cheap, tested, documents the contract).

## 2026-07-19 (session 2, data layout)

- **Provenance-nested data layout in ../training_data/**: `sources/` is the only
  physical home for video bytes (new_data session bundles moved whole; legacy_youtube
  for the 11 old originals + annotation provenance copies). Pipeline consumes
  flat interface layers only: `source_videos/` symlinks + flat `annotations/`
  JSONs (golden — edit sources and re-materialize, never the flat dir).
- **Normalize contract is a path tag**: `videos/norm=1280x720@5fps/` and
  `pose_data/norm=…/…` (paths.py norm_tag(); default threaded through
  raw_videos_dir/pose_*_dir so callers are unchanged). Future 1080p = sibling
  tree + court-detector retune, not an overwrite.
- **RallyClip/data deleted; RallyClip/raw_video dups deleted** (md5-verified vs
  canonical first; old annotations+runs archived in
  sources/legacy_youtube/legacy_rallyclip_data_archive.tar.gz). Old-model
  benchmarking uses the bundled models/rallyclip_v0.3.1 artifact.

## 2026-07-20 (session 3 — court fix, loader fix, court-vs-no-court results)

- **Court detection was silently failing corpus-wide** and nobody noticed: base.yaml
  points `yolo.model` at a manifest.json, `court.model_path: null` falls back to it,
  and `_load_yolo` handed the manifest to ultralytics `YOLO()` → "not a supported
  model format", caught → default/no mask. All 29→32 videos had been preprocessed
  with NO court filtering. **Fix:** `_load_yolo` now resolves a manifest/dir via
  `load_pose_backend(provider="cpu")` to the sha-verified ONNX on the CPU EP (court
  touches only a few frames; dynamic-CPU is faithful and avoids the static-letterbox
  concern). After fix: 30/32 detect; `abe49d12`, `e9a093a8` (both TRAIN videos) fail
  all anchors → preprocessed without a mask, flagged loudly.
- **Clean-frame extraction is now MIDDLE-anchored, outward-expanding** (user request).
  Anchor at video midpoint (not 60 s), mask occluding players, walk outward in ~10 s
  hops each way repainting still-occluded px from homography-aligned neighbours that
  have nobody over the same spot; stop a side when quad-IoU(neighbour→base) < 0.6
  (camera cut / pan-zoom) or video ends; stop entirely once no occluded px remain.
  `court.target_time: null` = midpoint (threaded Optional through pipeline →
  PreprocessConfig → fingerprint as "middle"); `CourtMaskCache` retries anchors
  0.5/0.4/0.6/0.3/0.7 of duration; preprocess prints a court-health summary.
  Inference path (`data_preprocessor`) made middle-first too.
- **HDF5 dataset loader now loads to RAM once** (`Hdf5SequenceDataset`, 6 GB guard).
  The dataset build writes features gzip-compressed with chunks (228,7,23) spanning
  228 sequences, so per-item DataLoader reads decompressed hugely-overlapping data:
  measured **~470 s/epoch** just in loading (800 random reads = 51.8 s) vs **2.35 s**
  to bulk-load the whole 1 GB array. In-memory index → epochs 8 min → ~30 s, bit-
  identical (verified). Root cause is the chunking; the in-mem load is the pragmatic
  fix (rechunking the build to (1,100,362) would fix the lazy path too — deferred).
- **Court filtering helps new_data, regresses legacy** (six-bin acceptable, each val-swept):
  classic no-court 31.1% new_data / 49.0% legacy → classic+court **37.1% new_data** / 31.3% legacy.
  Legacy loss is FN-driven. CONFOUNDED (court filtering AND +3 videos/retrain changed
  together); user notes the masks were legacy-tuned so the drop is likely training-mix
  + the middle-out frame shift, not mask over-aggression (Aditi masks more of frame
  than 9/5/15 yet scores fine). **Left unresolved per user** — new_data is the target domain,
  so this is a net win for the goal. Not committed to fixing the masks.
- **e2e head underperforms classic and overfits early; longer patience doesn't help.**
  e2e+court patience-5 (best ep2) 22.9% new_data / 34.0% legacy; patience-10 (best ep9,
  user-requested to avoid "cutting off too soon") 19.3% / 10.2% — WORSE. Val loss
  bottoms ~ep5 then climbs; the two runs differ mostly by MPS nondeterminism
  (best-val 13.7% vs 9.1%), which swamps the patience effect. Conclusion: the
  boundary-REGRESSION formulation is the ceiling (bad_seg still ~50%), not epoch
  count → motivates the Gaussian startness/endness heatmap head (TODO §3c).
- **Laptop sleep suspends training** (not kills): a run "died" 3× because macOS slept
  and suspended the python process; on wake, restarts stacked into 4 concurrent
  train.py clobbering the same run dir. Fix: wrap training in `caffeinate -ims`.

## 2026-07-20 (session 4) — Boundary-heatmap head (e2e_heatmap) beats classic+court on both domains

- **What:** added a third training head `train.head: e2e_heatmap` — twin per-frame
  startness/endness Gaussian heatmaps (BSN family) alongside the retained pointness
  head. New files only (no edits to seg_lstm/seg_loss/seg_loop/seg_evaluator):
  `models/heatmap_lstm.py` (TennisPointHeatmapLSTM, same BiLSTM backbone, 3 logits),
  `train/heatmap_loss.py` (E2EHeatmapLoss; soft Gaussian target from binary labels
  via `boundary_markers`+`_dist_to_nearest`, nearest-boundary == union-of-Gaussians),
  `eval/heatmap_evaluator.py`, `train/heatmap_loop.py`. Additive branches in
  pipeline.py `_run_train`/`_run_eval` + a knobs block in base.yaml. Reused
  `gt_segments_from_targets`, `compute_six_bin`, `_default_pos_weight`, the CPU-eval
  and verified-checkpoint patterns.
- **Result (val-swept, same 3 new_data + 3 legacy subsets as prior benches):**
  heatmap+court = **42.2% new_data / 46.3% legacy** — new best on new_data (vs classic+court
  37.1%) AND recovers most of the legacy regression (vs 31.3%). Only model strong on
  both domains at once. FN low (new_data 1.4%, legacy 13.9%) — recall preserved.
- **Why these choices (from a critical review of the first plan):**
  - **Hybrid decode, not pure peak-pair.** Pointness runs define segments (one per
    detected point → recall can't be lost to a missed boundary peak); start/end
    heatmaps only *refine* each edge via soft-argmax. The sweep picked hybrid over
    peakpair, confirming it. Pure peak-pick+greedy-pair was too fragile (a point
    needs both peaks AND a valid pairing — three conjunctive failure points, exactly
    the FN problem that sank e2e_seg). peakpair kept selectable for comparison.
  - **Balanced-BCE on the soft Gaussian as default**, CenterNet penalty-reduced
    focal as opt-in (`heatmap_loss: bce|focal`). BCE is simpler/harder to misconfig;
    focal has more knobs. Both implemented + tested.
  - Peak-NMS + final overlap-merge in the decoder (the first plan omitted both →
    would emit duplicate/overlapping phantom segments from jittered adjacent peaks).
- **Rejected / deferred:** soft-argmax *time* loss term (model is trained on heatmap
  SHAPE only, never directly on boundary-time error — obvious next lever for the
  still-~29% bad_seg); σ / focal hyperparameter sweep; production src/infer/ path
  (e2e_seg lacks it too). serve-convention label fix (TODO §3b-ii) still poisons
  boundary supervision for THIS head too — worth doing before trusting exact numbers.
- **Caveat:** single MPS-nondeterministic run, best epoch = 2 (overfits early like
  e2e_seg; patience-10 again burned 10 idle epochs). Ordering is clear + large but
  wants a couple of seeds before the exact numbers are final. Run:
  `runs/20260720_court_heatmap`; bench: `benchmarks/bench_court_heatmap.py`.

## 2026-07-21 (session 5) — Optimization-side levers all neutral-or-worse; +4 videos (corpus 37)

Batch of experiments on the leading heatmap head. **Meta-finding: every
optimization-side knob came back neutral or worse — the boundary/quality ceiling is
data/labels/metric, not optimization.** Do not re-run these:

- **Capacity is not a quality lever (3 confirmations).** GRU backbone (−24% params):
  underperforms LSTM (new_data 39.6 vs 42.2, legacy worse), and MPS has no fused GRU kernel
  (10x slower → train GRU on CPU, which is also deterministic). Hidden-size sweep
  (heatmap, matched decode): @128 42.2/48.3, @64 44.4/43.5, @32 41.1/40.1 — new_data flat
  across 85% param reduction, legacy declines gently, bad_seg never improves.
  **@64 is a free deployment win** (best new_data, 64% fewer params) but shrinking doesn't
  fix the early-overfit. Classic hidden-sweep inconclusive (postproc-operating-point
  confounded). GRU/backbone + hidden_size are config-switchable (heatmap_backbone,
  hidden_size); default lstm/128.
- **Soft-argmax time loss: NEGATIVE, shelved (opt-in, default off).** Added
  `heatmap_time_weight` — collapse each boundary's heatmap to a predicted time via
  soft-argmax, penalize squared time error (Gaussian NLL). It works on TRAIN (halved
  train time-error) but HURTS test monotonically: λ=1 new_data 42→36, λ=10 new_data 42→26 (good
  share collapsed 8.7→1.4%). Two reasons: (1) it optimizes the softmax MEAN while the
  hybrid decode reads the PEAK — moving the mean skews the bump and corrupts the peak;
  (2) boundary error is a generalization gap, not a train-fit gap. Kept in code
  (`heatmap_time_weight: 0.0`) for reproducibility; not shipped.
- **LR finder + cosine schedule: built, NEGATIVE for tuning.** New `lr_finder.py`
  (Smith/fastai LR range test: per-batch exp ramp, smoothed TRAIN loss, suggest
  min-loss-LR/10) + cosine scheduler (`lr_schedule: none|cosine`). Finder @128
  suggested 1.45e-3 ≈ current 1e-3 (validated the default). Cosine @128 (peak 1.45e-3):
  new_data 42→37, best ep1 (higher peak overfit faster). @64 finder suggested 6.28e-3 (flat
  min basin) → cosine @64 COLLAPSED (new_data 44→20). **Caveat: min/10 is unreliable when the
  finder curve has a flat minimum — it overshoots; use the steepest-descent elbow.**
  Cosine can't help when the model best-epochs at ep1-3 (LR hasn't decayed yet).
  Infra kept (default lr_schedule=none); not a lever here.
- **+4 new_data videos → corpus 37** (2026-07-21): 46cd→val (23min, shortest), 729→test
  (41min), f5f5+1e9d→train (42/76min, 2 longest). Split by duration (user). Court
  detection FAILED on 729 (angled) + 46cd → 5/37 maskless; 1e9d passed despite angle.
  Dataset `datasets/20260721_v37` (train25/val4/test8). Retrained the two leaders on
  it: heatmap@64 41.1 new_data / 47.6 leg (orig 3+3), classic@128 29.6/21.8; new 729 test
  video heatmap 35.7% vs classic 26.8%. Adding 2 train videos was within-noise neutral
  on the stable subset — data-quality-limited, not quantity-limited. heatmap@64 still
  leads. Runs: `20260721_heatmap64_v37`, `20260721_classic128_v37`.

**Next (not optimization): serve-convention label fix (§3b-ii) + absolute-time metric
(§3b-iii).** All the above dead ends point there.

---

_The entries from here to the port entry were written in the exported v0.5.0 snapshot
(`~/Documents/RallyClip-0.5.0`, not a git repo) and are carried over with the work in
`feat/port-v050-analysis` (append-only: wording kept, v0.6.0 deltas flagged inline and in
the port entry at the end)._

## 2026-10-07 — GUI upload cap 2 GB -> 32 GB, served by the backend

- **What:** `MAX_CONTENT_LENGTH` is now `gui.app.MAX_UPLOAD_BYTES` (default
  32 GB, override `RALLYCLIP_MAX_UPLOAD_GB`). `/api/config/defaults` returns
  `max_upload_bytes`; the frontend reads it for the drop-zone hint and the
  client-side size check instead of its own hardcoded 2 GB. A 413 now returns
  JSON (`{"error": ...}`) like every other API error.
- **Why:** Real full-match phone/camera recordings are 2.5-8 GB (1080p60,
  30-60 min). The 2 GB cap rejected every match in the user's labeled set.
  werkzeug streams multipart uploads to disk, so the cap was never a memory
  guard; it only needs to be a sanity bound.
- **Rejected:** Removing the cap entirely (an accidental drop of a multi-TB
  file would fill the disk); a path-based "open local file" API that skips the
  upload copy (better long-term for the desktop shell, but a larger change
  touching the WKWebView file picker and job dir semantics).

## 2026-10-11 — beta-tennis GT = `.tdproj`; the -1.6 s end offset is a label convention

- **What:** Real-match accuracy is scored against the human-confirmed `.tdproj`
  (clips with `validation != "deleted"`, warm-up before the first clip ignored),
  mirroring `beta-tennis/Swing/rally_service/eval/eval_vs_tdproj.py`, not the
  `_segments.csv`. The CSV's `last_value` carries an 80-frame (1.33 s) end margin
  added by `3_update_csv.py`, so GT ends sit ~1.3-1.6 s after the real point end.
  RallyClip's early ends are therefore not treated as a model defect; a constant
  end pad is the matching fix if that convention is wanted.
- **Why:** Last session's "ends 1.6 s early" would otherwise have pointed at
  retraining. The offset is tight and constant on both matches, and the margin is
  explicit in the labeling pipeline.
- **Rejected:** Retraining with later end labels; scoring against the padded CSV.

## 2026-10-11 — Decode tuning tooling lives in `scripts/eval/`, tracks in `logs/eval/`

- **What:** `scripts/eval/dump_heatmap_tracks.py` saves the TCN per-frame tracks
  once per match; `scripts/eval/score_heatmap_decode.py` re-decodes and scores
  offline. Track dumps go to gitignored `logs/eval/`.
- **Why:** The previous session's scratch evaluator vanished with its scratchpad and
  the re-extraction took hours of wall-clock. A decode sweep must not need pose.
- **Rejected:** Keeping the scripts in the session scratchpad; committing the npz.

## 2026-10-11 — Do not change the shipped `threshold` yet

_(Decided on the v0.5.0 artifact; the v0.6.0 manifest's threshold is also kept.)_

- **What:** `threshold` 0.65-0.7 (vs 0.5) halves fault+second-serve merges and lifts
  point F1 by 0.08-0.11 on the two JJTT matches, but the default stays 0.5.
- **Why:** Two matches, same player/venue; the manifest value is also what the
  golden fixture was generated with. Heatmap params currently have no config/CLI
  override, so the first code change should be an override path, then a wider sweep.
- **Rejected:** Editing `models/rallyclip_v0.5.0/manifest.json` now (breaks golden
  parity silently); lowering the threshold (makes merging worse).

## 2026-10-11 — Heatmap decode overrides via `RunRequest.decode_params`; `end_pad_sec` decode knob

- **What:** `threshold`, `merge_gap_sec`, `smooth_sigma_frames`,
  `hybrid_min_duration_sec`, `end_pad_sec` are overridable per run (CLI flag ->
  config.toml `[run]` -> manifest). The CLI/GUI pass only explicitly-set keys in a
  new optional `RunRequest.decode_params` (default None); `HeatmapHybridModel._decode_config`
  merges them over the manifest `postprocess.params`. One validator
  (`rallyclip_core.contracts.coerce_decode_overrides`) serves CLI (SystemExit) and
  `/api/upload-and-start` (400). New `HeatmapDecodeConfig.end_pad_sec` (default 0)
  extends ends after merge (hybrid: after the duration floor), re-merges
  touch/overlap only, clamps to the last timestamp. Classic pipeline ignores all of it.
- **Why:** Eval wants threshold 0.65-0.7 and a +1.3-1.6 s end pad for beta-tennis
  labels without editing the shipped manifest (golden parity) or forking an
  artifact. Unset keys resolve to nothing (not to the manifest value) so a classic
  artifact without these manifest keys never crashes and default runs stay
  byte-identical (`decode_params=None`).
- **Rejected:** Typed per-knob `RunRequest` fields (every new decode knob would
  touch contracts + CLI + GUI); resolving overrides to manifest values in the CLI via
  `_resolve_mutable` (crashes on classic manifests, which lack heatmap keys);
  re-merging padded segments with `merge_gap_sec` (pad would silently widen the merge
  budget); applying the pad before the duration floor (pad would rescue short
  false positives); negative pads (trimming can invert segments; rejected as out of range).

## 2026-10-11 — jp-junior decode profile: threshold 0.65, hybrid_min_duration_sec 1.0, end_pad_sec 1.6

_(v0.5.0 artifact values; on this branch superseded by the v0.6.0 values in the port entry below.)_

- **What:** For Japanese junior matches labelled with beta-tennis tdproj, the
  recommended per-run overrides are `threshold 0.65`, `hybrid_min_duration_sec 1.0`,
  `end_pad_sec 1.6` (mean point F1 over 1R/2R/3R/SF 0.80 -> 0.885; time-F1 ~0.85 ->
  0.87-0.92). Shipped manifest defaults are unchanged; the profile is applied through
  the new override path (config.toml / CLI / API).
- **Why:** 0.7 scores the same mean but doubles SF's pure false positives (12) by
  splitting between-point tails; 0.6 leaves twice the merges. A 1.0 s post-merge floor
  removes the fragments that appear once the threshold rises. The end pad reproduces
  the labeler's 80-frame margin.
- **Rejected:** Changing `models/rallyclip_v0.5.0/manifest.json` (golden parity, and
  the profile is label-convention specific); tuning `smooth_sigma_frames` /
  `merge_gap_sec` (no measurable effect); retraining before Phase 0 ships.

## 2026-10-11 — GUI decode presets are server-side data, chosen by id

- **What:** `src/gui/app.py::DECODE_PRESETS` defines named decode profiles
  (`default` = no overrides; `jp-junior` = the tuned profile, see the jp-junior entries)
  with a label and one-line description.
  `GET /api/config/defaults` lists them as `decode_presets`; `POST /api/upload-and-start`
  accepts `decode_preset` in its config, resolves it server-side in `_normalize_config`,
  then lets explicit per-key decode values in the same request win. Unknown id -> 400.
  The resolved id stays in the job config (`analysis_job.json`); library `meta.json`
  is unchanged (it records no analysis params today). The frontend shows a
  "Decode preset" select in Advanced settings with the description as helper text,
  remembers the choice in localStorage, and hides the control when an older backend
  omits `decode_presets`.
- **Why:** The jp-junior profile measurably helps beta-tennis matches, and GUI users
  had no way to apply it (API-only). Keeping presets on the server keeps one source
  for API clients and the UI and preserves the "all GUI behavior via /api/*" invariant;
  a test pins jp-junior to `configs/decode/jp-junior.toml` so CLI and GUI can't drift.
  `default` sends no params, so default GUI runs stay manifest-only.
- **Rejected:** Per-key numeric inputs in the UI (five unlabeled knobs invite
  untested combinations; the preset is the tuned unit); hardcoding presets in
  `script.js` (UI and API would drift, and non-browser clients couldn't discover them);
  reading `configs/decode/*.toml` at runtime (not shipped in the PyInstaller bundle);
  putting `decode_preset` in `DEFAULT_CONFIG` (the frontend spreads defaults into every
  upload, which would make the server default sticky).

## 2026-10-11 — Score state machine + server-sequence check (Phase 4, label-free part)

- **What:** `src/analysis/score.py` (standard library only, outside the runtime path):
  `MatchFormat` presets (bo3, bo3-mtb10, bo1, bo5; ad / no-ad; first server), an
  immutable `ScoreState` with `advance(state, winner)` (deuce/AD, no-ad deciding point,
  tiebreak with the 1-2-2 service rotation and ends every 6 points, ends after every
  odd game of a set, server alternating across sets with a tiebreak counted as one
  game, 10-point match tiebreak in lieu of the deciding set), `reconstruct` (per point)
  and `reconstruct_games` (per game), `infer_first_server_and_check` (Phase 1 eval
  hook), and beta-tennis winner-CSV readers. `scripts/eval/reconstruct_score.py` wraps it.
  Players are only "A"/"B"; the near/far mapping stays with Phase 1 (`sides` is a parity
  flag "same"/"swapped").
- **Why:** Once point winners exist (ball/outcome work, scoreboard OCR or a human),
  the score timeline, the server of every point and the change-of-ends points follow
  deterministically, and the same rules give a label-free check of a detected server
  sequence. With winners unknown, game boundaries are placed by an O(N x max_game_points)
  dynamic programme (games of 4..30 points, server constant within a game, alternating
  across games; partial last game allowed) that minimises disagreements; both first
  servers are tried (A wins ties). With winners known, the exact replay is used,
  tiebreak rotation included.
- **Finding:** TrimDeck `*_winner.csv` rows are GAME events (`gameEvents.gameWinner`),
  not points, and `score_after` is games summed over the match ("12-6" = 6-0 6-6), so
  the reader auto-detects the unit and accepts either per-set or cumulative strings.
  winner code 0 = "A" (TrimDeck team0, named from `winner_name`).
- **Rejected:** Tracking near/far identity here (Phase 1's job; keeps this module pure);
  modelling tiebreaks in the unknown-winners DP (needs point winners to know when 6-6
  is reached; a tiebreak shows up as one long mixed-server game and costs a few
  disagreements); an HMM/probabilistic decoder (the min-disagreement DP is exact and
  explainable; revisit when detector confidences exist); a third-party tennis-scoring
  package (stdlib-only keeps it vendorable into any worker).

## 2026-10-11 — Phase 2 first cut: rule-based shot events + stroke type, label-free

- **What:** `src/analysis/shots.py` (numpy only, outside the runtime path) detects swing
  events per player from the 5 fps pose dump (wrist speed relative to the mid-shoulder
  point, in body scales/s; per-step L/R swap repair; threshold
  `max(6 bs/s, median + 1.0 * 1.4826 * MAD)` per point; bounded prominence; 0.6 s
  refractory; parabolic peak time), classifies serve / overhead / volley / forehand /
  backhand / unknown with documented pose rules, and merges both sides with the rally
  alternation prior (opposite-side overlap < 0.4 s deduped, same-side events < 1.5 s
  apart collapsed to the faster, a prior serve / prior opponent shot inserted where
  alternation breaks; prior events are counted separately). Sanity tooling:
  `scripts/eval/score_shots.py` (label-free report) and `scripts/eval/export_shot_clips.py`
  (1.2 s clips + `labels_template.csv` for human stroke labels).
- **Why:** No stroke labels exist, so a learned classifier cannot be trained or evaluated
  yet; a transparent rule baseline gives rally length / serve / stroke-mix numbers now,
  drives the labelling tool (pre-filled predictions), and becomes the baseline the
  learned model must beat. Choices measured on beta-tennis 2R (tdproj points): the
  signal is the faster of the two wrists, not the dominant one, because the two wrists'
  95th-percentile speeds differ by only 6 % there (a camera side holds two players across
  change-overs; YOLO L/R labels swap on ~1-2 % of steps); `rel_k` 3 -> 1 plus a
  prominence test recovered the slower swings of one player (threshold had been 11 bs/s
  while real swings peaked at 9-11) while keeping running ripples out; same-side gap
  1.0 -> 1.5 s because serve toss + contact read as two swings ~1.0 s apart; overhead
  margin 0.15 -> 0.8 body scales (serves sit at >= 0.9; follow-throughs at 0.2-0.6); the
  pre-peak sample decides forehand/backhand (the post-peak sample is the follow-through,
  which crosses the body for both).
- **Far player:** the shipped preprocessor's near-court mask drops the far player before
  assignment (2R: `far_present` 7 / 10300 frames). The preprocessor is not changed; the
  dump now also saves raw pre-filter detections, and `score_shots.py` rebuilds a far
  track from them with a script-private heuristic (`_far_from_raw`) until Phase 1's
  `analysis.players` owns that choice.
- **Rejected:** a learned stroke classifier (TCN on normalised keypoints) before labels
  exist (nothing to fit or score it on; the roadmap keeps it for after ~2k labels);
  dominant-arm-only signal (not identifiable from labels at this resolution);
  wiring shots into the engine / CLI / GUI now (joined with Phase 1 later; no runtime
  path changes, golden parity untouched); changing the preprocessor's far-player filter
  (runtime behaviour and golden parity).

## 2026-10-11 — Phase 1 first cut: near-player-first server + A/B identity heuristics

- **What:** `src/analysis/players.py` (numpy only): `detect_server` = near player's
  wrist above the head with the arm extended (>= 0.7 torso lengths from the shoulder) in
  [start - 1.0, start + 2.5] s -> near serves; near clearly seen (arm keypoints valid in
  >= 70 % of the window) and never reaching overhead -> far serves by elimination (fixed
  confidence 0.3). `track_identities`: A = near player of point 1; each boundary with a
  gap >= 25 s scores 0.5 x gap (0 at 35 s, 0.6 at 60 s, 1 at >= 90 s) + 0.2 x near body
  signature change (median log limb-length ratios, scale-free) + 0.3 x serve-side
  continuity (+1 same side, -1 flip); >= 0.5 flips the ids. Engine opt-in
  `RunRequest.save_player_tracks` (default False, golden byte-equal) tees kept-frame
  near/far + raw pose frames and writes `<name>_players.npz` (dump-compatible) +
  `<name>_points.json`; CLI `--players`; GUI jobs always on, served at
  `/api/library/<id>/points`.
- **Why:** The far player is not detected by the shipped 960-px pose pass on beta-tennis
  (far present 5-22 frames per match out of 10-19k; raw pre-court-filter detections
  contain no far-court person either), so every decision has to come from the near
  player. Serve-side continuity is the key structural cue: a change of ends keeps the
  server on the same side (the new server stands where the old one did), while a game
  boundary without a change flips it. Evaluated against a reference derived from the
  tdproj scoreboard (game ends + gameWinner + format + initialServer): ends change after
  the 1st, 3rd, ... game OF EACH SET, so a 5-game short set (4-1) is followed by two
  changes in a row; the "cumulative odd games" shortcut was wrong and made 1R look like
  77 % server accuracy until fixed. Results (tdproj points): server 0.986 / 0.982 / 0.987
  / 1.0 / 0.968 (1R/2R/3R/SF/P1R), identity 1.0 x 4 and 0.968, side changes P 1.0,
  R 1.0 except P1R 7/8 (tiebreak); on jp-junior decoded points, matched to GT: server
  0.965-1.0, identity 0.977-1.0. Parameters were set from first principles and checked
  on 2R before the other matches were dumped; no grid search.
- **Rejected:** a court homography (not needed for side/serve; deferred); identity from
  gaps alone (non-changeover game breaks reach 100-160 s on 1R/2R/3R); a far-player
  raise as the primary far-serve signal (no far detections); confidence as a product of
  linear margins (collapsed to ~0.01 by the end of a match even when every decision was
  right -> logistic per-decision confidence, still multiplicative); changing the
  preprocessor's near-court filter (shipped feature contract + golden parity).

## 2026-10-11 — Far player needs v0.6.0's crop pass, upscaled; recommend v0.6.0 as the base

- **What:** The shipped v0.5.0 pose pass (YOLOv8n-pose at imgsz 960 on 1080p) does
  not detect the far player on the customer's footage (40-55 px tall); raw pre-filter
  detections contain a second person in ~2% of frames. Phases 1/2 were therefore
  built near-player-first. Upstream `feat/runtime-v060` (court-line homography +
  far-court crop pass + net-line slots) was cloned to `~/Documents/RallyClip-v060`
  and evaluated on 2R: geometry fits, point detection is equal or better, but the
  native-scale crop still finds the far player in only ~2% of in-point frames; a 2x
  upscaled crop finds them in 4 of 6 test frames. A geometry-driven upscaled crop is
  being prototyped on a local branch there. Recommendation to the user: adopt v0.6.0
  as the base and port this snapshot's work onto a fork.
- **Why:** Everything above Phase 1 (far strokes, outcomes) needs the far player;
  re-implementing crop + homography on v0.5.0 would duplicate upstream.
- **Rejected:** Lowering YOLO conf (tested 0.1: no gain); relying on the alternation
  prior for far shots long-term; building our own homography on v0.5.0.

## 2026-10-11 — Port the v0.5.0 snapshot work onto `feat/runtime-v060` (fork `ShotaOgino/RallyClip`)

- **What:** v0.6.0 is the base (user decision). The snapshot's uncommitted work (decode
  overrides + `end_pad_sec`, GUI decode presets, 32 GB upload cap, `scripts/eval/`,
  `src/analysis/` + the opt-in player outputs) lands as `feat/port-v050-analysis`,
  branched from `origin/feat/runtime-v060`, ported by 3-way cherry-pick from a local
  commit on `v0.5.0` (the snapshot's exact base) and resolved by hand. Deltas vs v0.5.0:
  1. **jp-junior = threshold 0.5, smooth_sigma_frames 1.5, merge_gap_sec 0.0,
     hybrid_min_duration_sec 1.0, end_pad_sec 1.6** (v0.6.0 2R point F1 0.83 default ->
     0.925; the v0.5.0 profile's threshold 0.65 is not used). All five keys are explicit so
     the profile does not move if the manifest's decode defaults change; the GUI preset
     and `configs/decode/jp-junior.toml` stay test-pinned to each other. The only material
     changes vs the v0.6.0 manifest are the 1.0 s floor (manifest 4.0 s drops short real
     points) and the 1.6 s end pad (label convention).
  2. **Player tee follows the v2 slot stream.** `build_feature_stream(player_sink=)` wraps
     `DataPreprocessor.iter_slot_players` (`_tee_slot_players`) for v2 feature sets and the
     v1 records (`_tee_player_records`) otherwise; raw pre-filter detections come from the
     pose frame the preprocessor is on (`_tee_pose_frames`) in both. The eval dump tees the
     same seams independently, so the engine npz can be checked against it (equal on the
     golden clip).
  3. **`PlayerTracks.with_raw` keeps the shipped far record per frame and fills only its
     gaps** from `pick_far_player` (v0.5.0 replaced it wholesale). On v0.6.0 the shipped far
     slot comes from the far-court crop pass and is the better source; v0.5.0 behaviour
     is unchanged when the shipped far record is empty (`far_source == "raw"`).
  4. **No 1280x720 assumption to remove:** every rule in `players.py` / `shots.py` is a
     fraction of the frame or of a body/box size and `ref_size` is read from the dump /
     request; only defaults (for old dumps without `ref_size`) and docstrings said 1280x720.
  5. **`score_heatmap_decode.py` bases its sweep on the dump's own manifest params**
     (fallback: the v0.5.0 values), so `{}` reproduces the run's decode on any artifact.
  6. **`tests/fixtures/golden_cli/clip.tracks.npz` regenerated** from this branch's golden
     clip with the ported dump (1 default segment, far present 36/120 frames, ref
     1920x1080, `savez_compressed`); the golden-tracks shot test now expects motion shots
     on both sides.
  7. **Upload cap:** `feat/runtime-v060` still had 2 GB, so the 32 GB cap is ported as-is.
     Upstream PR #60 (open, against `main`) removes the cap entirely; reconcile when either
     lands.
- **Why:** v0.6.0 brings the court homography and the far-court crop pass that every phase
  above point detection needs (the far player is invisible to the v0.5.0 pose pass);
  porting semantics rather than patches keeps the default path byte-identical (golden clip
  CSV equal to unmodified `feat/runtime-v060` on this machine).
- **Rejected:** carrying the v0.5.0 jp-junior numbers over (tuned on the other model; on
  v0.6.0 the threshold change no longer helps); re-deriving the far player only from raw
  full-frame detections on v0.6.0 (discards the crop pass); reusing the engine sink inside
  the dump script (loses the independent cross-check); committing `logs/eval/` outputs.

## 2026-10-11 — Serve side (deuce / ad) from position + score parity, cross-checked (Phase 1)

- **What:** `src/analysis/serve_side.py` (numpy only). Visual: the server's feet (ankle
  midpoint, else box bottom-centre) through the court homography, median over -1.0/+0.5 s
  around the server's first overhead raise (`detect_server` evidence; points without a
  raise use start + the match's median raise offset); without the server's track, the
  receiver's feet (`source: "receiver"`). Score: parity of the point's index within its game
  (even = deuce, also per tiebreak point) from winners / scoreboard game ids, else the
  label-free server-id segmentation (`infer_first_server_and_check`). `combine`: visual
  when its confidence >= 0.5, else score; disagreement flagged (`serve_side_agree: false`).
  Rows gain `serve_side`, `_confidence`, `_source`, `_score`, `_agree`, `_visual*`;
  `_points.json` gains `serve_side_check`. Geometry rides on `PlayerRecordSink.geometry`
  (set in `build_feature_stream` for v2 net-line runs; the default path is untouched) and
  in the npz as `geometry_json` (engine sidecar + `dump_heatmap_tracks.py`).
- **Side rule:** near player at camera-right (court x > 0) or far player at camera-left
  (x < 0) = deuce, for the server AND the receiver: each stands in their own deuce court
  (their right when facing the net), and the two deuce courts are diagonal. The task brief
  said "a near receiver on camera-left means the far server is on the deuce side"; that is
  the opposite and the data reject it: on 2R the near receiver of far serves stands at
  x = +3.3..+3.6 m on even (deuce) points and -3.4..-3.7 m on odd ones (22/23 agree with the
  scoreboard parity under the rule above, 1/23 under the brief's). The role still matters
  for the confidence scale (server ~0.4-1.4 m from the centre mark -> 0.25 m scale;
  receiver ~3.5 m -> 1.0 m).
- **Second-serve merge (`same_point_gap_s` 5.0):** beta-tennis tdproj sometimes clips a
  first-serve fault and the second serve separately. On 2R, consecutive clips served from
  the same side are 1.4-3.8 s apart (8 cases) while alternating ones are >= 6.7 s apart, so a
  clip starting < 5 s after the previous one keeps the point index. Clip parity 43/55 ->
  50/55 agreement.
- **Numbers (2R, v0.6.0 dump, tdproj points, scoreboard game ids):** visual side on 55/55
  points; agreement 50/55 = 0.909 — near serves (server position) 28/32 = 0.875, far serves
  (receiver position, far track absent) 22/23 = 0.957. Visual confidence median 0.975, 51/55
  >= 0.9, min ~0.72. The 5 disagreements: 4 near-serve points in two pairs (#19/#20, #27/#28;
  the 8.3 s / 11.1 s same-side gaps before them are too long for the merge) where frames at
  the serve show the server on the side the visual says (checked by eye), i.e. the clip
  parity is off by one (a repeated or missing clip); and #54, the trailing duplicate clip.
  On decoded points (jp-junior params) with the label-free segmentation the agreement is
  45/54 = 0.833 (game boundaries misplaced in one game; the visual answer wins there).
- **Why:** position is a direct observation and needs no point bookkeeping; score parity is
  independent and catches visual failures (players not seen, odd positions), so the
  cross-check is the label-free accuracy signal per match.
- **Rejected:** box centre x (the side-on server's tossing arm and racket shift it; ankles
  first); anchoring the score parity to confident visual points (makes the two sources
  dependent, so the cross-check stops meaning anything); a hard serve window at the point
  start (decoded and tdproj starts sit ~1.2-1.9 s before the raise); requiring the far track
  (present in ~2 % of in-point frames on beta-tennis).
