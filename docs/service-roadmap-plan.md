# Service roadmap — from point clipping to "who hit what, and how the point was won"

_Working doc (Tier 3). Written 2026-10-11. Move to docs/archive/ when superseded._

## Goal

Ship a paid/online service for Japanese junior tennis matches (fixed rear camera,
1080p60, 30-60 min) that returns, per point: start/end, server, each shot (who, stroke
type, timestamp), and the outcome (winner / forced / unforced error, in/out/net), plus
match-level stats. Point clipping (today's RallyClip) stays the first deliverable at
every stage, so every phase ships something usable.

## Ordering principle

Value ladder with the cheapest, most certain layer first, and each layer validated on
the 33 human-confirmed `.tdproj` matches before the next is built on top of it:

1. point boundaries -> 2. who is who -> 3. shot events + stroke type -> 4. ball /
bounce -> 5. outcome + score -> 6. service hardening (parallel from phase 2).

Rationale: layers 2-5 all consume the point boundaries and the per-frame pose that
RallyClip already produces; a bad layer 1 poisons everything above it. Ball tracking
(layer 4) is the most uncertain technically, so it is deferred until the pose-only
layers deliver stats customers already pay for (rally length, serve stats, stroke mix).

## Phase 0 — Solid point detection on JP junior data (1-2 weeks)

- [x] Heatmap decode params (`threshold`, `merge_gap_sec`, ...) overridable from
      config.toml / CLI / `/api` (today: manifest only). Add `end_pad_sec` postprocess
      knob (beta-tennis end convention = +1.3-1.6 s). Golden parity untouched by default.
      Done 2026-10-11: `[run]` keys / `--threshold` `--merge-gap-sec` `--smooth-sigma-frames`
      `--hybrid-min-duration-sec` `--end-pad-sec` / `/api/upload-and-start` config -> `RunRequest.decode_params`.
- [ ] Dump tracks for every match that has both video and tdproj (`scripts/eval/`),
      grid over threshold 0.5-0.8, pick a `jp-junior` profile. Target: point F1 >= 0.85,
      merged <= 5 %, missed <= 2 % on held-out matches.
- [x] Git home: fork `iroblesrazzaq/RallyClip` (user decision) so work is reviewable.
      Done 2026-10-11: `ShotaOgino/RallyClip`, branch `feat/port-v050-analysis`.
- [x] Decide on the v0.6.0 upstream runtime (court model, far-court crop): rebase or
      stay on v0.5.0 snapshot. Done 2026-10-11: v0.6.0 is the base; the snapshot work
      was ported onto `feat/runtime-v060`; jp-junior re-measured on v0.6.0 (2R point F1
      0.83 -> 0.925 with threshold 0.5 / smooth 1.5 / merge 0.0 / floor 1.0 / pad 1.6).

Exit: a config profile + eval report; the GUI/CLI produce clips customers accept.

## Phase 1 — Player identity (1-2 weeks)

- Court homography from the existing court detector -> near/far side per frame.
- Track the two main players (pose boxes + court position), handle side changes
  (odd games) and ball kids/umpire rejection.
- Output per point: `server`, `near_player`, `far_player`, players' court positions
  at 5 fps. Serve detection = first overhead-arm pose at point start, on the server's side.
- Eval: server correct >= 95 % on tdproj matches (label server via the scoreboard /
  manual spot check; ~20 points per match suffice).

Status (2026-10-11, first cut, near-player-first; see docs/PROGRESS.md "Phase 1"):
- [x] Server per point from the near player's overhead reach (else far by elimination):
      `src/analysis/players.py::detect_server`. Scoreboard-derived reference on 5 tdproj
      matches: 96.8-100 % (all-points 0.986 / 0.982 / 0.987 / 1.0 / 0.968); errors are the
      trailing duplicate clip and the P1R tiebreak.
- [x] Player ids A/B per point + side changes (gap + near body signature + serve-side
      continuity): `track_identities`; identity 100 % on 1R/2R/3R/SF, 96.8 % on P1R
      (one in-tiebreak change missed). Engine opt-in `RunRequest.save_player_tracks`,
      CLI `--players`, GUI jobs always on + `GET /api/library/<id>/points`.
- [x] Label-free check `serve_consistency`; eval `scripts/eval/analyze_players.py`.
- [x] Serve side (deuce / ad) per point, 2026-10-11: `src/analysis/serve_side.py`; visual (server feet in
      court metres via the v0.6.0 court homography, receiver when the server is unseen) cross-checked with the
      score parity of the point index in its game. 2R: visual on 55/55 points, agreement with the scoreboard
      parity 0.909 (near serves 0.875, far serves via the receiver 0.957); the disagreements are clip-parity
      errors (repeated / missing clips) or the trailing duplicate clip, not position errors.
- [ ] Court homography for the other Phase 1 outputs (positions are still reference pixels; the serve side
      is the first user of court metres).
- [ ] Far player: not detected by the shipped pose pass (far present < 0.2 % of frames,
      raw detections included), so far-side evidence and far court positions do not
      exist yet; `pick_far_player` is a best-effort helper for when a far-court crop /
      higher-res pass lands (v0.6.0 question).
- [ ] Tiebreak side changes (every 6 points, no break) are invisible to the gap cue.
- [ ] Only one player / venue / day (Tsukamoto, 2026-08-24); other venues untested.

## Phase 2 — Shot events + stroke classification (3-5 weeks)

- Shot event = swing peak in wrist/elbow keypoint velocity; de-duplicate against the
  opponent's swing; use rally alternation as a prior.
- Stroke type (serve / forehand / backhand / volley / overhead / slice optional) from a
  1-2 s pose window: start with a small TCN on normalized keypoints (same stack as the
  point model), pretrain on THETIS/TenniSet-style public data, fine-tune on ~2k labelled
  swings from the tdproj matches (labeling tool: extend the beta-tennis editor).
- Output per point: `shots[] = {t, player, stroke, confidence}`, rally length.
- Eval: event F1 >= 0.9 within 0.3 s; stroke accuracy >= 85 %.

Status (2026-10-11, first cut, label-free):
- [x] Shot events from 5 fps wrist speed (swap repair, per-point robust threshold,
      prominence, 0.6 s refractory, parabolic peak time) + alternation prior (dedupe,
      same-side suppression, prior serve / prior opponent shot, counted separately):
      `src/analysis/shots.py`. Not wired into engine/CLI/GUI (joined with Phase 1 later).
- [x] Rule-based stroke baseline (serve / overhead / volley / forehand / backhand /
      unknown; rules + limits in `classify_stroke`'s docstring).
- [x] Label-free report `scripts/eval/score_shots.py`: on 1R/2R/3R/SF (tdproj points)
      the inter-shot interval median is 1.45-1.61 s and the serve side agrees with the
      per-game majority on 93-97 % of points (server pairs n n f f ... follow the
      change-of-ends rule).
- [x] Labelling helper `scripts/eval/export_shot_clips.py` (1.2 s clips +
      `labels_template.csv` with predictions pre-filled).
- [ ] Far player: not detected by the shipped pose pass at all on beta-tennis (raw
      pre-filter detections hold no far-court person on SF), so every far shot is a
      prior (40-47 % of shots). Needs a far-court crop / higher-res second pose pass
      (v0.6.0 runtime question in Phase 0) before far stroke types exist.
- [ ] Label ~2k swings with the helper; measure event F1 / stroke accuracy; then the
      learned classifier (TCN on normalised keypoints) replaces `classify_stroke`.
- [ ] Per-player handedness from Phase 1 identity (a camera side mixes two players;
      forehand/backhand depends on it).

## Phase 3 — Ball tracking, bounce, in/out (4-8 weeks, parallelizable with 2)

- TrackNetV3-class ball detector fine-tuned on the rear fixed-camera footage; Kalman
  track; bounce = trajectory vertical-velocity sign change; map to court via homography.
- Only needed for outcome detail (where the ball landed, net). Start with the
  scoreboard OCR shortcut (Phase 4) if the venue has a visible scoreboard.
- Eval: bounce location error <= 50 cm on 200 labelled bounces.

## Phase 4 — Outcome and score reconstruction (2-4 weeks)

- Per point: last shot + ball/bounce evidence -> winner / error type; score state
  machine (deuce/ad, tiebreak, sides) with server inference as a consistency check.
- Fallback / cross-check: scoreboard OCR when present (beta-tennis `ScoreBoard`).
- Output: point log, game/set score timeline, player stats; PDF/HTML match report.
- [x] Score state machine + reconstruction (label-free part). Done 2026-10-11:
      `src/analysis/score.py` (`advance`, `reconstruct`, `reconstruct_games`; ad/no-ad,
      tiebreak rotation + ends, bo3 / bo3-mtb10), `infer_first_server_and_check` (Phase 1
      eval hook: agreement + disagreeing points), beta-tennis winner-CSV reader;
      `scripts/eval/reconstruct_score.py`. ITF winner CSV (18 game events) replays with 0
      mismatches. Not done: per-point outcome (needs ball/bounce), scoreboard OCR, stats,
      match report.

## Phase 5 — Service hardening (starts in parallel with Phase 2)

- Backend: job queue + GPU worker (the current Flask jobs dir -> Redis/RQ or Cloud
  Tasks), object storage for uploads (2.5-8 GB), resumable upload, progress API.
- Product: web upload + viewer (existing GUI frontend), per-point filters by player /
  stroke / outcome, share links, export (CSV / EDL / clips).
- Trust: auth, per-user data isolation, retention policy, delete on request,
  video never leaves region (JP).
- Ops: CI on every PR (already), model/version pinning via artifact registry (already),
  regression eval against tdproj set on every model change, monitoring, cost per match.
- Pricing / billing last; keep the desktop app as the offline tier.

## Compute note

Pose extraction is 10-20 min per match on an M3 on AC power; the 2026-10-11 run took
hours on battery. Batch extraction belongs on a GPU box (beta-tennis `olive`).
