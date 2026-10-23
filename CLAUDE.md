# NFL play-call predictor ("Telegraphed", X: @TelegraphedNFL)

Predicts pass vs. run before the snap. Kyle's thesis: play-callers are human and therefore
predictable; the model leans on human-decision patterns and personnel tells, not just game state.

## Layout
- `src/features.py` — base features from nflverse play-by-play + participation (2016–2026).
  Situation, human/cognitive (last-play result, streaks, tempo, in-game hot hand, emotional
  moments, coach/team habits from prior games), personnel (counts, formation, per-player tells).
- `src/features_v2.py` — second pass: per-role player tells, specialist/snap-share detection,
  exact skill-lineup tells, substitutions vs previous snap, team/coach tendencies by
  personnel x formation x shotgun, in-game lean by personnel, coach/team by down-distance, QB experience.
- `src/backtest.py` — v1 layered ablation (situation → +human → +personnel → all). Results in `results_v1/`.
- `src/backtest_v2.py` — `tune` (holdout 2024) and `run` (walk-forward 2019–2026, 3-config LightGBM ensemble,
  params in `src/best_params.json`). Two models: `full_v2` (needs participation data) and `live_v2`
  (play-by-play only, usable in-season).
- `src/features_v3.py` — third pass: per-coach decision biases (streak/pattern/balance/win-stay-lose-shift/repeat),
  defensive alignment (box count, defensive personnel), FTN charting pre-snap fields (2022+, published in-season).
- `src/backtest_v3.py` — `ablate` (2023 step; writes `data/ablate_v3.json`), `tune` (Optuna, train <=2021,
  early-stop 2022; resumable sqlite), `run` (resumable walk-forward, v2 baselines on holdout). Current model.
- `src/stage2.py` — stage-2 correction applied in `backtest_v3.merge`: nudges v3 by the team's season-to-date residual
  (prior games) and in-game residual (prior snaps). Frozen coefficients in `src/stage2.json` (fit on full_v3 2020–22).
  `preds_v3.parquet` holds corrected `full_v3`/`live_v3` plus raw `*_raw`.
- `src/features_v4.py` + `src/screen.py` — experiment bench: QB-at-snap (play-by-play only) and fast team-tendency
  features, screened with one LightGBM config on 2021–2023 (`screen.py base,qb` then `screen.py report`). Not in production.
- `overnight.sh` — unattended v3 pipeline through deploy + push (log `data/overnight.log`).
- `src/analyze.py` — v1 analysis (tells, coach table, EPA vs predictability). Reads v1 preds.
- `src/showcase_data.py` → `site/results.json`; `src/pick_drive.py` → `site/drive.json`.
- `src/weekly_report.py` — Tuesday "How predictable is your offense?" rankings (neutral-situation read rate by team,
  team labels only), share card PNG, draft X thread (`data/reports/`), and `site/weekly/weekly.json` for the public
  page. `site/weekly/index.html` is a hand-written template sharing the main page's tokens and chart code.
- `weekly.sh` — Tuesday refresh: new week → features → `backtest_v3.py live` (re-train current season's live model only)
  → report → push (Vercel deploys). No-ops if no new week (last reported week = `site/weekly/weekly.json`).
- `src/post_x.py` — posts the weekly tweet + card to X (@TelegraphedNFL). Creds from env (GitHub secrets) or
  `~/.config/telegraphed/x.env`, never the repo. Resumable via `state/x_posted.json`; a file `state/x_paused` stops it;
  refuses drafts >2 days old. `--dry-run` previews.
- Automation runs on **GitHub Actions**, not the laptop: `.github/workflows/weekly-refresh.yml` (Tue 11/14/16 UTC) and
  `weekly-post.yml` (Tue 17/19 UTC = 11am CT). Both can be run by hand from the Actions tab. A fresh checkout needs
  `artifacts/` (cached 2019–2025 preds, ablation decision, players table; see its README) and `state/` (tweet drafts,
  posting record). Refresh `artifacts/preds_v3/` after re-running `backtest_v3.py run`. `tools/pyexit/` works around
  a pyarrow hang at interpreter exit on macOS.
- `site/index.html` — public showcase (static; fetches the two JSON files). Deploy the `site/` folder only.
- `run_all.sh` — rebuilds everything end to end (`TUNE=1 ./run_all.sh` to re-tune).

## Setup
- Python 3.11 venv: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`.
  pandas is pinned <3 (code relies on 2.x semantics). Run scripts with `.venv/bin/python` (`run_all.sh` uses `$PY`).
- macOS without Homebrew libomp (Homebrew builds it from source on macOS 12): point LightGBM at scikit-learn's bundled copy:
  `SP=.venv/lib/python3.11/site-packages; install_name_tool -add_rpath "$PWD/$SP/sklearn/.dylibs" $SP/lightgbm/lib/lib_lightgbm.dylib`

## Rules
- Every feature must be knowable before the snap; historical rates use prior games only,
  in-game rates use prior snaps only. Walk-forward: season S is predicted by a model trained on seasons < S.
- The public site shows aggregate results only. Never put feature names, importances, tells or
  per-feature charts in `site/`. Keep this repo private; deploy only `site/`.

## Results so far (all out of sample, 2019–2025, ~243k plays)
| Model | Accuracy | Neutral downs | Log loss |
|---|---|---|---|
| Always pass | 61.0% | 52.2% | 0.669 |
| nflfastR xpass (public benchmark) | 71.2% | 61.0% | 0.527 |
| v1 situation only | 71.2% | 61.0% | 0.525 |
| v1 + human factors | 73.1% | 64.2% | 0.509 |
| v1 + personnel | 75.8% | 68.0% | 0.483 |
| v1 full | 76.4% | 69.0% | 0.472 |
| v2 full, 2024 holdout only | 78.2% | — | 0.461 |
v2 walk-forward partial: 2019 77.3% (ll 0.452), 2020 76.5% (ll 0.468). Finish with `python3 src/backtest_v2.py run`.

Key human tells (v1): stuffed run → +7 pts pass over expected; big run / turnover → lean run;
after 2–3 straight runs coaches over-pass ~6–7 pts (and vice versa); hot hand ±4–5 pts; no-huddle +4 pts.
Surprise passes gain ~+0.05 EPA/play more than predictable ones (descriptive).

### v3 (2026-10-23) — clean holdout: no modelling decision used 2024+
| Season | full_v3 | full_v2 | live_v3 | live_v2 | nflfastR xpass |
|---|---|---|---|---|---|
| 2024 | 79.6% (ll 0.433) | 78.7% (0.453) | 78.9% (0.444) | 75.0% (0.492) | 71.1% (0.531) |
| 2025 | 77.6% (0.459) | 77.2% (0.469) | 77.1% (0.467) | 75.6% (0.487) | 71.0% (0.536) |
| 2026 wk1–6 | — | — | 78.5% (0.451) | 76.6% (0.473) | 71.3% (0.525) |
Site headline (full_v3, 2019–2025, 243k plays): 77.5%, ll 0.455, AUC 0.856 vs public 71.2%.
2019–2023 full_v3 numbers are slightly optimistic (tuning saw 2022, ablation saw 2023).
Ablation on 2023 (ll): v2 0.4790 → +align 0.4718, +ftn 0.4743, +bias 0.4794 (no gain, dropped), all kept 0.4682,
+CatBoost blend 0.4635. Biggest tells: pistol (25% pass) vs shotgun (78%), empty backfield (94%), box count.
The coach-bias features added nothing beyond existing streak/prev-play/coach-PROE features — the trees already learn it.

### 2026-10-23 model deep dive
- Error analysis (full_v3 2021–23): 1st & 10 = 47% of log loss; team-season bias variance ~6x noise (first-half bias
  correlates 0.51 with second half) → the model is slow to learn a team's current identity.
- **Formation labels changed in 2023** (nflverse: EMPTY/I_FORM/SINGLEBACK/JUMBO/WILDCAT → SHOTGUN/UNDER CENTER/PISTOL).
  `features.py` now maps every season onto the new set. Rule: predictions use the taxonomy consistent with their season,
  so 2019–2022 full_v3 keep the old-label run (detailed labels were consistent then and better: 2021 0.454 vs 0.466)
  and 2023–2025 were re-run with harmonized labels (2023 0.4632 → 0.4527; 2024 0.4333 → 0.4327; 2025 flat).
- Stage-2 correction (decided on 2023, confirmed once on 2024–25): +0.0014–0.0016 ll on every holdout season.
  Final (after stage 2): full_v3 2024 79.7% (ll 0.431), 2025 77.7% (0.457); live_v3 2024 79.0% (0.443), 2025 77.2%
  (0.466). Site headline 2019–2025: 77.7%, ll 0.452, AUC 0.858, neutral 70.7%.
- Tried, no gain (dropped): QB-at-snap identity/quality/mobility features (+0.0009 before the formation fix, 0 after),
  last-3-games / season 1st-down / scripted-play / last-season lean, new head coach, in-game 1st-down lean; extra stage-2
  terms (opponent, down bucket, coach, drive, last 3 games: ≤ +0.0004).
- Next ideas: OOF stacking with model-residual features, multi-season tuning, play-caller (OC) data, injuries/inactives.

## Caveats
- nflfastR xpass was fit on seasons overlapping 2019–2020, so the benchmark is slightly flattered there.
- Participation (personnel) data for 2026 isn't published yet, so the live 2026 model can't use personnel.
  The live model was found using personnel-derived in-game features; fixed to play-by-play-only inputs.
- Recency weighting (half-life 3 seasons) hurt on the 2024 holdout; not used.

## Status / next steps
- Brand: **Telegraphed**, X account @TelegraphedNFL. Growth plan: weekly Tuesday rankings thread + card (Umpire
  Scorecards / Scorigami model); next: per-game cards, an "Open Book" callout (≥85% read). Never tweet tells/reasons.
- Site live at https://telegraphed.app (www.telegraphed.app primary; also telegraphed.vercel.app; Vercel project `nfl`, Git-connected to this repo, root directory `site/`,
  so every push to main deploys). Never change the root directory: the repo root would publish the source. GitHub repo
  `kyleleedixon/nfl` is currently PUBLIC — Kyle will make it private.
- End goal: real-time pre-snap estimator. Lag is fine for now; never drop data for lack of a live feed.
- Ideas not yet tried: own xpass/WP models (remove nflfastR dependency for live use), two-stage prediction
  (pre-huddle vs at line-up), weekly in-season refresh, multiclass target, test against betting lines,
  pre-snap tracking (Big Data Bowl data).
