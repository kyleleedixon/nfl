# NFL play-call predictor ("Pre-Snap Read")

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
- `src/analyze.py` — v1 analysis (tells, coach table, EPA vs predictability). Reads v1 preds.
- `src/showcase_data.py` → `site/results.json`; `src/pick_drive.py` → `site/drive.json`.
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

## Caveats
- nflfastR xpass was fit on seasons overlapping 2019–2020, so the benchmark is slightly flattered there.
- Participation (personnel) data for 2026 isn't published yet, so the live 2026 model can't use personnel.
  The live model was found using personnel-derived in-game features; fixed to play-by-play-only inputs.
- Recency weighting (half-life 3 seasons) hurt on the 2024 holdout; not used.

## Next steps
1. Finish v2 walk-forward, then `showcase_data.py` and `pick_drive.py`; preview `site/` locally.
2. Deploy `site/` to Vercel (`cd site && vercel deploy --prod`), private repo for everything else.
3. Ideas not yet tried: live personnel source for in-season use, CatBoost in the ensemble,
   multiclass target (short/deep pass, inside/outside run), test against live betting lines.
