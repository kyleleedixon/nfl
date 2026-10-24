Inputs the weekly GitHub Action can't rebuild cheaply from nflverse:
- `preds_v3/` — finished walk-forward predictions for completed seasons: full_v3 2019–2025 and live_v3 2019–2025
  (the live set is the record book the game cards compare against, live model vs live model). Only the current season's
  live model is re-trained each week (`backtest_v3.py live`); refresh these after re-running `backtest_v3.py run`.
- `ablate_v3.json` — the ablation's feature-group / CatBoost decision (dropped coach-bias features). Without it the
  model would silently fall back to all groups.
- `players.parquet` — the nflverse players table the features were built with (positions for lineup tells).
