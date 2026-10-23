"""Walk-forward backtest: train on all seasons before S, predict S. Layered ablation."""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

FEAT = sys.argv[1] if len(sys.argv) > 1 else "data/features.parquet"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data"

SIT = ["down", "ydstogo", "yardline_100", "score_differential", "game_seconds_remaining",
       "half_seconds_remaining", "qtr", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
       "wp", "vegas_wp", "spread_line", "total_line", "home", "outdoors", "temp", "wind", "goal_to_go"]
COG = ["prev1_type", "prev2_type", "prev_result", "prev_yards", "prev_epa", "prev_success",
       "prev_first_down", "first_of_drive", "secs_since_prev_snap", "same_type_streak", "streak_signed",
       "ingame_pass_rate", "ingame_pass_epa", "ingame_run_epa", "ingame_hot_hand", "ingame_pass_sr",
       "ingame_run_sr", "last5_pass_rate", "last5_epa", "since_timeout_any", "since_qend",
       "after_off_timeout", "after_def_timeout", "first_snap_after_2min", "opp_just_scored_td",
       "opp_just_scored_fg", "after_takeaway", "after_own_turnover", "after_own_3out_punt", "after_own_td",
       "after_turnover_on_downs", "scripted", "snap_no", "first_snap_2nd_half", "coach_neutral_pass",
       "coach_proe", "coach_proe_after_run", "coach_proe_after_pass", "coach_proe_after_fail",
       "coach_switch_rate", "team_neutral_pass_l8", "team_proe_l8", "opp_def_pass_epa",
       "opp_def_run_epa", "no_huddle"]
PER = ["n_rb", "n_fb", "n_te", "n_wr", "n_ol_extra", "n_def_on_off", "personnel_grp", "formation",
       "shotgun", "tell_max", "tell_min", "tell_sum", "tell_mean"]
CATS = ["prev1_type", "prev2_type", "prev_result", "personnel_grp", "formation"]
LAYERS = {"A_situation": SIT, "B_sit+cognitive": SIT + COG, "C_sit+personnel": SIT + PER,
          "D_all": SIT + COG + PER}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)


def metrics(y, p):
    return dict(n=int(len(y)), logloss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)),
                acc=float(((p > 0.5) == y).mean()), auc=float(roc_auc_score(y, p)))


def main():
    d = pd.read_parquet(FEAT)
    d = d[d.is_target == 1].reset_index(drop=True)
    for c in CATS:
        d[c] = d[c].astype("category")
    preds = []
    importances = {}
    for S in range(2019, 2027):
        tr = d[d.season < S]
        te = d[d.season == S]
        va_mask = tr.season == S - 1
        out = te[["game_id", "play_id", "season", "week", "posteam", "coach", "y", "xpass", "neutral",
                  "down", "epa", "prev_result", "has_personnel"]].copy()
        out["base"] = tr.y.mean()
        for name, cols in LAYERS.items():
            if S == 2026 and name in ("C_sit+personnel", "D_all"):
                out[name] = np.nan
                continue
            dtr = lgb.Dataset(tr.loc[~va_mask, cols], tr.loc[~va_mask, "y"])
            dva = lgb.Dataset(tr.loc[va_mask, cols], tr.loc[va_mask, "y"], reference=dtr)
            m = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            # refit on all prior seasons at the chosen iteration count
            m2 = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr["y"]), m.best_iteration)
            out[name] = m2.predict(te[cols])
            if S == 2025:
                importances[name] = dict(zip(cols, m2.feature_importance("gain").tolist()))
            print(S, name, m.best_iteration, round(log_loss(te.y, out[name]), 4), flush=True)
        preds.append(out)
    P = pd.concat(preds)
    P.to_parquet(f"{OUT}/preds.parquet", index=False)
    json.dump(importances, open(f"{OUT}/importances.json", "w"))


if __name__ == "__main__":
    main()
