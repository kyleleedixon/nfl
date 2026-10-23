"""v2 backtest: richer features, tuned LightGBM, seed/variant ensemble, recency weighting.

usage: backtest_v2.py tune            -> param search: train <=2022, early-stop on 2023, score 2024
       backtest_v2.py run             -> walk-forward 2019..2026, writes preds_v2.parquet
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
sys.path.insert(0, __import__("os").path.dirname(__file__))
from backtest_cols import SIT, COG, PER

D = "data"
NEW_PER = ["rb_tell_max", "rb_tell_mean", "rb_tell_min", "te_tell_max", "te_tell_mean", "te_tell_min",
           "wr_tell_max", "wr_tell_mean", "wr_tell_min", "min_share", "mean_share", "n_low_share", "rb_share",
           "qb_prior_snaps", "qb_tell", "qb_backup", "lineup_tell", "lineup_tell_n", "chg_n_rb", "chg_n_te",
           "chg_n_wr", "skill_players_changed", "same_lineup_as_prev", "prev_type_same_lineup",
           "team_grp_proe", "team_pf_proe", "team_pf_proe_n", "coach_pf_proe", "team_season_pf_proe",
           "league_pf_proe", "grp_usage", "ingame_grp_proe", "ingame_grp_n", "ingame_pf_proe", "ingame_pf_n",
           "prev_epa_x_same_lineup"]
NEW_COG = ["coach_dd_proe", "team_season_dd_proe", "team_season_proe"]
CATS = ["prev1_type", "prev2_type", "prev_result", "personnel_grp", "formation", "dd_bucket"]
FULL = SIT + COG + NEW_COG + PER + NEW_PER + ["dd_bucket"]
LIVE = SIT + COG + NEW_COG + ["shotgun", "dd_bucket"]  # only what play-by-play carries in-season (no participation data)

BASEP = dict(objective="binary", verbose=-1, bagging_freq=1, num_threads=6)


def load():
    d = pd.read_parquet(f"{D}/features_v2.parquet")
    d = d[d.is_target == 1].reset_index(drop=True)
    for c in CATS:
        d[c] = d[c].astype("category")
    return d


def fit(tr, va, cols, p, w=None, seed=0):
    params = {**BASEP, **p, "seed": seed}
    dtr = lgb.Dataset(tr[cols], tr.y, weight=w)
    dva = lgb.Dataset(va[cols], va.y, reference=dtr)
    m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def weights(seasons, S, half_life):
    if not half_life:
        return None
    return np.power(0.5, (S - 1 - seasons) / half_life).values


GRID = [
    dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=5),
    dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.8, lambda_l2=10),
    dict(learning_rate=0.03, num_leaves=255, min_data_in_leaf=500, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20),
    dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.7, lambda_l2=10, max_bin=127),
]


def tune():
    d = load()
    tr, va, te = d[d.season <= 2022], d[d.season == 2023], d[d.season == 2024]
    res = []
    for cols_name, cols in [("v1_full", SIT + COG + PER), ("v2_full", FULL)]:
        for i, p in enumerate(GRID if cols_name == "v2_full" else GRID[:1]):
            for hl in ([None, 3] if cols_name == "v2_full" else [None]):
                m = fit(tr, va, cols, p, weights(tr.season, 2023, hl))
                pr = m.predict(te[cols], num_iteration=m.best_iteration)
                r = dict(cols=cols_name, grid=i, half_life=hl, iters=m.best_iteration,
                         ll=round(log_loss(te.y, pr), 5), acc=round(((pr > .5) == te.y).mean(), 5))
                print(r, flush=True)
                res.append(r)
    json.dump(res, open(f"{D}/tune.json", "w"))


def run(best, hl):
    d = load()
    out_all = []
    imp = None
    for S in range(2019, 2027):
        tr = d[d.season < S]
        va_mask = (tr.season == S - 1).values
        te = d[d.season == S]
        out = te[["game_id", "play_id", "season", "week", "posteam", "defteam", "coach", "y", "xpass",
                  "neutral", "down", "ydstogo", "qtr", "epa", "has_personnel"]].copy()
        for name, cols in [("full_v2", FULL), ("live_v2", LIVE)]:
            if S == 2026 and name == "full_v2":
                out[name] = np.nan
                continue
            preds = []
            for seed, p in enumerate(best):
                w_fit = weights(tr.season[~va_mask], S - 1, hl)
                m = fit(tr[~va_mask], tr[va_mask], cols, p, w_fit, seed)
                n_it = int(m.best_iteration * 1.1)
                m2 = lgb.train({**BASEP, **p, "seed": seed}, lgb.Dataset(tr[cols], tr.y, weight=weights(tr.season, S, hl)), n_it)
                preds.append(m2.predict(te[cols]))
                if S == 2025 and name == "full_v2" and seed == 0:
                    imp = dict(zip(cols, m2.feature_importance("gain").tolist()))
            out[name] = np.mean(preds, axis=0)
            print(S, name, round(log_loss(te.y, out[name]), 4), round(((out[name] > .5) == te.y).mean(), 4), flush=True)
        out_all.append(out)
    P = pd.concat(out_all)
    P.to_parquet(f"{D}/preds_v2.parquet", index=False)
    json.dump(imp, open(f"{D}/importances_v2.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune()
    else:
        cfg = json.load(open("src/best_params.json"))
        run(cfg["params"], cfg["half_life"])
