"""v3 backtest: v2 features + coach decision biases + pre-snap alignment, LightGBM (+ CatBoost) ensemble.

Every modelling decision uses seasons <= 2023 only, so 2024+ is a clean holdout.

usage: backtest_v3.py ablate          -> feature-group / model ablation on the 2023 walk-forward step
       backtest_v3.py tune [trials]   -> Optuna LightGBM search: train <=2021, early-stop on 2022
       backtest_v3.py run             -> walk-forward 2019..2026, writes data/preds_v3.parquet
       backtest_v3.py live [season]   -> re-train only that season's live_v3 model (weekly refresh), rewrite preds
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
sys.path.insert(0, os.path.dirname(__file__))
from backtest_v2 import FULL, LIVE, BASEP, CATS as CATS_V2

D = os.environ.get("DATA", "data")
PARAMS = os.environ.get("PARAMS", "src/best_params_v3.json")
BIAS = ["prev3_type", "pat3", "ingame_proe", "drive_snap_no", "drive_proe", "prev_same_dd_type", "prev_same_dd_success",
        "coach_streak_proe", "coach_streak_proe_n", "coach_pat3_proe", "coach_pat3_proe_n", "coach_bal_proe",
        "coach_wsls_proe", "coach_wsls_proe_n", "coach_repeat_proe", "league_pat3_proe"]
ALIGN = ["def_box", "def_n_dl", "def_n_lb", "def_n_db"]
FTN = ["ftn_hash", "ftn_qb_loc", "ftn_backfield", "ftn_box", "ftn_motion"]
CATS = CATS_V2 + ["prev3_type", "pat3", "prev_same_dd_type", "ftn_hash", "ftn_qb_loc"]
FULL3 = FULL + BIAS + ALIGN + FTN
LIVE3 = LIVE + BIAS + FTN  # FTN charting is published during the season; participation is not
OUTCOLS = ["game_id", "play_id", "season", "week", "posteam", "defteam", "coach", "y", "xpass",
           "neutral", "down", "ydstogo", "qtr", "epa", "has_personnel"]


def load():
    d = pd.read_parquet(f"{D}/features_v3.parquet")
    d = d[d.is_target == 1].reset_index(drop=True)
    for c in CATS:
        d[c] = d[c].astype("category")
    return d


def lgb_fit_predict(tr, va, te, cols, p, seed):
    """v2 procedure: early-stop on the last training season, refit on all training seasons at 1.1x rounds."""
    params = {**BASEP, **p, "seed": seed}
    dtr = lgb.Dataset(tr[cols], tr.y)
    m = lgb.train(params, dtr, 5000, valid_sets=[lgb.Dataset(va[cols], va.y, reference=dtr)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    full = pd.concat([tr, va])
    m2 = lgb.train(params, lgb.Dataset(full[cols], full.y), int(m.best_iteration * 1.1))
    return m2.predict(te[cols]), m2


def cat_fit_predict(tr, va, te, cols, seed=0):
    from catboost import CatBoostClassifier
    cats = [c for c in cols if c in CATS]
    prep = lambda x: x[cols].assign(**{c: x[c].astype(str) for c in cats})
    kw = dict(loss_function="Logloss", learning_rate=0.08, depth=8, l2_leaf_reg=5, thread_count=int(os.environ.get("THREADS", 6)),
              random_seed=seed, verbose=False, one_hot_max_size=4)
    m = CatBoostClassifier(iterations=5000, od_type="Iter", od_wait=100, **kw)
    m.fit(prep(tr), tr.y, cat_features=cats, eval_set=(prep(va), va.y))
    full = pd.concat([tr, va])
    m2 = CatBoostClassifier(iterations=int(m.get_best_iteration() * 1.1) + 1, **kw)
    m2.fit(prep(full), full.y, cat_features=cats)
    return m2.predict_proba(prep(te))[:, 1]


def split(d, S):
    tr = d[d.season < S - 1]
    return tr, d[d.season == S - 1], d[d.season == S]


def lgb_ens(tr, va, te, cols, plist):
    return np.mean([lgb_fit_predict(tr, va, te, cols, p, seed)[0] for seed, p in enumerate(plist)], axis=0)


def score(y, p):
    return round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 5), round(float(((p > .5) == y).mean()), 5)


def params():
    if os.path.exists(PARAMS):
        return json.load(open(PARAMS))["params"]
    return json.load(open("src/best_params.json"))["params"]


MIN_GAIN = 0.0003  # log-loss improvement on 2023 a group (or CatBoost) must show to be kept
DECISION = f"{D}/ablate_v3.json"


def chosen():
    """Column sets and CatBoost switch picked by `ablate` (all groups if it hasn't run)."""
    dec = json.load(open(DECISION)) if os.path.exists(DECISION) else {"groups": ["bias", "align", "ftn"], "cat": True}
    g = {"bias": BIAS, "align": ALIGN, "ftn": FTN}
    full = FULL + sum((g[k] for k in dec["groups"]), [])
    live = LIVE + sum((g[k] for k in dec["groups"] if k != "align"), [])
    return full, live, dec["cat"]


def ablate():
    d = load()
    tr, va, te = split(d, 2023)
    plist = params()[:1]
    log = {}
    def ev(name, cols):
        t0 = time.time()
        p = lgb_ens(tr, va, te, cols, plist)
        log[name] = score(te.y, p)
        print(name, *log[name], f"{time.time() - t0:.0f}s", flush=True)
        return p
    ev("v2_full", FULL)
    keep = []
    for k, cols in [("bias", BIAS), ("align", ALIGN), ("ftn", FTN)]:
        ev("+" + k, FULL + cols)
        if log["v2_full"][0] - log["+" + k][0] > MIN_GAIN:
            keep.append(k)
    json.dump({"groups": keep, "cat": False, "log": log}, open(DECISION, "w"), indent=1)
    full, live, _ = chosen()
    pl = ev("v3_full", full) if keep else None
    if pl is None:
        pl = lgb_ens(tr, va, te, full, plist)
    t0 = time.time()
    pc = cat_fit_predict(tr, va, te, full)
    log["catboost"], log["lgb+cat"] = score(te.y, pc), score(te.y, (pl + pc) / 2)
    print("catboost", *log["catboost"], f"{time.time() - t0:.0f}s", "| lgb+cat", *log["lgb+cat"], flush=True)
    cat = log["lgb+cat"][0] < score(te.y, pl)[0] - MIN_GAIN
    json.dump({"groups": keep, "cat": cat, "log": log}, open(DECISION, "w"), indent=1)
    print("decision: groups", keep, "catboost", cat, flush=True)


def tune(n_trials):
    import optuna
    d = load()
    full, _, _ = chosen()
    tr, va = d[d.season <= 2021], d[d.season == 2022]
    dtr = lgb.Dataset(tr[full], tr.y, free_raw_data=False, params={"feature_pre_filter": False})
    dva = lgb.Dataset(va[full], va.y, reference=dtr, free_raw_data=False)

    def objective(t):
        p = dict(learning_rate=t.suggest_float("learning_rate", 0.02, 0.08, log=True),
                 num_leaves=t.suggest_int("num_leaves", 31, 255, log=True),
                 min_data_in_leaf=t.suggest_int("min_data_in_leaf", 50, 1000, log=True),
                 feature_fraction=t.suggest_float("feature_fraction", 0.3, 0.9),
                 bagging_fraction=t.suggest_float("bagging_fraction", 0.6, 1.0),
                 lambda_l2=t.suggest_float("lambda_l2", 0.1, 50, log=True),
                 min_gain_to_split=t.suggest_float("min_gain_to_split", 0, 1),
                 cat_smooth=t.suggest_float("cat_smooth", 1, 100, log=True))
        m = lgb.train({**BASEP, **p, "seed": 0}, dtr, 5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        t.set_user_attr("iters", m.best_iteration)
        return m.best_score["valid_0"]["binary_logloss"]

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0), study_name="v3",
                             storage=f"sqlite:///{D}/optuna_v3.db", load_if_exists=True)  # resumable
    if not st.trials:
        st.enqueue_trial(params()[0])
    done = len([t for t in st.trials if t.value is not None])
    st.optimize(objective, n_trials=max(0, n_trials - done), callbacks=[lambda s, t: print(t.number, round(t.value, 5), t.params, flush=True)])
    top = sorted([t for t in st.trials if t.value is not None], key=lambda t: t.value)[:3]
    json.dump({"params": [t.params for t in top], "valid_ll": [t.value for t in top]}, open(PARAMS, "w"), indent=1)
    print("best", top[0].value, top[0].params)


def run():
    """Resumable: each (season, model) is cached in data/preds_v3/, so a killed run picks up where it stopped.
    Holdout seasons go first, with the v2 models (v2 params, no CatBoost) alongside as the baseline."""
    d = load()
    FULL3, LIVE3, use_cat = chosen()
    v2p = json.load(open("src/best_params.json"))["params"]
    jobs = [(S, "full_v3", FULL3, params(), use_cat) for S in (2024, 2025)]
    jobs += [(S, "full_v2", FULL, v2p, False) for S in (2024, 2025)]
    jobs += [(S, "live_v3", LIVE3, params(), use_cat) for S in (2024, 2025, 2026)]
    jobs += [(S, "live_v2", LIVE, v2p, False) for S in (2024, 2025, 2026)]
    jobs += [(S, "full_v3", FULL3, params(), use_cat) for S in range(2019, 2024)]
    os.makedirs(f"{D}/preds_v3", exist_ok=True)
    for S, name, cols, plist, cat in jobs:
        path = f"{D}/preds_v3/{S}_{name}.parquet"
        if os.path.exists(path):
            continue
        tr, va, te = split(d, S)
        t0 = time.time()
        p = lgb_ens(tr, va, te, cols, plist)
        if cat:
            p = (p + cat_fit_predict(tr, va, te, cols)) / 2
        te[["game_id", "play_id"]].assign(**{name: p}).to_parquet(path, index=False)
        print(S, name, *score(te.y, p), f"{time.time() - t0:.0f}s", flush=True)
    merge(d)


def live(S):
    """Weekly in-season refresh: the season-S live model trains on seasons < S, so re-fitting it only adds this
    week's plays to the predictions. Other seasons' cached predictions are reused as they are."""
    d = load()
    _, LIVE3, use_cat = chosen()
    tr, va, te = split(d, S)
    t0 = time.time()
    p = lgb_ens(tr, va, te, LIVE3, params())
    if use_cat:
        p = (p + cat_fit_predict(tr, va, te, LIVE3)) / 2
    os.makedirs(f"{D}/preds_v3", exist_ok=True)
    te[["game_id", "play_id"]].assign(live_v3=p).to_parquet(f"{D}/preds_v3/{S}_live_v3.parquet", index=False)
    print(S, "live_v3", *score(te.y, p), f"{time.time() - t0:.0f}s", flush=True)
    merge(d)


def merge(d):
    out = d.loc[d.season >= 2019, OUTCOLS].copy()
    for name in ["full_v3", "full_v2", "live_v3", "live_v2"]:
        fs = [pd.read_parquet(f"{D}/preds_v3/{f}") for f in sorted(os.listdir(f"{D}/preds_v3")) if f.endswith(f"_{name}.parquet")]
        out = out.merge(pd.concat(fs), on=["game_id", "play_id"], how="left")
    out.to_parquet(f"{D}/preds_v3.parquet", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ablate":
        ablate()
    elif cmd == "tune":
        tune(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "live":
        live(int(sys.argv[2]) if len(sys.argv) > 2 else 2026)
    else:
        run()
