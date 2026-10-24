"""Defense: will they blitz? Predicts, before the snap, whether the defense sends extra rushers (FTN n_blitzers >= 1).

FTN charts blitzes on dropbacks only (2022+, published in-season), so models train and are graded on dropbacks. The
prediction itself uses only what is knowable before the snap: situation, the offense's pre-snap look (FTN QB alignment,
backfield, motion, box; personnel in the full model), the defense's habits from prior games and prior snaps today, and
the QB they face. The play-caller is the defensive coordinator: same rules as the offense side.

usage: defense.py build              -> data/defense.parquet (features + target, every season with FTN)
       defense.py run [S ...]        -> walk-forward (train < S), default 2023; preds to data/defense_preds/
       defense.py report [S ...]     -> scores vs baselines
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v2 import prior_game_rate

D = os.environ.get("DATA", "data")
RAW = f"{D}/raw"
OUT = f"{D}/defense.parquet"
PRED = f"{D}/defense_preds"
LEAGUE = 0.27   # share of dropbacks with a blitz, 2022-25

SIT = ["down", "ydstogo", "yardline_100", "score_differential", "half_seconds_remaining", "game_seconds_remaining", "qtr",
       "wp", "goal_to_go", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "spread_line", "xpass", "no_huddle"]
LOOK = ["shotgun", "ftn_qb_loc", "ftn_backfield", "ftn_motion", "ftn_box", "ftn_hash"]   # FTN pre-snap, in-season
PERS = ["n_rb", "n_te", "n_wr", "def_n_dl", "def_n_lb", "def_n_db", "def_box"]           # participation, after season
HABIT = ["d_blitz", "d_blitz_n", "d_blitz_3rd", "d_blitz_neu", "d_blitz_last", "d_blitz_l3",
         "d_ingame", "d_ingame_n", "d_prev_blitz", "d_since_blitz", "d_prev_blitz_epa", "d_prev_epa_against"]
OPP = ["o_vs_blitz_epa", "o_blitzed", "qb_db", "qb_starts_team", "qb_epa_db", "qb_changed_today"]
CATS = ["ftn_qb_loc", "ftn_hash"]
MODELS = {"situation": SIT, "live": SIT + LOOK + HABIT + OPP, "full": SIT + LOOK + PERS + HABIT + OPP}


def ftn_blitz():
    fs = []
    for s in range(2022, 2027):
        try:
            fs.append(pd.read_parquet(f"{RAW}/ftn_{s}.parquet", columns=["nflverse_game_id", "nflverse_play_id", "qb_location", "n_blitzers"]))
        except FileNotFoundError:
            pass
    f = pd.concat(fs, ignore_index=True).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f = f[f.qb_location.astype(str) != "0"].drop(columns="qb_location").drop_duplicates(["game_id", "play_id"])
    f["play_id"] = f.play_id.astype(float)
    return f


def build():
    cols = ["game_id", "play_id", "season", "week", "posteam", "defteam", "gorder", "is_target", "neutral", "epa", "ptype",
            "cur_qb", "qb_dropback_flag"] + SIT + LOOK + PERS + [c for c in OPP if c.startswith("qb_")]
    src = pd.read_parquet(f"{D}/features_v5.parquet", columns=[c for c in cols if c != "qb_dropback_flag"])
    src = src[src.season >= 2021].sort_values(["game_id", "play_id"]).reset_index(drop=True)
    src["play_id"] = src.play_id.astype(float)
    db = pd.concat([pd.read_parquet(f"{RAW}/pbp_{s}.parquet", columns=["game_id", "play_id", "qb_dropback"])
                    for s in range(2021, 2027) if os.path.exists(f"{RAW}/pbp_{s}.parquet")])
    db["play_id"] = db.play_id.astype(float)
    d = src.merge(db, on=["game_id", "play_id"], how="left").merge(ftn_blitz(), on=["game_id", "play_id"], how="left")
    dropback = d.qb_dropback.eq(1) & d.is_target.eq(1) & d.n_blitzers.notna()
    d["blitz"] = np.where(dropback, (d.n_blitzers >= 1).astype(float), np.nan)
    d["dropback"] = dropback

    def rate(mask, keys, k, name, val, base=LEAGUE):
        sub = d.assign(_r=np.where(mask, val - base, 0.0), _n=mask.astype(float))
        r = prior_game_rate(sub, keys, k, name)
        return r

    b = d.blitz.fillna(0)
    for mask, k, name in [(dropback, 60, "d_blitz"), (dropback & d.down.eq(3), 30, "d_blitz_3rd"), (dropback & d.neutral.eq(1), 40, "d_blitz_neu")]:
        r = rate(mask, ["defteam", "season"], k, name, b)
        d = d.merge(r if name == "d_blitz" else r.drop(columns=name + "_n"), on=["defteam", "season", "game_id"], how="left")
    # last season and last three games (defenses change coordinators between seasons, so season-keyed)
    gm = d[dropback].groupby(["defteam", "season", "gorder", "game_id"], as_index=False).agg(bz=("blitz", "sum"), n=("blitz", "size")).sort_values("gorder")
    roll = gm.groupby("defteam")[["bz", "n"]].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    gm["d_blitz_l3"] = (roll.bz + LEAGUE * 20) / (roll.n + 20)
    d = d.merge(gm[["game_id", "defteam", "d_blitz_l3"]], on=["game_id", "defteam"], how="left")
    se = d[dropback].groupby(["defteam", "season"], as_index=False).blitz.mean().rename(columns={"blitz": "d_blitz_last"})
    se["season"] += 1
    d = d.merge(se, on=["defteam", "season"], how="left")
    # today, prior snaps (defense's view)
    d["_b"] = np.where(dropback, d.blitz, 0.0)
    d["_n"] = dropback.astype(float)
    g = d.groupby(["game_id", "defteam"], sort=False)
    d["d_ingame_n"] = g._n.cumsum() - d._n
    d["d_ingame"] = (g._b.cumsum() - d._b + LEAGUE * 4) / (d.d_ingame_n + 4)
    d["_last"] = d.blitz.where(dropback)
    d["d_prev_blitz"] = g._last.shift(1)
    d["d_prev_blitz"] = d.groupby(["game_id", "defteam"], sort=False).d_prev_blitz.ffill()
    d["_be"] = d.epa.where(dropback & d.blitz.eq(1))   # how the last blitz went (offense EPA: high = it got burned)
    d["d_prev_blitz_epa"] = d.groupby(["game_id", "defteam"], sort=False)._be.shift(1)
    d["d_prev_blitz_epa"] = d.groupby(["game_id", "defteam"], sort=False).d_prev_blitz_epa.ffill()
    d["d_prev_epa_against"] = g.epa.shift(1)
    d["_since"] = dropback.astype(int)
    cnt = d.groupby(["game_id", "defteam"], sort=False)._since.cumsum()
    lastb = (cnt.where(dropback & d.blitz.eq(1))).groupby([d.game_id, d.defteam]).ffill()
    d["d_since_blitz"] = (cnt - d._since) - lastb.groupby([d.game_id, d.defteam]).shift(1).fillna(0)
    # the offense: how it has fared against blitzes, how often it gets blitzed (prior games)
    e = d.blitz.eq(1) & dropback
    r = prior_game_rate(d.assign(_r=np.where(e, d.epa.fillna(0), 0.0), _n=e.astype(float)), ["posteam", "season"], 30, "o_vs_blitz_epa")
    d = d.merge(r.drop(columns="o_vs_blitz_epa_n"), on=["posteam", "season", "game_id"], how="left")
    r = prior_game_rate(d.assign(_r=np.where(dropback, b - LEAGUE, 0.0), _n=dropback.astype(float)), ["posteam", "season"], 60, "o_blitzed")
    d = d.merge(r.drop(columns="o_blitzed_n"), on=["posteam", "season", "game_id"], how="left")
    for c in HABIT + ["o_vs_blitz_epa", "o_blitzed"]:
        d[c] = d[c].astype(float).where(d.season >= 2022)
    d = d[d.season >= 2022].drop(columns=[c for c in d.columns if c.startswith("_")])
    d.to_parquet(OUT, index=False)
    t = d[d.dropback]
    print(f"{len(t)} dropbacks with a blitz label; blitz rate by season:", t.groupby("season").blitz.mean().round(3).to_dict())
    print(t[HABIT + OPP].describe().T[["count", "mean", "std"]].round(3))


def load():
    d = pd.read_parquet(OUT)
    d = d[d.dropback].reset_index(drop=True)
    for c in CATS:
        d[c] = d[c].astype("category")
    return d


P0 = dict(objective="binary", verbose=-1, learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=int(os.environ.get("THREADS", 6)))


def fit_predict(tr, te, cols, seed=0):
    """Early-stop on the last quarter of the latest training season's weeks, then refit on everything at 1.1x."""
    last = tr.season.max()
    cut = tr[tr.season == last].week.quantile(0.75)
    va_m = (tr.season == last) & (tr.week > cut)
    p = {**P0, "seed": seed}
    dtr = lgb.Dataset(tr.loc[~va_m, cols], tr.blitz[~va_m])
    m = lgb.train(p, dtr, 5000, valid_sets=[lgb.Dataset(tr.loc[va_m, cols], tr.blitz[va_m], reference=dtr)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m2 = lgb.train(p, lgb.Dataset(tr[cols], tr.blitz), int(m.best_iteration * 1.1))
    return m2.predict(te[cols])


def run(seasons):
    d = load()
    os.makedirs(PRED, exist_ok=True)
    for S in seasons:
        tr, te = d[d.season < S], d[d.season == S]
        out = te[["game_id", "play_id", "season", "week", "defteam", "posteam", "blitz", "down", "ydstogo", "neutral", "epa"]].copy()
        for name, cols in MODELS.items():
            t0 = time.time()
            out[name] = np.mean([fit_predict(tr, te, cols, s) for s in range(3)], axis=0)
            print(S, name, *score(te.blitz, out[name]), f"{time.time() - t0:.0f}s", flush=True)
        out.to_parquet(f"{PRED}/{S}.parquet", index=False)


def score(y, p):
    return (f"ll {log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}", f"auc {roc_auc_score(y, p):.3f}", f"acc {((p > .5) == y).mean():.3f}")


def report(seasons):
    for S in seasons:
        o = pd.read_parquet(f"{PRED}/{S}.parquet")
        base = o.blitz.mean()
        print(f"\n{S}: {len(o)} dropbacks, blitz rate {base:.1%}; 'never blitz' is right {1 - base:.1%}; "
              f"base-rate log loss {log_loss(o.blitz, np.full(len(o), base)):.4f}")
        for m in MODELS:
            p = o[m]
            flag = p >= 0.5
            print(f"  {m:9s} {' '.join(score(o.blitz, p))}  said blitz on {flag.mean():.1%} of dropbacks, right {o.blitz[flag].mean():.0%} of those, "
                  f"caught {flag[o.blitz == 1].mean():.0%} of blitzes; top-10% most likely: {o.blitz[p >= p.quantile(.9)].mean():.0%} blitzed")


if __name__ == "__main__":
    cmd, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    {"build": lambda: build(), "run": lambda: run(args or [2023]), "report": lambda: report(args or [2023])}[cmd]()
