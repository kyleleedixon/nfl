"""Fast feature screen: one LightGBM config, walk-forward steps 2021-2023 (never 2024+).

usage: screen.py arm[,arm...]   arms: base, qb, team, all, rpo (needs FEAT=data/features_v5.parquet)  (each arm x {full, live} x {2021, 2022, 2023})
Test predictions go to data/screen/{S}_{model}_{arm}.parquet (resumable); `screen.py report` compares arms with a
paired bootstrap over games.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from backtest_v3 import chosen, CATS, split, lgb_fit_predict, params, score
from features_v4 import QB, TEAM, CATS as CATS4
from features_v5 import HABIT

D = os.environ.get("DATA", "data")
FEAT = os.environ.get("FEAT", f"{D}/features_v4.parquet")
OUT = os.environ.get("OUT", f"{D}/screen")
SEASONS = [int(s) for s in os.environ.get("SEASONS", "2021,2022,2023").split(",")]


def load():
    d = pd.read_parquet(FEAT)
    d = d[d.is_target == 1].reset_index(drop=True)
    for c in CATS + CATS4:
        if c in d:
            d[c] = d[c].astype("category")
    return d


def arms():
    full, live, _ = chosen()
    callers = [c for c in pd.read_parquet(FEAT).columns if c.startswith("caller")] if "both" in FEAT else []
    add = {"base": [], "qb": QB, "team": TEAM, "all": QB + TEAM, "rpo": HABIT, "caller": callers}
    return {(m, a): base + extra for a, extra in add.items() for m, base in [("full", full), ("live", live)]}


def run(names):
    d = load()
    p0 = params()[:1]
    os.makedirs(OUT, exist_ok=True)
    for S in SEASONS:
        tr, va, te = split(d, S)
        for (m, a), cols in arms().items():
            path = f"{OUT}/{S}_{m}_{a}.parquet"
            if a not in names or os.path.exists(path):
                continue
            t0 = time.time()
            p = lgb_fit_predict(tr, va, te, cols, p0[0], 0)[0]
            te[["game_id", "play_id"]].assign(p=p).to_parquet(path, index=False)
            print(S, m, a, *score(te.y, p), f"{time.time() - t0:.0f}s", flush=True)


def report():
    d = load()[["game_id", "play_id", "season", "y"]]
    rng = np.random.default_rng(0)
    for m in ["full", "live"]:
        got = {}
        for a in ["base", "qb", "team", "all", "rpo", "caller"]:
            fs = [f"{OUT}/{S}_{m}_{a}.parquet" for S in SEASONS]
            if all(os.path.exists(f) for f in fs):
                got[a] = d.merge(pd.concat(pd.read_parquet(f) for f in fs), on=["game_id", "play_id"])
        if "base" not in got:
            continue
        b = got["base"]
        ll = lambda x: -(x.y * np.log(x.p.clip(1e-6)) + (1 - x.y) * np.log((1 - x.p).clip(1e-6)))
        b["l"] = ll(b)
        for a, x in got.items():
            x = x.merge(b[["game_id", "play_id", "l"]].rename(columns={"l": "lb"}), on=["game_id", "play_id"])
            x["g"] = x["lb"] - ll(x)
            per = x.groupby("game_id")["g"].agg(["sum", "size"])
            boots = [per["sum"].values[i].sum() / per["size"].values[i].sum()
                     for i in (rng.integers(0, len(per), len(per)) for _ in range(500))]
            seas = x.groupby("season")["g"].mean().round(4).to_dict()
            print(f"{m:4} {a:5} ll {ll(x).mean():.4f} acc {((x.p > .5) == x.y).mean():.4f} gain {x.g.mean():+.4f} "
                  f"CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}] by season {seas}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        run(sys.argv[1].split(","))
