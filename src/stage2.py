"""Stage-2 correction: nudge a model's probability by how wrong it has been about this team lately.

X1 = this team's season-to-date residual (y - p) from prior games, shrunk: sum / (n + K_SEASON)
X2 = this team's in-game residual from prior snaps today, shrunk: sum / (n + K_GAME)
p' = sigmoid(a * X1 + b * X2 + c * logit(p)). Both inputs are known before the snap.

usage: stage2.py fit MODEL S1-S2      -> fit (a, b, c) on those seasons of data/preds_v3.parquet, write src/stage2.json
       stage2.py eval MODEL S1-S2     -> score the frozen coefficients on those seasons
"""
import json
import sys
import numpy as np
import pandas as pd

K_SEASON, K_GAME = 300, 40
COEF = "src/stage2.json"


def inputs(df, pcol):
    """df sorted or not; needs game_id, play_id, season, posteam, y and pcol. Returns X (n x 3) aligned to df."""
    d = df[["game_id", "play_id", "season", "posteam", "y", pcol]].copy()
    d["_o"] = np.arange(len(d))
    d = d.sort_values(["game_id", "play_id"])
    d["r"] = d["y"] - d[pcol]
    g = d.groupby(["game_id", "posteam"], sort=False)
    x2 = (g["r"].cumsum() - d["r"]) / (g.cumcount() + K_GAME)
    gs = d.groupby(["season", "posteam", "game_id"], as_index=False, sort=True).agg(r=("r", "sum"), n=("r", "size"))
    sg = gs.groupby(["season", "posteam"])
    gs["x1"] = (sg["r"].cumsum() - gs["r"]) / (sg["n"].cumsum() - gs["n"] + K_SEASON)
    d = d.assign(x2=x2).merge(gs[["game_id", "posteam", "x1"]], on=["game_id", "posteam"], how="left")
    d = d.sort_values("_o")
    p = d[pcol].clip(1e-6, 1 - 1e-6)
    return np.column_stack([d["x1"].values, d["x2"].values, np.log(p / (1 - p)).values])


def apply(df, pcol, coef=None):
    c = np.array(coef or json.load(open(COEF))["coef"])
    return 1 / (1 + np.exp(-inputs(df, pcol) @ c))


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


if __name__ == "__main__":
    cmd, model, rng = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b = map(int, rng.split("-"))
    d = pd.read_parquet(sys.argv[4] if len(sys.argv) > 4 else "data/preds_v3.parquet")
    d = d[d.season.between(a, b) & d[model].notna()]
    if cmd == "fit":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1e3, fit_intercept=False, max_iter=1000).fit(inputs(d, model), d.y)
        json.dump({"coef": m.coef_[0].round(4).tolist(), "model": model, "seasons": rng,
                   "k": [K_SEASON, K_GAME]}, open(COEF, "w"), indent=1)
        print("coef", m.coef_[0].round(4))
    for s, x in d.groupby("season"):
        q = apply(x, model)
        print(s, f"ll {ll(x.y, x[model]):.4f} -> {ll(x.y, q):.4f}  acc {((x[model] > .5) == x.y).mean():.4f} -> {((q > .5) == x.y).mean():.4f}")
