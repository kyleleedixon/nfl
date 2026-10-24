"""Aggregate-only results for the public showcase. Nothing here names a feature."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from season import current

D = os.environ.get("DATA", "data")
P = pd.read_parquet(f"{D}/{os.environ.get('PREDS', 'preds_v3')}.parquet")
M, LV = os.environ.get("MODEL", "full_v3"), os.environ.get("LIVE", "live_v3")
P = P.rename(columns={M: "model", LV: "live"})
OUT = "site/results.json"


def acc(y, p):
    return float(((p > 0.5) == y).mean())


def ll(y, p):
    return float(log_loss(y, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1]))


# historical = every completed season with the full (personnel) model; live = the season in progress
NOW = current()
FULL_SEASONS = sorted(int(s) for s in P.season[P.model.notna()].unique())   # full model runs only after a season (season_close.sh)
bt = P[P.season.isin(FULL_SEASONS)].copy()
base_rate = {S: float(P[P.season < S].y.mean()) if (P.season < S).any() else 0.6 for S in P.season.unique()}
R = {}
R["n_plays"] = int(len(bt))
R["n_games"] = int(bt.game_id.nunique())
R["seasons"] = f"{FULL_SEASONS[0]}–{FULL_SEASONS[-1]}"
R["headline"] = dict(
    acc=round(acc(bt.y, bt.model), 4), acc_public=round(acc(bt.y, bt.xpass), 4),
    acc_always_pass=round(float(bt.y.mean()), 4),
    acc_neutral=round(acc(bt[bt.neutral == 1].y, bt[bt.neutral == 1].model), 4),
    acc_neutral_public=round(acc(bt[bt.neutral == 1].y, bt[bt.neutral == 1].xpass), 4),
    ll=round(ll(bt.y, bt.model), 4), ll_public=round(ll(bt.y, bt.xpass), 4),
    auc=round(float(roc_auc_score(bt.y, bt.model)), 4), auc_public=round(float(roc_auc_score(bt.y, bt.xpass)), 4),
)
# error reduction: share of public model's misses we get right
miss_pub = ((bt.xpass > .5) != bt.y)
R["headline"]["miss_reduction"] = round(1 - ((bt.model > .5) != bt.y).sum() / miss_pub.sum(), 4)

R["by_season"] = [dict(season=int(s), acc=round(acc(g.y, g.model), 4), acc_public=round(acc(g.y, g.xpass), 4),
                       acc_always_pass=round(max(g.y.mean(), 1 - g.y.mean()), 4), plays=int(len(g)))
                  for s, g in bt.groupby("season")]

sits = {
    "1st down": bt.down == 1, "2nd down": bt.down == 2, "3rd down": bt.down == 3,
    "Neutral game state": bt.neutral == 1,
    "Red zone": None, "Final 2 min of half": None,
}
F = pd.read_parquet(f"{D}/features_v2.parquet", columns=["game_id", "play_id", "yardline_100", "half_seconds_remaining"])
bt = bt.merge(F, on=["game_id", "play_id"], how="left")
sits["Red zone"] = bt.yardline_100 <= 20
sits["Final 2 min of half"] = bt.half_seconds_remaining <= 120
sits["1st down"], sits["2nd down"], sits["3rd down"], sits["Neutral game state"] = \
    bt.down == 1, bt.down == 2, bt.down == 3, bt.neutral == 1
R["by_situation"] = [dict(situation=k, acc=round(acc(bt[m].y, bt[m].model), 4),
                          acc_public=round(acc(bt[m].y, bt[m].xpass), 4), plays=int(m.sum()))
                     for k, m in sits.items()]

# confidence tiers
conf = np.maximum(bt.model, 1 - bt.model)
tiers = [(0.9, 1.01, "90%+"), (0.8, 0.9, "80–90%"), (0.7, 0.8, "70–80%"), (0.6, 0.7, "60–70%"), (0.5, 0.6, "50–60%")]
R["confidence"] = []
for lo, hi, lab in tiers:
    m = (conf >= lo) & (conf < hi)
    R["confidence"].append(dict(tier=lab, share=round(float(m.mean()), 4), hit=round(acc(bt[m].y, bt[m].model), 4),
                                plays=int(m.sum())))
m = conf >= 0.8
R["headline"]["hi_conf_share"] = round(float(m.mean()), 4)
R["headline"]["hi_conf_hit"] = round(acc(bt[m].y, bt[m].model), 4)
mp = np.maximum(bt.xpass, 1 - bt.xpass) >= 0.8
R["headline"]["hi_conf_share_public"] = round(float(mp.mean()), 4)

# calibration
bins = np.linspace(0, 1, 11)
bt["b"] = pd.cut(bt.model, bins, include_lowest=True)
cal = bt.groupby("b", observed=True).agg(pred=("model", "mean"), actual=("y", "mean"), plays=("y", "size")).reset_index(drop=True)
R["calibration"] = cal.round(4).to_dict("records")

# team predictability, latest completed season (full model, else the live model if participation isn't out yet)
LAST = NOW - 1 if (P.season == NOW - 1).any() else FULL_SEASONS[-1]
sl = P[P.season == LAST].assign(m=lambda x: x.model.fillna(x.live))
R["teams_last"] = dict(season=int(LAST), teams=sorted([dict(team=t, acc=round(acc(g.y, g.m), 4), acc_public=round(acc(g.y, g.xpass), 4),
                       plays=int(len(g))) for t, g in sl.groupby("posteam")], key=lambda r: -r["acc"]))

# season in progress, live model (runs without participation data), by week
s26 = P[(P.season == NOW) & P.live.notna()]
R["live"] = dict(season=NOW, acc=round(acc(s26.y, s26.live), 4) if len(s26) else None,
                      acc_public=round(acc(s26.y, s26.xpass), 4) if len(s26) else None,
                      plays=int(len(s26)), through_week=int(s26.week.max()) if len(s26) else 0,
                      weeks=[dict(week=int(w), acc=round(acc(g.y, g.live), 4), acc_public=round(acc(g.y, g.xpass), 4),
                                  plays=int(len(g))) for w, g in s26.groupby("week")])
lb = bt[bt.live.notna()]
R["live_backtest"] = dict(acc=round(acc(lb.y, lb.live), 4), seasons=sorted(int(x) for x in lb.season.unique()))
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps({k: R[k] for k in ["headline", "by_season", "by_situation", "confidence", "live", "live_backtest"]}, indent=1))
