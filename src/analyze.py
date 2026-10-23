"""Summarise backtest predictions into dashboard-ready JSON."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

D = "data"
P = pd.read_parquet(f"{D}/preds.parquet")
F = pd.read_parquet(f"{D}/features.parquet")
F = F[F.is_target == 1]
P = P.merge(F[["game_id", "play_id", "after_off_timeout", "after_def_timeout", "scripted", "first_of_drive",
               "opp_just_scored_td", "after_own_turnover", "after_takeaway", "after_own_3out_punt",
               "first_snap_after_2min", "first_snap_2nd_half", "ingame_hot_hand", "streak_signed",
               "no_huddle", "shotgun", "personnel_grp", "score_differential", "qtr"]],
            on=["game_id", "play_id"], how="left")
L = ["base", "xpass", "A_situation", "B_sit+cognitive", "C_sit+personnel", "D_all"]
R = {}


def met(df, col):
    y, p = df.y.values, df[col].clip(1e-4, 1 - 1e-4).values
    return dict(logloss=round(float(log_loss(y, p, labels=[0, 1])), 4),
                acc=round(float(((p > 0.5) == y).mean()), 4),
                auc=round(float(roc_auc_score(y, p)), 4) if col != "base" else 0.5)


bt = P[P.season.between(2019, 2025)]
R["overall"] = {c: met(bt, c) for c in L}
R["neutral"] = {c: met(bt[bt.neutral == 1], c) for c in L}
R["by_down"] = {int(dn): {c: met(bt[bt.down == dn], c) for c in ["base", "A_situation", "B_sit+cognitive", "D_all"]}
                for dn in [1, 2, 3]}
R["by_season"] = {int(s): {c: met(g, c) for c in L if g[c].notna().all()} for s, g in P.groupby("season")}
R["n_plays"] = int(len(bt))
R["pass_rate"] = round(float(bt.y.mean()), 4)

# calibration of D
bt = bt.copy()
bt["bin"] = pd.cut(bt["D_all"], np.linspace(0, 1, 11), include_lowest=True)
cal = bt.groupby("bin", observed=True).agg(pred=("D_all", "mean"), actual=("y", "mean"), n=("y", "size"))
R["calibration"] = cal.round(4).reset_index(drop=True).to_dict("records")

# human tells: pass rate over what the SITUATION model expects, by previous play result
bt["resid_A"] = bt.y - bt["A_situation"]
def tell(mask_or_group):
    g = mask_or_group
    se = np.sqrt((g["A_situation"] * (1 - g["A_situation"])).sum()) / len(g)
    return dict(proe=round(float(g.resid_A.mean()), 4), se=round(float(se), 4), n=int(len(g)),
                actual=round(float(g.y.mean()), 4), expected=round(float(g["A_situation"].mean()), 4))
nb = bt[(bt.down.isin([1, 2, 3]))]
R["after_result"] = {k: tell(g) for k, g in nb.groupby("prev_result") if len(g) > 2000}
R["after_result_neutral"] = {k: tell(g) for k, g in bt[bt.neutral == 1].groupby("prev_result") if len(g) > 2000}
flags = ["after_off_timeout", "after_def_timeout", "scripted", "first_of_drive", "opp_just_scored_td",
         "after_own_turnover", "after_takeaway", "after_own_3out_punt", "first_snap_after_2min",
         "first_snap_2nd_half", "no_huddle"]
R["moments"] = {f: tell(bt[bt[f] == 1]) for f in flags}
bt["hot_bin"] = pd.cut(bt.ingame_hot_hand, [-9, -0.5, -0.2, 0.2, 0.5, 9])
R["hot_hand"] = {str(k): tell(g) for k, g in bt[bt.neutral == 1].groupby("hot_bin", observed=True)}
bt["streak_bin"] = bt.streak_signed.clip(-4, 4)
R["streak"] = {int(k): tell(g) for k, g in bt[bt.neutral == 1].groupby("streak_bin")}

# where does each layer's lift come from: per-play logloss improvement B vs A by down
# coach predictability 2023-2025
rc = P[P.season.between(2023, 2025)].copy()
def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
rc["ll_A"], rc["ll_B"], rc["ll_D"], rc["ll_C"] = (ll(rc.y, rc[c]) for c in
                                                ["A_situation", "B_sit+cognitive", "D_all", "C_sit+personnel"])
rc["hit_D"] = ((rc.D_all > .5) == rc.y)
co = rc.groupby("coach").agg(n=("y", "size"), acc_D=("hit_D", "mean"), ll_A=("ll_A", "mean"),
                             ll_B=("ll_B", "mean"), ll_C=("ll_C", "mean"), ll_D=("ll_D", "mean"),
                             neutral_pass=("y", "mean"))
co = co[co.n >= 1500]
co["cog_lift_pct"] = (co.ll_A - co.ll_B) / co.ll_A * 100
co["cog_over_personnel_pct"] = (co.ll_C - co.ll_D) / co.ll_C * 100
co = co.sort_values("acc_D", ascending=False)
R["coaches"] = co.round(4).reset_index().to_dict("records")

# exploitability: EPA when the call matched vs surprised the model (neutral situations)
nt = bt[(bt.neutral == 1) & bt.epa.notna()].copy()
nt["pbin"] = pd.cut(nt.D_all, [0, .2, .4, .6, .8, 1])
ex = nt.groupby(["pbin", "y"], observed=True).agg(epa=("epa", "mean"), n=("epa", "size")).reset_index()
ex["pbin"] = ex.pbin.astype(str)
R["exploit"] = ex.round(4).to_dict("records")

# importance grouped by layer for D (2025 model)
imp = json.load(open(f"{D}/importances.json"))["D_all"]
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from backtest_cols import SIT, COG, PER
tot = sum(imp.values())
R["importance_groups"] = {"Situation": round(sum(imp[c] for c in SIT) / tot, 4),
                          "Human/cognitive": round(sum(imp[c] for c in COG) / tot, 4),
                          "Personnel/formation": round(sum(imp[c] for c in PER) / tot, 4)}
R["importance_top"] = [dict(f=k, share=round(v / tot, 4)) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:20]]
impB = json.load(open(f"{D}/importances.json"))["B_sit+cognitive"]
totB = sum(impB.values())
R["importance_cog_top"] = [dict(f=k, share=round(v / totB, 4)) for k, v in
                           sorted(((k, v) for k, v in impB.items() if k in COG), key=lambda x: -x[1])[:15]]

json.dump(R, open(f"{D}/results.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["overall", "neutral", "by_down", "importance_groups"]}, indent=1))
print(pd.DataFrame(R["after_result"]).T.sort_values("proe"))
print(pd.DataFrame(R["moments"]).T)
print(pd.DataFrame(R["hot_hand"]).T); print(pd.DataFrame(R["streak"]).T)
print(co[["n", "acc_D", "cog_lift_pct", "cog_over_personnel_pct"]].round(3).to_string())
print(pd.DataFrame(R["exploit"]))
print(R["importance_cog_top"])
