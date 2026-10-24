"""Fifth pass layered on features_v4.parquet: RPO and play-action habits (experiment).

FTN tags RPO / play action / screen after the snap, so they are never read for the current play. They are used only
as history: a team's, coach's and QB's rates over prior games, and today's rates over prior snaps. FTN starts in 2022,
so these are NaN before then (unknown, not zero).
Also writes the call-type label `call` (0 run, 1 RPO kept, 2 RPO thrown, 3 play-action pass, 4 other pass) for the
call-type experiment; it is a label, never a feature.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from features_v2 import prior_game_rate

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
BASE = sys.argv[2] if len(sys.argv) > 2 else "data/features_v4.parquet"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/features_v5.parquet"
TAGS = {"is_rpo": "rpo", "is_play_action": "pa", "is_screen_pass": "scr"}
LEAGUE = {"rpo": 0.05, "pa": 0.13, "scr": 0.055, "keep": 0.73}
HABIT = ["team_rpo", "team_pa", "team_scr", "team_rpo_neu", "team_pa_neu", "team_rpo_keep", "team_last_rpo", "team_last_pa",
         "coach_rpo", "coach_pa", "qb_rpo", "qb_rpo_keep", "ingame_rpo", "ingame_pa", "ingame_rpo_keep", "prev_rpo", "prev_pa"]


def ftn_tags(raw):
    fs = []
    for s in range(2022, 2027):
        try:
            fs.append(pd.read_parquet(f"{raw}/ftn_{s}.parquet", columns=["nflverse_game_id", "nflverse_play_id", "qb_location"] + list(TAGS)))
        except FileNotFoundError:
            pass
    f = pd.concat(fs, ignore_index=True).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id", **TAGS})
    f = f[f["qb_location"].astype(str) != "0"].drop(columns="qb_location")   # '0' = not charted
    f["play_id"] = f["play_id"].astype(float)
    for c in TAGS.values():
        f[c] = f[c].astype(float)
    return f.drop_duplicates(["game_id", "play_id"])


def main():
    d = pd.read_parquet(BASE).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    d["play_id"] = d["play_id"].astype(float)
    d = d.merge(ftn_tags(RAW), on=["game_id", "play_id"], how="left")
    tgt = d["is_target"].eq(1) & d["rpo"].notna()
    ftn_era = d["season"] >= 2022
    run = d["y"].eq(0)
    d["call"] = np.select([d["rpo"].eq(1) & run, d["rpo"].eq(1), run, d["pa"].eq(1)], [1, 2, 0, 3], 4).astype(float)
    d.loc[~tgt, "call"] = np.nan

    def rate(mask, keys, k, name, val, base):
        sub = d.assign(_r=np.where(mask, val - base, 0.0), _n=mask.astype(float))
        r = prior_game_rate(sub, keys, k, name)
        return d.merge(r.drop(columns=name + "_n"), on=keys + ["game_id"], how="left")

    neu = tgt & d["neutral"].eq(1)
    rpo_play = tgt & d["rpo"].eq(1)
    for t in ["rpo", "pa", "scr"]:
        d = rate(tgt, ["posteam", "season"], 60, f"team_{t}", d[t], LEAGUE[t])
    for t in ["rpo", "pa"]:
        d = rate(neu, ["posteam", "season"], 40, f"team_{t}_neu", d[t], LEAGUE[t])
        d = rate(tgt, ["coach"], 200, f"coach_{t}", d[t], LEAGUE[t])
    d = rate(rpo_play, ["posteam", "season"], 15, "team_rpo_keep", run.astype(float), LEAGUE["keep"])
    q = d["cur_qb"].notna()
    d = rate(tgt & q, ["cur_qb"], 150, "qb_rpo", d["rpo"], LEAGUE["rpo"])
    d = rate(rpo_play & q, ["cur_qb"], 20, "qb_rpo_keep", run.astype(float), LEAGUE["keep"])

    # last season's rates
    se = d[tgt].groupby(["posteam", "season"], as_index=False)[["rpo", "pa"]].mean()
    se["season"] += 1
    d = d.merge(se.rename(columns={"rpo": "team_last_rpo", "pa": "team_last_pa"}), on=["posteam", "season"], how="left")

    # today, prior snaps only
    for t in ["rpo", "pa"]:
        d["_v"] = np.where(tgt, d[t], 0.0)
        g = d.groupby(["game_id", "posteam"], sort=False)["_v"]
        d[f"ingame_{t}"] = g.cumsum() - d["_v"]
        d[f"prev_{t}"] = g.shift(1)
    d["_k"] = np.where(rpo_play, run.astype(float), 0.0)
    d["_kn"] = rpo_play.astype(float)
    g = d.groupby(["game_id", "posteam"], sort=False)
    d["ingame_rpo_keep"] = (g["_k"].cumsum() - d["_k"] + 2 * LEAGUE["keep"]) / (g["_kn"].cumsum() - d["_kn"] + 2)

    for c in HABIT:
        d[c] = d[c].astype(float).where(ftn_era)
    d = d.drop(columns=[c for c in d.columns if c.startswith("_")] + list(TAGS.values()))
    d.to_parquet(OUT, index=False)
    t = d[d.is_target == 1]
    print(t[HABIT].describe().T[["count", "mean", "std", "min", "max"]].round(3))
    print("call labels by season:", t.groupby("season")["call"].apply(lambda s: s.notna().mean()).round(3).to_dict())


if __name__ == "__main__":
    main()
