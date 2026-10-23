"""Third pass layered on features_v2.parquet: per-coach decision biases and more pre-snap alignment.

Biases (humans are bad at being random): response to streaks, call patterns, in-game balance seeking,
win-stay/lose-shift, repeat-avoidance in the same down/distance. Coach rates use prior games only;
in-game rates use prior snaps only.
Alignment: defenders in box and defensive personnel (participation), and FTN charting (2022+):
QB alignment, backfield count, box count, pre-snap motion, hash. Post-snap FTN fields are never read.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from features_v2 import prior_game_rate

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
BASE = sys.argv[2] if len(sys.argv) > 2 else "data/features_v2.parquet"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/features_v3.parquet"
FTN_PRESNAP = {"starting_hash": "ftn_hash", "qb_location": "ftn_qb_loc", "n_offense_backfield": "ftn_backfield",
               "n_defense_box": "ftn_box", "is_motion": "ftn_motion"}
DL, LB, DB = ["DL", "DE", "DT", "NT"], ["LB", "ILB", "OLB", "MLB"], ["DB", "CB", "S", "FS", "SS"]


def defense(raw):
    parts = []
    for s in range(2016, 2027):
        try:
            p = pd.read_parquet(f"{raw}/part_{s}.parquet",
                                columns=["nflverse_game_id", "play_id", "defenders_in_box", "defense_personnel"])
            parts.append(p.rename(columns={"nflverse_game_id": "game_id"}))
        except FileNotFoundError:
            pass
    p = pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "play_id"])
    p["play_id"] = p["play_id"].astype(float)
    ex = p["defense_personnel"].fillna("").str.extractall(r"(\d+)\s+([A-Z]+)")
    ex[0] = ex[0].astype(int)
    piv = ex.reset_index().pivot_table(index="level_0", columns=1, values=0, aggfunc="sum").reindex(p.index)
    has = p["defense_personnel"].notna()
    for name, cs in [("def_n_dl", DL), ("def_n_lb", LB), ("def_n_db", DB)]:
        p[name] = sum(piv[c].fillna(0) for c in cs if c in piv.columns).where(has)
    p["def_box"] = p["defenders_in_box"]
    return p[["game_id", "play_id", "def_box", "def_n_dl", "def_n_lb", "def_n_db"]]


def ftn(raw):
    fs = []
    for s in range(2022, 2027):
        try:
            fs.append(pd.read_parquet(f"{raw}/ftn_{s}.parquet", columns=["nflverse_game_id", "nflverse_play_id"] + list(FTN_PRESNAP)))
        except FileNotFoundError:
            pass
    f = pd.concat(fs, ignore_index=True).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id", **FTN_PRESNAP})
    f = f.drop_duplicates(["game_id", "play_id"])
    f["play_id"] = f["play_id"].astype(float)
    uncharted = f["ftn_qb_loc"].eq("0")   # '0' marks plays FTN didn't chart (kicks etc.)
    for c in ["ftn_hash", "ftn_qb_loc"]:
        f[c] = f[c].where(~f[c].eq("0"))
    f["ftn_box"] = f["ftn_box"].astype(float).where(~uncharted)
    f["ftn_motion"] = f["ftn_motion"].astype(float).where(~uncharted)
    return f


def main():
    d = pd.read_parquet(BASE).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    tgt = (d["ptype"].isin(["pass", "run"]) & d["play_type"].isin(["pass", "run"]) & d["xpass"].notna()).values
    d["_r"] = np.where(tgt, (d["ptype"] == "pass").astype(float) - d["xpass"], 0.0)
    d["_n"] = tgt.astype(float)

    # ---------- the caller's recent sequence ----------
    tg = d.groupby(["game_id", "posteam"], sort=False)
    d["prev3_type"] = tg["ptype"].shift(3).fillna("none")
    short = lambda c: d[c].map({"pass": "P", "run": "R"}).fillna("x")
    d["pat3"] = short("prev3_type") + short("prev2_type") + short("prev1_type")
    d["streak_b"] = d["streak_signed"].clip(-4, 4)
    d["wsls"] = d["prev1_type"].astype(str) + "_" + d["prev_success"].fillna(-1).astype(int).astype(str)

    # in-game balance: how far today's calls have run from situation-expected (small-numbers correction)
    d["ingame_proe"] = (tg["_r"].cumsum() - d["_r"]) / (tg["_n"].cumsum() - d["_n"] + 4)
    d["bal_b"] = pd.cut(d["ingame_proe"], [-np.inf, -0.08, 0.08, np.inf], labels=["under", "even", "over"]).astype(str)
    dg = d.groupby(["game_id", "posteam", "fixed_drive"], sort=False)
    d["drive_snap_no"] = dg.cumcount() + 1
    d["drive_proe"] = (dg["_r"].cumsum() - d["_r"]) / (dg["_n"].cumsum() - d["_n"] + 2)

    # repeat avoidance: last call this game in the same down/distance bucket
    sg = d.groupby(["game_id", "posteam", "dd_bucket"], sort=False)
    d["prev_same_dd_type"] = sg["ptype"].shift(1).fillna("none")
    d["prev_same_dd_success"] = sg["success"].shift(1)

    # per-coach bias rates from prior games
    for key, name in [("streak_b", "coach_streak_proe"), ("pat3", "coach_pat3_proe"), ("bal_b", "coach_bal_proe"),
                      ("wsls", "coach_wsls_proe"), ("prev_same_dd_type", "coach_repeat_proe")]:
        r = prior_game_rate(d, ["coach", key], 150, name)
        d = d.merge(r, on=["coach", key, "game_id"], how="left")
    r = prior_game_rate(d, ["pat3"], 500, "league_pat3_proe")
    d = d.merge(r.drop(columns="league_pat3_proe_n"), on=["pat3", "game_id"], how="left")

    # ---------- pre-snap alignment ----------
    d = d.merge(defense(RAW), on=["game_id", "play_id"], how="left")
    d = d.merge(ftn(RAW), on=["game_id", "play_id"], how="left")

    d = d.drop(columns=[c for c in d.columns if c.startswith("_")])
    d.to_parquet(OUT, index=False)
    t = d[d.is_target == 1]
    new = ["coach_streak_proe", "coach_pat3_proe", "coach_bal_proe", "coach_wsls_proe", "coach_repeat_proe",
           "league_pat3_proe", "ingame_proe", "drive_proe", "def_box", "def_n_db", "ftn_box", "ftn_backfield", "ftn_motion"]
    print(t[new].describe().T[["count", "mean", "std", "min", "max"]].round(3))
    print(t.groupby("season")[["def_box", "ftn_qb_loc"]].apply(lambda g: g.notna().mean()).round(3))


if __name__ == "__main__":
    main()
