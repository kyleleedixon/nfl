"""Fourth pass layered on features_v3.parquet: who is this team *this year*.

Error analysis on v3 found team-season bias ~6x noise: the model is slow to learn a team's current identity.
QB at the snap: play-by-play only (works in-season). The QB is the passer on this team's most recent dropback
today (prior snaps only), else last game's main QB. Per-QB rates use prior games only.
Fast-adapting team tendencies: last-3-games lean, season-to-date 1st-down / scripted-play lean, last season's lean,
new head coach; in-game 1st-down lean and the last 1st-down call today (prior snaps only).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from features_v2 import prior_game_rate

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
BASE = sys.argv[2] if len(sys.argv) > 2 else "data/features_v3.parquet"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/features_v4.parquet"
PLAYERS = "data/ref/players.parquet"
QB = ["qb_proe", "qb_proe_n", "qb_team_proe", "qb_team_proe_n", "qb_db", "qb_epa_db", "qb_run_rate",
      "qb_not_prev_main", "qb_changed_today", "qb_starts_team"]
TEAM = ["team_l3_proe", "team_season_1d_proe", "team_season_script_proe", "team_last_season_proe", "coach_new_team",
        "coach_team_games", "ingame_1d_proe", "prev_1d_type", "prev_1d_success"]
CATS = ["prev_1d_type"]


def pbp_qb(raw):
    fs = []
    for s in range(2016, 2027):
        try:
            fs.append(pd.read_parquet(f"{raw}/pbp_{s}.parquet", columns=["game_id", "play_id", "qb_dropback", "passer_id",
                                                                       "rusher_id", "qb_scramble", "qb_epa"]))
        except FileNotFoundError:
            pass
    p = pd.concat(fs, ignore_index=True).drop_duplicates(["game_id", "play_id"])
    p["play_id"] = p["play_id"].astype(float)
    return p


def prior_asof(ev, tgt, key, cols, k=None):
    """For each (key, gorder) row in tgt, sums of ev[cols] over that key's games strictly before gorder."""
    g = ev.groupby([key, "gorder"], as_index=False)[cols].sum().sort_values("gorder")
    c = g.groupby(key)[cols].cumsum()
    g[cols] = c
    t = tgt[[key, "gorder"]].dropna().drop_duplicates().sort_values("gorder")
    return pd.merge_asof(t, g, on="gorder", by=key, allow_exact_matches=False)


def main():
    d = pd.read_parquet(BASE).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    q = pbp_qb(RAW)
    d = d.merge(q[[c for c in q.columns if c not in d.columns or c in ("game_id", "play_id")]], on=["game_id", "play_id"], how="left")
    tgt = (d["ptype"].isin(["pass", "run"]) & d["play_type"].isin(["pass", "run"]) & d["xpass"].notna()).values
    d["_r"] = np.where(tgt, (d["ptype"] == "pass").astype(float) - d["xpass"], 0.0)
    d["_n"] = tgt.astype(float)
    tg = d.groupby(["game_id", "posteam"], sort=False)

    # ---------- QB at the snap ----------
    d["_qb"] = d["passer_id"].where(d["qb_dropback"] == 1)
    d["_q1"] = tg["_qb"].shift(1)
    d["cur_qb"] = d.groupby(["game_id", "posteam"], sort=False)["_q1"].ffill()
    first = d.dropna(subset=["_qb"]).drop_duplicates(["game_id", "posteam"])[["game_id", "posteam", "_qb"]]
    main_ = (d.dropna(subset=["_qb"]).groupby(["game_id", "posteam", "gorder", "_qb"], as_index=False).size()
             .sort_values(["size"]).drop_duplicates(["game_id", "posteam"], keep="last").sort_values("gorder"))
    main_["prev_main"] = main_.groupby("posteam")["_qb"].shift(1)
    d = d.merge(main_[["game_id", "posteam", "prev_main"]], on=["game_id", "posteam"], how="left")
    d = d.merge(first.rename(columns={"_qb": "first_qb"}), on=["game_id", "posteam"], how="left")
    d["prev_main"] = d.groupby("posteam")["prev_main"].ffill()  # team-game rows with no dropbacks get carried
    known = d["cur_qb"].notna()
    d["cur_qb"] = d["cur_qb"].fillna(d["prev_main"])
    d["qb_not_prev_main"] = (d["cur_qb"] != d["prev_main"]).astype(float).where(d["cur_qb"].notna() & d["prev_main"].notna())
    d["qb_changed_today"] = ((d["cur_qb"] != d["first_qb"]) & known).astype(float)
    # starts for this team before today (games as main QB)
    main_["_one"] = 1.0
    st = prior_asof(main_.rename(columns={"_qb": "cur_qb"}).assign(qt=lambda x: x.cur_qb + "|" + x.posteam),
                    d.assign(qt=d.cur_qb + "|" + d.posteam), "qt", ["_one"])
    d = d.assign(qt=d.cur_qb + "|" + d.posteam).merge(st.rename(columns={"_one": "qb_starts_team"}), on=["qt", "gorder"], how="left")
    d["qb_starts_team"] = d["qb_starts_team"].fillna(0).where(d["cur_qb"].notna())

    # play-calling lean with this QB (any team) and with this QB on this team
    for keys, k, name in [(["cur_qb"], 300, "qb_proe"), (["qt"], 200, "qb_team_proe")]:
        r = prior_game_rate(d[d["cur_qb"].notna()], keys, k, name)
        d = d.merge(r, on=keys + ["game_id"], how="left")
    # QB quality and mobility, career to date
    db = d[d["qb_dropback"] == 1].assign(_db=1.0, _e=d["qb_epa"].fillna(0))
    a = prior_asof(db.rename(columns={"passer_id": "pid"}), d.rename(columns={"cur_qb": "pid"}), "pid", ["_db", "_e"])
    pos = pd.read_parquet(PLAYERS, columns=["gsis_id", "position"]).dropna().drop_duplicates("gsis_id")
    qbs = set(pos.gsis_id[pos.position == "QB"])
    runs = d[(d["play_type"] == "run") & d["rusher_id"].isin(qbs) & (d["qb_scramble"] != 1)].assign(_qr=1.0)
    snaps = d[tgt & d["cur_qb"].notna()].assign(_s=1.0)
    rr = prior_asof(runs.rename(columns={"rusher_id": "pid"}), d.rename(columns={"cur_qb": "pid"}), "pid", ["_qr"])
    ss = prior_asof(snaps.rename(columns={"cur_qb": "pid"}), d.rename(columns={"cur_qb": "pid"}), "pid", ["_s"])
    a = a.merge(rr, on=["pid", "gorder"], how="outer").merge(ss, on=["pid", "gorder"], how="outer").fillna(0)
    a["qb_db"] = a["_db"]
    a["qb_epa_db"] = a["_e"] / (a["_db"] + 150)
    a["qb_run_rate"] = (a["_qr"] + 0.03 * 200) / (a["_s"] + 200)
    d = d.merge(a[["pid", "gorder", "qb_db", "qb_epa_db", "qb_run_rate"]].rename(columns={"pid": "cur_qb"}),
                on=["cur_qb", "gorder"], how="left")

    # ---------- this year's team, fast ----------
    games = d.drop_duplicates(["game_id", "posteam"])[["game_id", "posteam", "season", "gorder"]].dropna()
    gs = d[tgt].groupby(["game_id", "posteam"])[["_r", "_n"]].sum().reset_index()
    gs = games.merge(gs, on=["game_id", "posteam"], how="left").fillna({"_r": 0, "_n": 0}).sort_values("gorder")
    roll = gs.groupby("posteam")[["_r", "_n"]].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    gs["team_l3_proe"] = roll["_r"] / (roll["_n"] + 50)
    se = gs.groupby(["posteam", "season"], as_index=False)[["_r", "_n"]].sum()
    se["season"] += 1
    se["team_last_season_proe"] = se["_r"] / (se["_n"] + 100)
    gs = gs.merge(se[["posteam", "season", "team_last_season_proe"]], on=["posteam", "season"], how="left")
    d = d.merge(gs[["game_id", "posteam", "team_l3_proe", "team_last_season_proe"]], on=["game_id", "posteam"], how="left")
    neu = d["neutral"] == 1
    for mask, k, name in [((d["down"] == 1) & neu, 40, "team_season_1d_proe"), (d["scripted"] == 1, 40, "team_season_script_proe")]:
        sub = d[mask]
        r = prior_game_rate(sub, ["posteam", "season"], k, name)
        d = d.merge(r.drop(columns=name + "_n"), on=["posteam", "season", "game_id"], how="left")
        d[name] = d.groupby(["posteam", "season"])[name].ffill().fillna(0)  # games' rows outside the mask
    cg = games.merge(d.drop_duplicates(["game_id", "posteam"])[["game_id", "posteam", "coach"]], on=["game_id", "posteam"])
    cg = cg.sort_values("gorder")
    cg["coach_team_games"] = cg.groupby(["coach", "posteam"]).cumcount()
    first_season = cg.groupby(["coach", "posteam"])["season"].transform("min")
    cg["coach_new_team"] = ((cg["season"] == first_season) & (cg["season"] > 2016)).astype(float)
    d = d.merge(cg[["game_id", "posteam", "coach_team_games", "coach_new_team"]], on=["game_id", "posteam"], how="left")

    # in-game 1st downs (prior snaps)
    fd = (d["down"] == 1) & tgt
    d["_fr"], d["_fn"] = np.where(fd, d["_r"], 0.0), fd.astype(float)
    tg = d.groupby(["game_id", "posteam"], sort=False)
    d["ingame_1d_proe"] = (tg["_fr"].cumsum() - d["_fr"]) / (tg["_fn"].cumsum() - d["_fn"] + 3)
    d["_fd_type"] = d["ptype"].where(fd)
    d["_fd_succ"] = d["success"].where(fd)
    d["prev_1d_type"] = tg["_fd_type"].shift(1)
    d["prev_1d_success"] = tg["_fd_succ"].shift(1)
    tg = d.groupby(["game_id", "posteam"], sort=False)
    d["prev_1d_type"] = tg["prev_1d_type"].ffill().fillna("none")
    d["prev_1d_success"] = tg["prev_1d_success"].ffill()

    drop = [c for c in d.columns if c.startswith("_")] + ["qb_dropback", "passer_id", "rusher_id",
                                                          "qb_epa", "prev_main", "first_qb", "qt"]
    d = d.drop(columns=drop)
    d.to_parquet(OUT, index=False)
    t = d[d.is_target == 1]
    print(t[[c for c in QB + TEAM if c not in CATS]].describe().T[["count", "mean", "std", "min", "max"]].round(3))
    print(t.groupby("season")["cur_qb"].apply(lambda s: s.notna().mean()).round(3).to_dict())


if __name__ == "__main__":
    main()
