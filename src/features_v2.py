"""Second-pass features layered on top of features.parquet.

All historical rates use games strictly before the current one (or snaps strictly before
the current snap for in-game rates). Residual = pass - xpass (situation-neutral lean).
"""
import sys
import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
BASE = sys.argv[2] if len(sys.argv) > 2 else "data/features.parquet"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/features_v2.parquet"
PLAYERS = "data/ref/players.parquet"


def prior_game_rate(df, keys, k, prefix, extra_keys=()):
    """Shrunk prior-game mean residual for each key, plus count. df has gorder, game_id, _r, _n."""
    g = df.groupby(list(keys) + ["game_id", "gorder"], as_index=False, observed=True).agg(r=("_r", "sum"), n=("_n", "sum"))
    g = g.sort_values("gorder")
    grp = g.groupby(list(keys), observed=True)
    g["cr"] = grp["r"].cumsum() - g["r"]
    g["cn"] = grp["n"].cumsum() - g["n"]
    g[prefix] = g["cr"] / (g["cn"] + k)
    g[prefix + "_n"] = g["cn"]
    return g[list(keys) + ["game_id", prefix, prefix + "_n"]]


def main():
    d = pd.read_parquet(BASE)
    d["season"] = d["season"].astype(int)
    gorder = d.drop_duplicates("game_id")[["game_id"]].reset_index(drop=True)
    gorder["gorder"] = np.arange(len(gorder))
    d = d.merge(gorder, on="game_id", how="left")

    tgt = (d["ptype"].isin(["pass", "run"]) & d["play_type"].isin(["pass", "run"])).values
    d["_r"] = np.where(tgt, (d["ptype"] == "pass").astype(float) - d["xpass"], 0.0)
    d["_n"] = tgt.astype(float)

    # ---------- players on field ----------
    parts = []
    for s in range(2016, 2027):
        try:
            p = pd.read_parquet(f"{RAW}/part_{s}.parquet", columns=["nflverse_game_id", "play_id", "offense_players"])
            parts.append(p.rename(columns={"nflverse_game_id": "game_id"}))
        except FileNotFoundError:
            pass
    part = pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "play_id"])
    part["play_id"] = part["play_id"].astype(float)
    d = d.merge(part, on=["game_id", "play_id"], how="left")

    pos = pd.read_parquet(PLAYERS, columns=["gsis_id", "position"]).dropna().drop_duplicates("gsis_id")
    posmap = dict(zip(pos.gsis_id, pos.position))
    role = {"RB": "RB", "FB": "RB", "HB": "RB", "TE": "TE", "WR": "WR", "QB": "QB"}

    ex = d.loc[d["offense_players"].notna(), ["game_id", "play_id", "posteam", "gorder", "season",
                                               "_r", "_n", "offense_players"]].copy()
    ex["pl"] = ex["offense_players"].str.split(";")
    ex = ex.drop(columns="offense_players").explode("pl")
    ex["role"] = ex["pl"].map(posmap).map(role)
    ex = ex[ex["role"].notna()]

    # player tell overall and early-down only
    pt = prior_game_rate(ex, ["pl"], 60, "ptell")
    ex = ex.merge(pt, on=["pl", "game_id"], how="left")

    # player snap share over his team's last 4 games (specialist detector)
    team_snaps = d[d["offense_players"].notna()].groupby(["posteam", "game_id", "gorder"], as_index=False).size().rename(columns={"size": "tsn"})
    team_snaps = team_snaps.sort_values("gorder")
    team_snaps["tsn_l4"] = team_snaps.groupby("posteam")["tsn"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
    pgm = ex.groupby(["pl", "posteam", "game_id", "gorder"], as_index=False).size().rename(columns={"size": "psn"})
    # player's snaps in team's last 4 games: build team game index per player-team
    pgm = pgm.merge(team_snaps[["posteam", "game_id", "tsn", "tsn_l4"]], on=["posteam", "game_id"])
    tg_idx = team_snaps[["posteam", "game_id", "gorder"]].copy()
    tg_idx["tidx"] = tg_idx.groupby("posteam").cumcount()
    pgm = pgm.merge(tg_idx[["posteam", "game_id", "tidx"]], on=["posteam", "game_id"])
    # dense per (pl, posteam) over team games
    full = tg_idx[["posteam", "game_id", "tidx"]].merge(pgm[["pl", "posteam"]].drop_duplicates(), on="posteam")
    full = full.merge(pgm[["pl", "posteam", "game_id", "psn"]], on=["pl", "posteam", "game_id"], how="left").fillna({"psn": 0})
    full = full.sort_values(["pl", "posteam", "tidx"])
    full["psn_l4"] = full.groupby(["pl", "posteam"])["psn"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
    full = full.merge(team_snaps[["posteam", "game_id", "tsn_l4"]], on=["posteam", "game_id"])
    full["share_l4"] = full["psn_l4"] / full["tsn_l4"]
    ex = ex.merge(full[["pl", "posteam", "game_id", "share_l4"]], on=["pl", "posteam", "game_id"], how="left")

    # QB prior experience (snaps before this game)
    qbx = ex[ex["role"] == "QB"].groupby(["pl", "game_id", "gorder"], as_index=False).size().sort_values("gorder")
    qbx["qb_prior_snaps"] = qbx.groupby("pl")["size"].cumsum() - qbx["size"]
    ex = ex.merge(qbx[["pl", "game_id", "qb_prior_snaps"]], on=["pl", "game_id"], how="left")

    sk = ex[ex["role"] != "QB"]
    agg = sk.groupby(["game_id", "play_id"]).agg(
        rb_tell=("ptell", lambda s: np.nan), te_tell=("ptell", lambda s: np.nan))
    def role_agg(r, name):
        a = sk[sk["role"] == r].groupby(["game_id", "play_id"])["ptell"].agg(["max", "mean", "min"])
        a.columns = [f"{name}_tell_max", f"{name}_tell_mean", f"{name}_tell_min"]
        return a
    agg = pd.concat([role_agg("RB", "rb"), role_agg("TE", "te"), role_agg("WR", "wr"),
                     sk.groupby(["game_id", "play_id"]).agg(min_share=("share_l4", "min"),
                                                           mean_share=("share_l4", "mean"),
                                                           n_low_share=("share_l4", lambda s: (s < 0.25).sum()))], axis=1)
    rbshare = sk[sk["role"] == "RB"].groupby(["game_id", "play_id"])["share_l4"].max().rename("rb_share")
    qb = ex[ex["role"] == "QB"].groupby(["game_id", "play_id"]).agg(qb_prior_snaps=("qb_prior_snaps", "max"),
                                                                   qb_tell=("ptell", "max"))
    agg = agg.join(rbshare).join(qb).reset_index()
    d = d.merge(agg, on=["game_id", "play_id"], how="left")
    d["qb_backup"] = (d["qb_prior_snaps"] < 300).astype(float).where(d["qb_prior_snaps"].notna())

    # exact skill-group (lineup) tells
    skey = sk.sort_values("pl").groupby(["game_id", "play_id"])["pl"].agg(lambda s: "|".join(s)).rename("skill_key").reset_index()
    d = d.merge(skey, on=["game_id", "play_id"], how="left")
    lk = d[d["skill_key"].notna()]
    lt = prior_game_rate(lk, ["posteam", "skill_key"], 25, "lineup_tell")
    d = d.merge(lt, on=["posteam", "skill_key", "game_id"], how="left")

    # substitutions vs previous snap
    tg = d.groupby(["game_id", "posteam"], sort=False)
    for c in ["n_rb", "n_te", "n_wr"]:
        d["chg_" + c] = d[c] - tg[c].shift(1)
    d["prev_skill_key"] = tg["skill_key"].shift(1)
    def nchanged(a, b):
        if not isinstance(a, str) or not isinstance(b, str):
            return np.nan
        return len(set(a.split("|")) ^ set(b.split("|"))) / 2
    d["skill_players_changed"] = [nchanged(a, b) for a, b in zip(d["skill_key"], d["prev_skill_key"])]
    d["same_lineup_as_prev"] = (d["skill_players_changed"] == 0).astype(float).where(d["skill_players_changed"].notna())
    d["prev_type_same_lineup"] = np.where(d["same_lineup_as_prev"] == 1,
                                          np.where(d["prev1_type"] == "pass", 1, np.where(d["prev1_type"] == "run", -1, 0)), 0)

    # team / coach tendencies by personnel & formation
    d["pf"] = d["personnel_grp"].astype(str) + "_" + d["formation"].astype(str) + "_" + d["shotgun"].fillna(-1).astype(int).astype(str)
    has = d[d["has_personnel"] == 1]
    for keys, k, name in [(["posteam", "personnel_grp"], 80, "team_grp_proe"),
                          (["posteam", "pf"], 60, "team_pf_proe"),
                          (["coach", "pf"], 80, "coach_pf_proe"),
                          (["posteam", "season", "pf"], 40, "team_season_pf_proe"),
                          (["pf"], 300, "league_pf_proe")]:
        r = prior_game_rate(has, keys, k, name)
        d = d.merge(r, on=keys + ["game_id"], how="left")
    # usage rarity of this personnel group for this team (prior games)
    cnt = has.groupby(["posteam", "personnel_grp", "game_id", "gorder"], as_index=False).size()
    cnt = cnt.sort_values("gorder")
    cnt["c"] = cnt.groupby(["posteam", "personnel_grp"])["size"].cumsum() - cnt["size"]
    tot = has.groupby(["posteam", "game_id", "gorder"], as_index=False).size().sort_values("gorder")
    tot["t"] = tot.groupby("posteam")["size"].cumsum() - tot["size"]
    cnt = cnt.merge(tot[["posteam", "game_id", "t"]], on=["posteam", "game_id"])
    cnt["grp_usage"] = (cnt["c"] + 1) / (cnt["t"] + 10)
    d = d.merge(cnt[["posteam", "personnel_grp", "game_id", "grp_usage"]], on=["posteam", "personnel_grp", "game_id"], how="left")

    # in-game: pass lean so far today in this personnel group / formation
    for key, name in [("personnel_grp", "ingame_grp"), ("pf", "ingame_pf")]:
        gg = d.groupby(["game_id", "posteam", key], sort=False, observed=True)
        cr = gg["_r"].cumsum() - d["_r"]
        cn = gg["_n"].cumsum() - d["_n"]
        d[name + "_proe"] = cr / (cn + 4)
        d[name + "_n"] = cn

    # coach tendency by down/distance bucket
    dd = np.select([(d.down == 1), (d.down == 2) & (d.ydstogo <= 3), (d.down == 2) & (d.ydstogo <= 7),
                    (d.down == 2), (d.down == 3) & (d.ydstogo <= 3), (d.down == 3)],
                   ["1", "2s", "2m", "2l", "3s", "3l"], "4")
    d["dd_bucket"] = dd
    for keys, k, name in [(["coach", "dd_bucket"], 150, "coach_dd_proe"),
                          (["posteam", "season", "dd_bucket"], 60, "team_season_dd_proe"),
                          (["posteam", "season"], 150, "team_season_proe")]:
        r = prior_game_rate(d, keys, k, name)
        d = d.merge(r, on=keys + ["game_id"], how="left")

    # previous snap's personnel-conditioned outcome: did the same lineup just fail?
    d["prev_epa_x_same_lineup"] = d["prev_epa"] * d["same_lineup_as_prev"]

    drop = [c for c in d.columns if c.startswith("_")] + ["offense_players", "prev_skill_key", "skill_key"]
    d = d.drop(columns=drop)
    d.to_parquet(OUT, index=False)
    t = d[d.is_target == 1]
    new = ["rb_tell_max", "wr_tell_mean", "te_tell_mean", "min_share", "rb_share", "qb_prior_snaps", "qb_tell",
           "lineup_tell", "skill_players_changed", "team_grp_proe", "team_pf_proe", "coach_pf_proe",
           "team_season_pf_proe", "league_pf_proe", "grp_usage", "ingame_grp_proe", "ingame_pf_proe",
           "coach_dd_proe", "team_season_dd_proe", "team_season_proe", "chg_n_wr"]
    print(t[new].describe().T[["count", "mean", "std", "min", "max"]].round(3))
    print(len(t), "plays; cols", d.shape[1])


if __name__ == "__main__":
    main()
