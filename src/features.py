"""Build play-level features for NFL pass/run prediction.

Layers:
  situation  - what the down/distance/clock/score says (the 'rational' baseline)
  cognitive  - what's in the play-caller's head: last play's result, streaks,
               what's been working today, emotional events (just got scored on,
               just turned it over), scripted openers, timeouts, coach habits
  personnel  - who's on the field and in what formation, incl. per-player 'tells'

Every rolling / historical feature uses only information available before the snap.
"""
import sys
import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/features.parquet"
SEASONS = range(2016, 2027)

PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team",
    "play_type", "pass", "rush", "down", "ydstogo", "yardline_100", "score_differential",
    "game_seconds_remaining", "half_seconds_remaining", "qtr", "game_half",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "wp", "vegas_wp",
    "spread_line", "total_line", "roof", "temp", "wind", "goal_to_go", "shotgun", "no_huddle",
    "qb_kneel", "qb_spike", "two_point_attempt", "yards_gained", "epa", "success",
    "incomplete_pass", "sack", "interception", "fumble_lost", "qb_scramble", "penalty",
    "penalty_team", "timeout", "timeout_team", "fixed_drive", "fixed_drive_result",
    "home_coach", "away_coach", "xpass", "desc", "touchdown", "first_down",
]


def load():
    frames, parts = [], []
    for s in SEASONS:
        d = pd.read_parquet(f"{RAW}/pbp_{s}.parquet", columns=PBP_COLS)
        frames.append(d)
        try:
            p = pd.read_parquet(f"{RAW}/part_{s}.parquet",
                                columns=["nflverse_game_id", "play_id", "offense_formation",
                                         "offense_personnel", "offense_players"])
            parts.append(p.rename(columns={"nflverse_game_id": "game_id"}))
        except FileNotFoundError:
            pass
    pbp = pd.concat(frames, ignore_index=True)
    part = pd.concat(parts, ignore_index=True)
    part["play_id"] = part["play_id"].astype(float)
    pbp["play_id"] = pbp["play_id"].astype(float)
    return pbp, part


def main():
    pbp, part = load()
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    desc = pbp["desc"].fillna("")

    # ---- events between snaps (timeouts, 2-minute warning, quarter breaks) ----
    pbp["ev_timeout_off"] = ((pbp["timeout"] == 1) & (pbp["timeout_team"] == pbp["posteam"])).astype(int)
    pbp["ev_timeout_any"] = (pbp["timeout"] == 1).astype(int)
    pbp["ev_2mw"] = desc.str.contains("Two-Minute Warning", case=False).astype(int)
    pbp["ev_qend"] = desc.str.contains("END QUARTER", case=False).astype(int)
    g = pbp.groupby("game_id")
    for c in ["ev_timeout_off", "ev_timeout_any", "ev_2mw", "ev_qend"]:
        pbp[c + "_cum"] = g[c].cumsum()

    # timeout rows don't carry the right posteam reliably; recompute 'our timeout' via timeout_team later
    is_snap = pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"]) | (
        (pbp["play_type"] == "no_play") & pbp["posteam"].notna())
    snaps = pbp[is_snap & pbp["posteam"].notna() & pbp["down"].notna()].copy()

    # ---- what kind of snap / result was it ----
    pt = np.where(snaps["play_type"].eq("no_play"),
                  np.where(snaps["pass"] == 1, "pass", np.where(snaps["rush"] == 1, "run", "pen")),
                  np.where(snaps["play_type"].isin(["qb_kneel"]), "run",
                           np.where(snaps["play_type"].eq("qb_spike"), "pass",
                                    np.where(snaps["pass"] == 1, "pass", snaps["play_type"]))))
    snaps["ptype"] = pt
    y = snaps["yards_gained"].fillna(0)
    res = np.select(
        [snaps["play_type"].eq("no_play") & snaps["penalty_team"].eq(snaps["posteam"]),
         snaps["play_type"].eq("no_play"),
         (snaps["interception"] == 1) | ((snaps["fumble_lost"] == 1)),
         snaps["sack"] == 1,
         snaps["incomplete_pass"] == 1,
         (snaps["ptype"] == "pass") & (snaps["qb_scramble"] == 1),
         (snaps["ptype"] == "pass") & (y >= 15),
         (snaps["ptype"] == "pass") & (y >= 5),
         (snaps["ptype"] == "pass"),
         (snaps["ptype"] == "run") & (y <= 0),
         (snaps["ptype"] == "run") & (y <= 3),
         (snaps["ptype"] == "run") & (y <= 9),
         (snaps["ptype"] == "run")],
        ["pen_off", "pen_def", "turnover", "sack", "incomplete", "scramble", "comp_15plus",
         "comp_5_14", "comp_lt5", "run_stuffed", "run_1_3", "run_4_9", "run_10plus"],
        default="other")
    snaps["result"] = res

    # ---- sequence features within team-game ----
    snaps = snaps.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    tg = snaps.groupby(["game_id", "posteam"], sort=False)
    snaps["snap_no"] = tg.cumcount() + 1
    for k in [1, 2]:
        snaps[f"prev{k}_type"] = tg["ptype"].shift(k).fillna("none")
    snaps["prev_result"] = tg["result"].shift(1).fillna("none")
    snaps["prev_yards"] = tg["yards_gained"].shift(1)
    snaps["prev_epa"] = tg["epa"].shift(1)
    snaps["prev_success"] = tg["success"].shift(1)
    snaps["prev_first_down"] = tg["first_down"].shift(1)
    snaps["prev_drive"] = tg["fixed_drive"].shift(1)
    snaps["first_of_drive"] = (snaps["prev_drive"] != snaps["fixed_drive"]).astype(int)
    snaps["secs_since_prev_snap"] = tg["game_seconds_remaining"].shift(1) - snaps["game_seconds_remaining"]
    snaps.loc[snaps["first_of_drive"] == 1, "secs_since_prev_snap"] = np.nan
    snaps["secs_since_prev_snap"] = snaps["secs_since_prev_snap"].where(snaps["secs_since_prev_snap"].between(0, 120))
    prev_hs = tg["half_seconds_remaining"].shift(1)
    snaps["first_snap_after_2min"] = ((prev_hs >= 120) & (snaps["half_seconds_remaining"] < 120) &
                                      (snaps["half_seconds_remaining"] < prev_hs)).astype(int)

    # streak of same play type (pass/run only)
    pr = snaps["ptype"].where(snaps["ptype"].isin(["pass", "run"]))
    snaps["_pr"] = pr
    def streak(s):
        out, cur, last = [], 0, None
        for v in s:
            out.append(cur if last is not None else 0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            cur = cur + 1 if v == last else 1
            last = v
        return out
    vals = []
    for _, grp in snaps.groupby(["game_id", "posteam"], sort=False)["_pr"]:
        vals.append(pd.Series(streak(grp.tolist()), index=grp.index))
    snaps["same_type_streak"] = pd.concat(vals)
    snaps["streak_signed"] = np.where(snaps["prev1_type"] == "pass", snaps["same_type_streak"],
                                      np.where(snaps["prev1_type"] == "run", -snaps["same_type_streak"], 0))

    # in-game memory: what's been working today (prior snaps only)
    is_p = (snaps["ptype"] == "pass").astype(float)
    is_r = (snaps["ptype"] == "run").astype(float)
    epa = snaps["epa"].fillna(0)
    snaps["_p"], snaps["_r"] = is_p, is_r
    snaps["_pe"], snaps["_re"] = is_p * epa, is_r * epa
    snaps["_ps"], snaps["_rs"] = is_p * snaps["success"].fillna(0), is_r * snaps["success"].fillna(0)
    tg = snaps.groupby(["game_id", "posteam"], sort=False)
    for c in ["_p", "_r", "_pe", "_re", "_ps", "_rs"]:
        snaps[c + "c"] = tg[c].cumsum() - snaps[c]
    snaps["ingame_pass_rate"] = (snaps["_pc"] + 1) / (snaps["_pc"] + snaps["_rc"] + 2)
    snaps["ingame_pass_epa"] = snaps["_pec"] / (snaps["_pc"] + 3)
    snaps["ingame_run_epa"] = snaps["_rec"] / (snaps["_rc"] + 3)
    snaps["ingame_hot_hand"] = snaps["ingame_pass_epa"] - snaps["ingame_run_epa"]
    snaps["ingame_pass_sr"] = (snaps["_psc"] + 1) / (snaps["_pc"] + 2)
    snaps["ingame_run_sr"] = (snaps["_rsc"] + 1) / (snaps["_rc"] + 2)
    # last 5 snaps rolling
    snaps["last5_pass_rate"] = tg["_p"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    snaps["last5_epa"] = tg["epa"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # events since previous snap (any team)
    for c in ["ev_timeout_any", "ev_qend"]:
        prevcum = snaps.groupby("game_id")[c + "_cum"].shift(1).fillna(0)
        snaps["since_" + c[3:]] = (snaps[c + "_cum"] - prevcum).clip(upper=1)
    # timeout by offense specifically: look at rows between snaps
    tos = pbp[pbp["timeout"] == 1][["game_id", "play_id", "timeout_team"]]
    snaps["prev_snap_pid"] = snaps.groupby("game_id")["play_id"].shift(1).fillna(-1)
    m = snaps[["game_id", "play_id", "prev_snap_pid", "posteam"]].merge(tos, on="game_id")
    m = m[(m["play_id_y"] > m["prev_snap_pid"]) & (m["play_id_y"] < m["play_id_x"])]
    off_to = m[m["timeout_team"] == m["posteam"]].groupby(["game_id", "play_id_x"]).size()
    def_to = m[m["timeout_team"] != m["posteam"]].groupby(["game_id", "play_id_x"]).size()
    key = pd.MultiIndex.from_frame(snaps[["game_id", "play_id"]])
    snaps["after_off_timeout"] = off_to.reindex(key).fillna(0).clip(upper=1).values
    snaps["after_def_timeout"] = def_to.reindex(key).fillna(0).clip(upper=1).values

    # ---- drive-level emotional context ----
    drives = (pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()]
              .groupby(["game_id", "fixed_drive"], as_index=False)
              .agg(dposteam=("posteam", "first"), dres=("fixed_drive_result", "first"),
                   dstart=("play_id", "min")))
    drives = drives.sort_values(["game_id", "fixed_drive"])
    drives["prev_drive_team"] = drives.groupby("game_id")["dposteam"].shift(1)
    drives["prev_drive_res"] = drives.groupby("game_id")["dres"].shift(1).fillna("none")
    # my previous drive result
    drives["my_prev_res"] = drives.groupby(["game_id", "dposteam"])["dres"].shift(1).fillna("none")
    snaps = snaps.merge(drives[["game_id", "fixed_drive", "prev_drive_team", "prev_drive_res", "my_prev_res"]],
                        on=["game_id", "fixed_drive"], how="left")
    opp_prev = snaps["prev_drive_team"] != snaps["posteam"]
    snaps["opp_just_scored_td"] = (opp_prev & snaps["prev_drive_res"].eq("Touchdown")).astype(int)
    snaps["opp_just_scored_fg"] = (opp_prev & snaps["prev_drive_res"].eq("Field goal")).astype(int)
    snaps["after_takeaway"] = (opp_prev & snaps["prev_drive_res"].isin(["Turnover", "Opp touchdown"])).astype(int)
    snaps["after_own_turnover"] = snaps["my_prev_res"].isin(["Turnover", "Opp touchdown"]).astype(int)
    snaps["after_own_3out_punt"] = snaps["my_prev_res"].eq("Punt").astype(int)
    snaps["after_own_td"] = snaps["my_prev_res"].eq("Touchdown").astype(int)
    snaps["after_turnover_on_downs"] = snaps["my_prev_res"].eq("Turnover on downs").astype(int)
    snaps["scripted"] = (snaps["snap_no"] <= 15).astype(int)
    snaps["first_snap_2nd_half"] = ((snaps["game_half"] == "Half2") &
                                     (snaps.groupby(["game_id", "posteam", "game_half"]).cumcount() == 0)).astype(int)

    # ---- coach & team habits from PRIOR games only ----
    snaps["coach"] = np.where(snaps["posteam"] == snaps["home_team"], snaps["home_coach"], snaps["away_coach"])
    snaps["home"] = (snaps["posteam"] == snaps["home_team"]).astype(int)
    tgt = snaps["ptype"].isin(["pass", "run"]) & (snaps["play_type"].isin(["pass", "run"]))
    snaps["_tgt"] = tgt
    neutral = tgt & snaps["down"].isin([1, 2]) & snaps["wp"].between(0.2, 0.8) & (snaps["qtr"] <= 3)
    snaps["_n"] = neutral.astype(float)
    snaps["_np"] = (neutral & (snaps["ptype"] == "pass")).astype(float)
    snaps["_resid"] = np.where(tgt, (snaps["ptype"] == "pass").astype(float) - snaps["xpass"], np.nan)
    after_run = tgt & (snaps["prev1_type"] == "run")
    after_pass = tgt & (snaps["prev1_type"] == "pass")
    snaps["_ar_n"], snaps["_ar_r"] = after_run.astype(float), np.where(after_run, snaps["_resid"], 0)
    snaps["_ap_n"], snaps["_ap_r"] = after_pass.astype(float), np.where(after_pass, snaps["_resid"], 0)
    bad = tgt & snaps["prev_result"].isin(["incomplete", "sack", "run_stuffed"])
    snaps["_bad_n"], snaps["_bad_r"] = bad.astype(float), np.where(bad, snaps["_resid"], 0)
    snaps["_all_n"], snaps["_all_r"] = tgt.astype(float), np.where(tgt, snaps["_resid"], 0)
    snaps["_switch"] = (tgt & snaps["prev1_type"].isin(["pass", "run"]) & (snaps["ptype"] != snaps["prev1_type"])).astype(float)
    snaps["_switch_n"] = (tgt & snaps["prev1_type"].isin(["pass", "run"])).astype(float)

    games = snaps[["game_id", "season", "week", "posteam", "coach", "defteam"]].drop_duplicates(["game_id", "posteam"])
    gid_order = snaps.drop_duplicates("game_id")[["game_id"]].reset_index(drop=True)
    gid_order["gorder"] = np.arange(len(gid_order))
    agg_cols = ["_n", "_np", "_ar_n", "_ar_r", "_ap_n", "_ap_r", "_bad_n", "_bad_r", "_all_n", "_all_r",
                "_switch", "_switch_n"]
    gsum = snaps.groupby(["game_id", "posteam"], as_index=False)[agg_cols].sum()
    gsum = gsum.merge(games, on=["game_id", "posteam"]).merge(gid_order, on="game_id")
    # game ids sort chronologically by season_week, so gorder is chronological

    def prior_cum(df, keys, prefix, cols):
        df = df.sort_values("gorder")
        c = df.groupby(keys)[cols].cumsum() - df[cols]
        c.columns = [prefix + x for x in cols]
        return pd.concat([df[["game_id", "posteam"]], c], axis=1)

    def prior_roll(df, keys, prefix, cols, n):
        df = df.sort_values("gorder")
        c = df.groupby(keys)[cols].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
        c.columns = [prefix + x for x in cols]
        return pd.concat([df[["game_id", "posteam"]], c], axis=1)

    cc = prior_cum(gsum, ["coach"], "cc", agg_cols)
    tr = prior_roll(gsum, ["posteam"], "tr", agg_cols, 8)
    hist = cc.merge(tr, on=["game_id", "posteam"])
    K = 30
    hist["coach_neutral_pass"] = (hist["cc_np"] + 0.55 * 200) / (hist["cc_n"] + 200)
    hist["coach_proe"] = hist["cc_all_r"] / (hist["cc_all_n"] + 300)
    hist["coach_proe_after_run"] = hist["cc_ar_r"] / (hist["cc_ar_n"] + 200)
    hist["coach_proe_after_pass"] = hist["cc_ap_r"] / (hist["cc_ap_n"] + 200)
    hist["coach_proe_after_fail"] = hist["cc_bad_r"] / (hist["cc_bad_n"] + 200)
    hist["coach_switch_rate"] = (hist["cc_switch"] + 0.5 * 200) / (hist["cc_switch_n"] + 200)
    hist["coach_games_n"] = hist["cc_all_n"]
    hist["team_neutral_pass_l8"] = (hist["tr_np"] + 0.55 * 50) / (hist["tr_n"] + 50)
    hist["team_proe_l8"] = hist["tr_all_r"] / (hist["tr_all_n"] + 100)
    keep = ["game_id", "posteam", "coach_neutral_pass", "coach_proe", "coach_proe_after_run",
            "coach_proe_after_pass", "coach_proe_after_fail", "coach_switch_rate", "coach_games_n",
            "team_neutral_pass_l8", "team_proe_l8"]
    snaps = snaps.merge(hist[keep], on=["game_id", "posteam"], how="left")

    # opponent defense (game-plan): last 8 games EPA allowed vs pass / run
    tgt = snaps["_tgt"]
    epa = snaps["epa"].fillna(0)
    snaps["_dp_e"] = np.where(tgt & (snaps["ptype"] == "pass"), epa, 0)
    snaps["_dr_e"] = np.where(tgt & (snaps["ptype"] == "run"), epa, 0)
    snaps["_dp_n"] = (tgt & (snaps["ptype"] == "pass")).astype(float)
    snaps["_dr_n"] = (tgt & (snaps["ptype"] == "run")).astype(float)
    dsum = snaps.groupby(["game_id", "defteam"], as_index=False)[["_dp_e", "_dr_e", "_dp_n", "_dr_n"]].sum()
    dsum = dsum.merge(gid_order, on="game_id").sort_values("gorder")
    dc = dsum.groupby("defteam")[["_dp_e", "_dr_e", "_dp_n", "_dr_n"]].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).sum())
    dsum["opp_def_pass_epa"] = dc["_dp_e"] / (dc["_dp_n"] + 100)
    dsum["opp_def_run_epa"] = dc["_dr_e"] / (dc["_dr_n"] + 100)
    snaps = snaps.merge(dsum[["game_id", "defteam", "opp_def_pass_epa", "opp_def_run_epa"]],
                        on=["game_id", "defteam"], how="left")

    # ---- personnel & formation ----
    part = part.drop_duplicates(["game_id", "play_id"])
    snaps = snaps.merge(part, on=["game_id", "play_id"], how="left")
    per = snaps["offense_personnel"].fillna("")
    parsed = per.str.extractall(r"(\d+)\s+([A-Z]+)")
    parsed[0] = parsed[0].astype(int)
    piv = parsed.reset_index().pivot_table(index="level_0", columns=1, values=0, aggfunc="sum").reindex(snaps.index)
    def col(*cs):
        out = sum(piv[c].fillna(0) for c in cs if c in piv.columns)
        return out.where(snaps["offense_personnel"].notna())
    snaps["n_rb"] = col("RB", "FB", "HB")
    snaps["n_fb"] = col("FB")
    snaps["n_te"] = col("TE")
    snaps["n_wr"] = col("WR")
    snaps["n_ol_extra"] = col("OL", "T", "G", "C")
    snaps.loc[snaps["n_ol_extra"] >= 5, "n_ol_extra"] -= 5   # new format lists all 5 OL
    snaps["n_def_on_off"] = col("DL", "DT", "DE", "NT", "LB", "ILB", "OLB", "MLB")
    snaps["has_personnel"] = snaps["offense_personnel"].notna().astype(int)
    snaps["personnel_grp"] = np.where(snaps["has_personnel"] == 1,
                                      snaps["n_rb"].fillna(0).astype(int).astype(str) + snaps["n_te"].fillna(0).astype(int).astype(str),
                                      "na")
    # nflverse switched formation labels in 2023 (EMPTY/I_FORM/SINGLEBACK/JUMBO/WILDCAT -> SHOTGUN/UNDER CENTER/PISTOL);
    # map every season onto the new set so models trained on old seasons read new ones
    f = snaps["offense_formation"]
    sg = snaps["shotgun"] == 1
    snaps["formation"] = np.where(f.isna(), "na", np.where(f.eq("PISTOL"), "PISTOL",
                                  np.where(f.isin(["SHOTGUN", "EMPTY", "WILDCAT", "UNDER CENTER"]) & sg, "SHOTGUN",
                                           np.where(f.eq("SHOTGUN"), "SHOTGUN", "UNDER CENTER"))))

    # per-player tells: each skill player's prior residual pass rate when on the field
    sk = snaps.loc[(snaps["has_personnel"] == 1) & snaps["_tgt"], ["game_id", "play_id", "posteam", "season",
                                                     "offense_players", "_resid"]].copy()
    sk = sk[sk["_resid"].notna()]
    sk = sk.merge(gid_order, on="game_id")
    sk["pl"] = sk["offense_players"].str.split(";")
    ex = sk[["game_id", "play_id", "posteam", "gorder", "_resid", "pl"]].explode("pl")
    pg = ex.groupby(["pl", "game_id", "gorder"], as_index=False).agg(r=("_resid", "sum"), n=("_resid", "size"))
    pg = pg.sort_values("gorder")
    pg["cr"] = pg.groupby("pl")["r"].cumsum() - pg["r"]
    pg["cn"] = pg.groupby("pl")["n"].cumsum() - pg["n"]
    pg["tell"] = pg["cr"] / (pg["cn"] + 60)
    ex = ex.merge(pg[["pl", "game_id", "tell", "cn"]], on=["pl", "game_id"], how="left")
    # RB share-of-snaps tell: is the 'third-down back' in?
    agg = ex.groupby(["game_id", "play_id"]).agg(
        tell_max=("tell", "max"), tell_min=("tell", "min"), tell_sum=("tell", "sum"),
        tell_mean=("tell", "mean"))
    snaps = snaps.merge(agg, on=["game_id", "play_id"], how="left")

    # ---- final target set ----
    snaps["is_target"] = (snaps["play_type"].isin(["pass", "run"]) & (snaps["qb_kneel"] == 0) &
                          (snaps["qb_spike"] == 0) & (snaps["two_point_attempt"] == 0)).astype(int)
    snaps["y"] = (snaps["pass"] == 1).astype(int)
    snaps["neutral"] = (snaps["down"].isin([1, 2]) & snaps["wp"].between(0.2, 0.8) &
                        ~((snaps["half_seconds_remaining"] < 120))).astype(int)
    snaps["outdoors"] = snaps["roof"].isin(["outdoors", "open"]).astype(int)

    drop = [c for c in snaps.columns if c.startswith("_") or c.startswith("ev_") or c in
            ["desc", "offense_players", "offense_personnel", "offense_formation",
             "prev_drive", "prev_snap_pid"]]
    snaps = snaps.drop(columns=drop)
    snaps.to_parquet(OUT, index=False)
    t = snaps[snaps.is_target == 1]
    print("target plays:", len(t), "pass rate", t.y.mean().round(3), "with personnel", t.has_personnel.mean().round(3))
    print(t.groupby("season").size())


if __name__ == "__main__":
    main()
