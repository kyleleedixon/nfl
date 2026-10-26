"""Who called the offensive plays: src/playcallers.csv on top of the head coach (nflverse home_coach/away_coach).

The CSV lists only the exceptions: team-seasons where someone other than the head coach called plays, and mid-season
handoffs (from_week). A row applies when its head_coach matches the game's head coach (nflverse doesn't always record
interim head coaches, so rows are keyed on the coach the data shows); a blank head_coach applies to any (2026: nflverse's
head coaches are stale for some teams, so those rows list every team's caller outright).
Sources: team announcements and reporting (ESPN, AP, team sites), checked 2026-10-24; 'confidence' marks the rest. 'unknown' = not known yet; that offense gets its own
'TEAM SEASON OC' identity with no history rather than a guess.
Seasons the CSV doesn't cover yet: a head coach known to delegate (a CSV row the season before) keeps last season's
caller, unless that caller is a head coach somewhere this season; anyone else is assumed to call his own plays.
"""
import os
import pandas as pd

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playcallers.csv")
ALIAS = {"Klint Kubliak": "Klint Kubiak", "Jay Rosburg": "Jerry Rosburg"}   # nflverse spellings


HCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headcoaches.csv")


def head_coaches(games):
    """Corrected head coach per row (season, week, posteam, coach): src/headcoaches.csv overrides nflverse, whose head
    coach for some 2026 teams is stale (ARI, ATL, BUF). Maintained by check_coaches.py."""
    coach = games["coach"].replace(ALIAS)
    if not os.path.exists(HCS):
        return coach
    h = pd.read_csv(HCS).sort_values("from_week")
    for (s, team), rows in h.groupby(["season", "team"]):
        m = (games["season"] == s) & (games["posteam"] == team)
        for r in rows.itertuples():
            coach = coach.mask(m & (games["week"] >= r.from_week), r.head_coach)
    return coach


def resolve(games):
    """games: one row per (game_id, posteam) with season, week, posteam, coach (head coach). Returns the caller per row."""
    g = games.copy()
    g["coach"] = g["coach"].replace(ALIAS)
    t = pd.read_csv(CSV)
    t["head_coach"] = t["head_coach"].replace(ALIAS)
    out = []
    covered = set(zip(t.season, t.team))
    hcs = {s: set(x.coach) for s, x in g.groupby("season")}
    last = {}   # (team, season) -> caller at season's end
    for (s, team), x in g.sort_values(["season", "week"]).groupby(["season", "posteam"], sort=True):
        rows = t[(t.season == s) & (t.team == team)].sort_values("from_week")
        for idx, r in x.iterrows():
            m = rows[(rows.head_coach.isna() | (rows.head_coach == r.coach)) & (rows.from_week <= r.week)]
            if len(m):
                c = m.iloc[-1].caller
                c = f"{team} {s} OC" if c == "unknown" else c
            elif (s, team) not in covered and (s - 1, team) in covered and last.get((team, s - 1)) not in (None, r.coach) \
                    and t[(t.season == s - 1) & (t.team == team)].head_coach.eq(r.coach).any():
                prev = last[(team, s - 1)]
                c = f"{team} {s} OC" if prev in hcs.get(s, set()) or prev.endswith(" OC") else prev
            else:
                c = r.coach
            out.append((idx, c))
        last[(team, s)] = out[-1][1]
    return pd.Series(dict(out)).reindex(g.index)


if __name__ == "__main__":
    d = pd.read_parquet("data/features.parquet", columns=["game_id", "season", "week", "posteam", "coach"]).dropna()
    gm = d.drop_duplicates(["game_id", "posteam"]).reset_index(drop=True)
    gm["caller"] = resolve(gm)
    diff = gm[gm.caller != gm.coach.replace(ALIAS)]
    print(f"{len(gm)} team-games; caller differs from head coach in {len(diff)} ({len(diff) / len(gm):.0%})")
    print(gm[gm.season == 2026].drop_duplicates(["posteam", "caller"])[["posteam", "coach", "caller"]].sort_values("posteam").to_string(index=False))
