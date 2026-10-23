"""Pick a representative 2025 drive for the hero: long, mixed calls, model mostly right but not perfect."""
import json
import pandas as pd

D = "data"
P = pd.read_parquet(f"{D}/preds_v2.parquet")
F = pd.read_parquet(f"{D}/features_v2.parquet",
                    columns=["game_id", "play_id", "fixed_drive", "yardline_100", "score_differential",
                             "qtr", "half_seconds_remaining", "game_seconds_remaining", "yards_gained", "result",
                             "home_team", "posteam", "defteam", "week", "season_type", "is_target", "touchdown"])
P = P[P.season == 2025].merge(F.drop(columns=["posteam", "defteam", "week"]), on=["game_id", "play_id"])
P["hit"] = (P.full_v2 > .5) == P.y
best = []
for (g, dr), x in P.groupby(["game_id", "fixed_drive"]):
    if not (9 <= len(x) <= 12):
        continue
    mix = x.y.mean()
    if not (0.35 <= mix <= 0.65):
        continue
    if x.down.isin([3, 4]).sum() > 3:
        continue
    acc = x.hit.mean()
    if not (0.8 <= acc < 1.0):
        continue
    nonobvious = ((x.full_v2 - 0.5).abs() < 0.35).sum()
    best.append((acc, nonobvious, x.touchdown.max(), g, dr, x.qtr.iloc[0]))
best.sort(key=lambda t: (-t[2], -t[0], -t[1]))
acc, _, _, g, dr, _ = best[0]
x = P[(P.game_id == g) & (P.fixed_drive == dr)].sort_values("play_id")
lab = {"run_stuffed": "stopped at or behind the line", "run_1_3": "short gain", "run_4_9": "solid gain",
       "run_10plus": "big gain", "incomplete": "incomplete", "sack": "sack", "comp_lt5": "short completion",
       "comp_5_14": "completion", "comp_15plus": "big completion", "scramble": "QB scramble",
       "turnover": "turnover", "pen_off": "penalty", "pen_def": "penalty"}
def field(yl, pos, dfn):
    return f"own {100 - int(yl)}" if yl > 50 else ("midfield" if yl == 50 else f"opp {int(yl)}")
def clock(hs, q):
    s = int(hs) - (900 if q in (1, 3) else 0)
    return f"Q{int(q)} {s // 60}:{s % 60:02d}"
snaps = []
for _, r in x.iterrows():
    dn = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[int(r.down)]
    togo = "Goal" if r.yardline_100 <= r.ydstogo else int(r.ydstogo)
    yds = int(r.yards_gained) if pd.notna(r.yards_gained) else 0
    res = lab.get(r.result, "")
    res = f"{res}, {yds:+d} yds" if r.result not in ("incomplete",) and res else res
    if r.touchdown == 1:
        res = "touchdown"
    sd = int(r.score_differential)
    score = "tied" if sd == 0 else (f"up {sd}" if sd > 0 else f"down {-sd}")
    snaps.append(dict(situation=f"{dn} & {togo}", context=f"{clock(r.half_seconds_remaining, r.qtr)}, {field(r.yardline_100, r.posteam, r.defteam)}, {score}",
                      p_pass=round(float(r.full_v2), 3), actual="pass" if r.y == 1 else "run", result=res))
r0 = x.iloc[0]
out = dict(title=f"One real drive: {r0.posteam} vs {r0.defteam}, 2025 week {int(r0.week)}",
           hits=int(x.hit.sum()), n=int(len(x)), snaps=snaps)
json.dump(out, open("site/drive.json", "w"), indent=1)
print(out["title"], out["hits"], "/", out["n"])
for s in snaps:
    print(s)
