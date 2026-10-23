"""Weekly predictability report: which offenses were easiest / hardest to read before the snap.

Usage: python src/weekly_report.py [season] [week]   (defaults: latest season and week in preds)
Writes data/reports/{season}_wk{week:02d}.md (rankings + draft tweet thread) and _card.png (share image).
Aggregate results only: no feature names or tells (same rule as site/).
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = os.environ.get("DATA", "data")
P = pd.read_parquet(f"{D}/{os.environ.get('PREDS', 'preds_v3')}.parquet")
BRAND = os.environ.get("BRAND", "Telegraphed")
HANDLE = "@TelegraphedNFL"
SITE = os.environ.get("SITE", "telegraphed.vercel.app")
MIN_NEUTRAL = 60  # season-to-date neutral snaps needed to be ranked

NICK = dict(ARI="Cardinals", ATL="Falcons", BAL="Ravens", BUF="Bills", CAR="Panthers", CHI="Bears",
            CIN="Bengals", CLE="Browns", DAL="Cowboys", DEN="Broncos", DET="Lions", GB="Packers",
            HOU="Texans", IND="Colts", JAX="Jaguars", KC="Chiefs", LA="Rams", LAC="Chargers", LV="Raiders",
            MIA="Dolphins", MIN="Vikings", NE="Patriots", NO="Saints", NYG="Giants", NYJ="Jets",
            PHI="Eagles", PIT="Steelers", SF="49ers", SEA="Seahawks", TB="Buccaneers", TEN="Titans",
            WAS="Commanders")

season = int(sys.argv[1]) if len(sys.argv) > 1 else int(P.season.max())
S = P[P.season == season].copy()
week = int(sys.argv[2]) if len(sys.argv) > 2 else int(S.week.max())
S = S[S.week <= week]
# best available model per play: full (personnel) when it exists, else live
S["p"] = S.full_v3.fillna(S.live_v3) if "full_v3" in S else S.live_v3
S["hit"] = (S.p > 0.5) == S.y
S["hit_pub"] = (S.xpass > 0.5) == S.y


def table(df):
    g = df.groupby("posteam").agg(plays=("hit", "size"), acc=("hit", "mean"), acc_pub=("hit_pub", "mean"))
    n = df[df.neutral == 1].groupby("posteam").agg(neutral=("hit", "size"), acc_neutral=("hit", "mean"))
    return g.join(n).reset_index()


def ranked(df):
    t = table(df)
    t = t[t.neutral >= MIN_NEUTRAL].sort_values("acc_neutral", ascending=False).reset_index(drop=True)
    t["rank"] = t.index + 1
    return t


ytd = ranked(S)
prev_rank = dict(zip(*ranked(S[S.week < week])[["posteam", "rank"]].values.T))
ytd["move"] = [prev_rank.get(t, r) - r for t, r in zip(ytd.posteam, ytd["rank"])]
neutral_avg = S[S.neutral == 1].hit.mean()

wk = table(S[S.week == week]).sort_values("acc", ascending=False).reset_index(drop=True)
opp = S[S.week == week].groupby("posteam").defteam.first()
league = dict(acc=S.hit.mean(), pub=S.hit_pub.mean(), plays=len(S))


def pct(x):
    return f"{100 * x:.0f}%"


L = [f"# Predictability report: {season} week {week}", "",
     f"Model called {pct(league['acc'])} of {league['plays']:,} plays through week {week} "
     f"(public benchmark {pct(league['pub'])}).", "",
     "## Season to date, neutral situations (ranked)",
     "Neutral = downs 1–3, win prob 20–80%, outside the last 2 min of each half. Removes garbage time and obvious downs.", "",
     "| Rank | Move | Offense | Neutral read rate | All plays | Neutral snaps |",
     "|---|---|---|---|---|---|"]
for r in ytd.itertuples():
    mv = f"+{r.move}" if r.move > 0 else (str(r.move) if r.move < 0 else "–")
    L.append(f"| {r.rank} | {mv} | {NICK[r.posteam]} | {pct(r.acc_neutral)} | {pct(r.acc)} | {r.neutral} |")
L += ["", f"## Week {week} games (all plays)", "", "| Offense | Opp | Read rate | Plays |", "|---|---|---|---|"]
for r in wk.itertuples():
    L.append(f"| {NICK[r.posteam]} | {opp[r.posteam]} | {pct(r.acc)} | {r.plays} |")

top, bot = ytd.head(5), ytd.tail(5).iloc[::-1]
wt, wb = wk.iloc[0], wk.iloc[-1]
tweets = [
    f"Telegraphed: Week {week} Rankings 🧵\n\nEvery play caller has a tell. We predict pass or run before every NFL snap, "
    f"and through week {week} we called {pct(league['acc'])} of them.\n\nWho's an Open Book, and who has a Poker Face?",
    "📖 OPEN BOOK: most predictable offenses (neutral situations, season to date)\n\n"
    + "\n".join(f"{r.rank}. {NICK[r.posteam]}: {pct(r.acc_neutral)}" for r in top.itertuples()),
    "🃏 POKER FACE: least predictable\n\n"
    + "\n".join(f"{r.rank}. {NICK[r.posteam]}: {pct(r.acc_neutral)}" for r in bot.itertuples()),
    f"Week {week} extremes:\n\n🔓 {NICK[wt.posteam]} vs {NICK[opp[wt.posteam]]}: we called {pct(wt.acc)} of their plays\n"
    f"🔒 {NICK[wb.posteam]} vs {NICK[opp[wb.posteam]]}: just {pct(wb.acc)}\n\nFull rankings: {SITE}/weekly",
]
L += ["", "## Draft thread", ""]
for i, t in enumerate(tweets, 1):
    L += [f"**{i}/{len(tweets)}** ({len(t)} chars{' ⚠️ over 280' if len(t) > 280 else ''})", "", "```", t, "```", ""]

os.makedirs(f"{D}/reports", exist_ok=True)
out = f"{D}/reports/{season}_wk{week:02d}"
open(f"{out}.md", "w").write("\n".join(L) + "\n")


# brand (from the banner): field stripes, chalk, first-down yellow, line-of-scrimmage blue
FIELD, FIELD2, CHALK, CHALK2, YELLOW, LOS = "#1f6139", "#1b5734", "#f7f9f4", "#cfe0d4", "#ffd23f", "#7db7f0"
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")


def card(path):
    """Share image (1200x1500): every offense's neutral read rate vs league average.
    Open Book (above average) = solid yellow bars, Poker Face (below) = hollow chalk bars, like the banner's legend dots."""
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    for f in os.listdir(FONTS):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONTS, f))
    plt.rcParams["font.family"] = ["Barlow", "DejaVu Sans"]
    disp = dict(family="Barlow Condensed", weight="heavy")
    fig = plt.figure(figsize=(8, 10), dpi=150, facecolor=FIELD)
    for i in range(10):  # 5-yard stripes
        fig.patches.append(Rectangle((0, i / 10), 1, 0.1, transform=fig.transFigure, color=FIELD if i % 2 else FIELD2, zorder=-2))
    for x0 in (0.0, 0.985):  # sideline hash marks
        for i in range(60):
            fig.lines.append(plt.Line2D([x0, x0 + 0.015], [i / 60] * 2, transform=fig.transFigure, color=CHALK, alpha=.18, lw=1))

    fig.text(0.06, 0.885, "HOW PREDICTABLE IS\nYOUR OFFENSE?", color=CHALK, fontsize=34, linespacing=0.9, va="bottom", **disp)
    fig.lines.append(plt.Line2D([0, 1], [0.872] * 2, transform=fig.transFigure, color=YELLOW, lw=3.5))
    fig.text(0.06, 0.838, f"Share of pass/run calls predicted before the snap · {season} weeks 1–{week}",
             color=CHALK, fontsize=11)
    fig.text(0.06, 0.815, f"Neutral situations only · league average {pct(neutral_avg)}", color=CHALK2, fontsize=10.5)

    ax = fig.add_axes([0.25, 0.105, 0.67, 0.645])
    ax.set_facecolor("none")
    d = (ytd.acc_neutral - neutral_avg) * 100
    y = list(range(len(ytd)))[::-1]
    for yi, v in zip(y, d):
        if v >= 0:
            ax.barh(yi, v, height=0.66, color=YELLOW, zorder=3)
        else:
            ax.barh(yi, v, height=0.62, facecolor="none", edgecolor=CHALK, lw=1.4, zorder=3)
    ax.axvline(0, color=LOS, lw=2, zorder=4)
    lim = max(abs(d)) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(ytd) - 0.3)
    for yi, r, v in zip(y, ytd.itertuples(), d):
        ax.annotate(f"{NICK[r.posteam]}", (0, yi), xycoords=("axes fraction", "data"), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", color=CHALK, fontsize=10, weight="medium")
        ax.annotate(f"{r.rank}", (0, yi), xycoords=("axes fraction", "data"), xytext=(-92, 0),
                    textcoords="offset points", ha="right", va="center", color=CHALK2, fontsize=9)
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, pct(r.acc_neutral), ha="left" if v >= 0 else "right",
                va="center", color=CHALK if v >= 0 else CHALK2, fontsize=8.5, zorder=5)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=CHALK2, labelsize=8.5, length=0, pad=6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "avg"))
    ax.grid(axis="x", color=CHALK, alpha=.12, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("points vs league average", color=CHALK2, fontsize=9)
    # legend either side of the line of scrimmage, as on the banner: filled dot = Open Book, hollow = Poker Face
    lx, ly = 0.25 + 0.67 / 2, 0.775
    fig.lines.append(plt.Line2D([lx + 0.025], [ly], transform=fig.transFigure, marker="o", ms=8, color=YELLOW, lw=0))
    fig.text(lx + 0.042, ly, "Open Book", color=CHALK, fontsize=11, weight="semibold", va="center")
    fig.lines.append(plt.Line2D([lx - 0.025], [ly], transform=fig.transFigure, marker="o", ms=8, mfc="none", mec=CHALK,
                                mew=1.8, lw=0))
    fig.text(lx - 0.042, ly, "Poker Face", color=CHALK, fontsize=11, weight="semibold", va="center", ha="right")
    fig.text(0.06, 0.026, "TELEGRAPHED", color=CHALK, fontsize=19, **disp)
    fig.text(0.94, 0.03, f"{HANDLE}  ·  {SITE}", color=CHALK2, fontsize=10, ha="right")
    fig.savefig(path, facecolor=FIELD)


card(f"{out}_card.png")


def page():
    """Public site/weekly/: latest card + full table. Aggregate results only."""
    import html
    import shutil
    os.makedirs("site/weekly", exist_ok=True)
    img = f"{season}_wk{week:02d}.png"
    shutil.copy(f"{out}_card.png", f"site/weekly/{img}")
    rows = "".join(
        f"<tr><td>{r.rank}</td><td class=mv>{'▲' + str(r.move) if r.move > 0 else ('▼' + str(-r.move) if r.move < 0 else '–')}</td>"
        f"<td>{html.escape(NICK[r.posteam])}</td><td class=n>{pct(r.acc_neutral)}</td><td class=n>{pct(r.acc)}</td></tr>"
        for r in ytd.itertuples())
    title = f"How predictable is your offense? {season} week {week}"
    open("site/weekly/index.html", "w").write(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictability Rankings</title>
<meta name="description" content="{title}: every NFL offense ranked by how often we called pass or run before the snap.">
<meta property="og:title" content="{title}"><meta property="og:image" content="https://{SITE}/weekly/{img}">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="https://{SITE}/weekly/{img}">
<link rel="icon" href="/favicon.png"><link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{ --field:#1f6139; --field-2:#1b5734; --chalk:#f7f9f4; --chalk-2:#cfe0d4; --yellow:#ffd23f; --los:#7db7f0;
  --grid:rgba(247,249,244,.14); color-scheme:dark; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--field); color:var(--chalk); font:1.0625rem/1.55 Barlow, system-ui, sans-serif; }}
.wrap {{ max-width:44rem; margin:0 auto; padding:0 16px 3rem; }}
header {{ display:flex; justify-content:space-between; align-items:baseline; padding:1.25rem 0; border-bottom:3px solid var(--yellow); }}
header a {{ font:800 1.75rem "Barlow Condensed", sans-serif; letter-spacing:.01em; color:var(--chalk); text-decoration:none; }}
h1 {{ font:800 clamp(2.2rem,8vw,3.6rem)/.95 "Barlow Condensed", sans-serif; text-transform:uppercase; margin:2rem 0 .75rem; text-wrap:balance; }}
p {{ color:var(--chalk-2); margin:0 0 1.5rem; }}
img {{ width:100%; height:auto; border-radius:8px; display:block; }}
table {{ width:100%; border-collapse:collapse; margin-top:2rem; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:.45rem .5rem; border-bottom:1px solid var(--grid); }}
th {{ font-weight:500; color:var(--chalk-2); font-size:.9rem; }}
td.n, th.n {{ text-align:right; }} td.mv {{ color:var(--chalk-2); font-size:.9rem; }}
a {{ color:var(--yellow); }}\ntbody tr:nth-child(-n+5) td:nth-child(3)::after {{ content:" ●"; color:var(--yellow); }}\ntbody tr:nth-last-child(-n+5) td:nth-child(3)::after {{ content:" ○"; color:var(--chalk); }}
</style></head><body><div class="wrap">
<header><a href="/">{html.escape(BRAND.upper())}</a><span style="color:var(--chalk-2)">Week {week}</span></header>
<h1>How predictable is your offense?</h1>
<p>Share of pass/run calls our model predicted before the snap, {season} weeks 1–{week}. Neutral situations only
(downs 1–3, close game, outside the last two minutes of each half), so garbage time and obvious downs don't count.
League average: {pct(neutral_avg)}. ● Open Book = top five, ○ Poker Face = bottom five. New rankings every Tuesday.</p>
<img src="{img}" alt="Bar chart ranking all NFL offenses by predictability; the table below has the same numbers." width="1200" height="1500">
<table><thead><tr><th>#</th><th>Move</th><th>Offense</th><th class=n>Neutral</th><th class=n>All plays</th></tr></thead>
<tbody>{rows}</tbody></table>
<p style="margin-top:1.5rem"><a href="/">How the model works →</a> · Follow <a href="https://x.com/TelegraphedNFL">{HANDLE}</a></p>
</div></body></html>
""")


page()
print(f"{out}.md\n{out}_card.png\nsite/weekly/index.html\n")
print("\n\n".join(tweets))
