"""Weekly predictability report: which offenses were easiest / hardest to read before the snap.

Usage: python src/weekly_report.py [season] [week]   (defaults: latest season and week in preds)
Writes data/reports/{season}_wk{week:02d}.md (rankings + draft tweet thread) and _card.png (share image),\nand site/weekly/weekly.json + card PNGs for the public page (site/weekly/index.html is a hand-written template).
Aggregate results only: no feature names or tells (same rule as site/).
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
json.dump(dict(season=season, week=week, card=f"{out}_card.png", tweets=tweets), open(f"{out}_thread.json", "w"), indent=1)


# site tokens (site/index.html, light mode): the card must look like the site
BOARD, INK, INK2, GRID, PASS = "#FAFBFA", "#1B1D22", "#5A606B", "#DCE0E5", "#1E5BC6"
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")


def card(path):
    """Share image (1200x1500), a still of the /weekly page: header, headline tiles, and the ranked dot chart.
    Open Book (above average) = solid dot, Poker Face (below) = hollow dot, dashed line = league average."""
    from matplotlib import font_manager
    for f in os.listdir(FONTS):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONTS, f))
    plt.rcParams["font.family"] = ["Barlow"]
    disp = dict(family="Barlow Condensed")
    fig = plt.figure(figsize=(8, 10), dpi=150, facecolor=BOARD)
    line = lambda y, lw, c=INK: fig.lines.append(plt.Line2D([0.06, 0.94], [y, y], transform=fig.transFigure, color=c, lw=lw))

    # header, as on the site
    fig.text(0.06, 0.952, "TELEGRAPHED", color=INK, fontsize=17, weight="heavy", va="baseline", **disp)
    fig.text(0.94, 0.952, f"{season} · Week {week}", color=INK2, fontsize=10, ha="right", va="baseline")
    line(0.938, 1.6)
    fig.text(0.06, 0.875, "How predictable is your offense?", color=INK, fontsize=33, weight="bold", **disp)
    fig.text(0.06, 0.848, f"Share of pass/run calls we predicted before the snap, neutral situations, weeks 1–{week}",
             color=INK2, fontsize=10.5)
    # headline tiles
    top, bot = ytd.iloc[0], ytd.iloc[-1]
    for x, v, k, c in [(0.06, top.acc_neutral, f"{NICK[top.posteam]}, the most open book", INK),
                       (0.38, neutral_avg, "League average", INK),
                       (0.70, bot.acc_neutral, f"{NICK[bot.posteam]}, the best poker face", "#9AA1AC")]:
        fig.text(x, 0.775, pct(v), color=c, fontsize=36, weight="bold", **disp)
        fig.text(x, 0.757, k, color=INK2, fontsize=9.5)
    line(0.735, 0.8, GRID)

    # key
    ky = 0.712
    fig.lines.append(plt.Line2D([0.066], [ky], transform=fig.transFigure, marker="o", ms=6.5, color=PASS, lw=0))
    fig.text(0.078, ky, "Open Book: above average", color=INK2, fontsize=9.5, va="center")
    fig.lines.append(plt.Line2D([0.33], [ky], transform=fig.transFigure, marker="o", ms=6.5, mfc=BOARD, mec=PASS, mew=1.8, lw=0))
    fig.text(0.342, ky, "Poker Face: below average", color=INK2, fontsize=9.5, va="center")
    fig.lines.append(plt.Line2D([0.60, 0.60], [ky - 0.008, ky + 0.008], transform=fig.transFigure, color=INK2, lw=1.2, ls=(0, (2, 2))))
    fig.text(0.612, ky, "League average", color=INK2, fontsize=9.5, va="center")

    # ranked dot chart, same marks as the page
    ax = fig.add_axes([0.25, 0.085, 0.69, 0.605], facecolor=BOARD)
    n = len(ytd)
    yy = list(range(n))[::-1]
    v = ytd.acc_neutral.values
    lo, hi = np.floor((v.min() - .02) * 50) / 50, np.ceil((v.max() + .02) * 50) / 50
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axvline(neutral_avg, color=INK2, lw=1.1, ls=(0, (4, 4)), zorder=2)
    for yi, r in zip(yy, ytd.itertuples()):
        up = r.acc_neutral >= neutral_avg
        ax.plot([neutral_avg, r.acc_neutral], [yi, yi], color=PASS, alpha=.35, lw=3.2, solid_capstyle="round", zorder=3)
        ax.plot(r.acc_neutral, yi, "o", ms=7.5, mfc=PASS if up else BOARD, mec=PASS, mew=0 if up else 2, zorder=4)
        ax.text(r.acc_neutral + (0.0045 if up else -0.0045), yi, pct(r.acc_neutral), ha="left" if up else "right",
                va="center", color=INK, fontsize=8.5, zorder=5)
        ax.annotate(f"{r.rank}", (0, yi), xycoords=("axes fraction", "data"), xytext=(-120, 0), textcoords="offset points",
                    ha="right", va="center", color=INK2, fontsize=8.5)
        ax.annotate(NICK[r.posteam], (0, yi), xycoords=("axes fraction", "data"), xytext=(-112, 0), textcoords="offset points",
                    ha="left", va="center", color=INK, fontsize=10)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5, length=0, pad=6)
    ax.grid(axis="x", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # footer, as on the site
    line(0.05, 1.6)
    fig.text(0.06, 0.024, HANDLE, color=INK, fontsize=10.5, weight="semibold")
    fig.text(0.94, 0.024, f"{SITE}/weekly", color=INK2, fontsize=10, ha="right")
    fig.savefig(path, facecolor=BOARD)


card(f"{out}_card.png")


def page():
    """Data for the public site/weekly/ page (a static template that fetches weekly.json), plus the share card.
    Aggregate results only."""
    import json
    import shutil
    os.makedirs("site/weekly", exist_ok=True)
    img = f"{season}_wk{week:02d}.png"
    shutil.copy(f"{out}_card.png", f"site/weekly/{img}")
    shutil.copy(f"{out}_card.png", "site/weekly/latest.png")  # stable URL for link previews
    data = dict(
        season=season, week=week, card=img,
        league_acc=round(float(league["acc"]), 4), league_neutral=round(float(neutral_avg), 4),
        teams=[dict(rank=int(r.rank), team=NICK[r.posteam], move=int(r.move), neutral=round(float(r.acc_neutral), 4),
                    acc=round(float(r.acc), 4), snaps=int(r.neutral)) for r in ytd.itertuples()],
        games=[dict(team=NICK[r.posteam], opp=NICK[opp[r.posteam]], acc=round(float(r.acc), 4), plays=int(r.plays))
               for r in wk.itertuples()],
    )
    json.dump(data, open("site/weekly/weekly.json", "w"), indent=1)


page()
print(f"{out}.md\n{out}_card.png\nsite/weekly/weekly.json\n")
print("\n\n".join(tweets))
