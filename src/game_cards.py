"""Per-game cards: how predictable was each offense in one game, vs the league average (Umpire Scorecards style).

Usage: python src/game_cards.py [season] [game_id ...]   (default: ready games from the last 6 days without a card)
A game is ready once FTN has charted it (the live model leans on FTN; without it accuracy drops 2-4 pts).
Writes site/games/{game_id}.png (card) and state/games/{game_id}.json (tweet + card path, posted by post_x.py games).
Results only: read rates and the model's per-play calls, never features or tells (same rule as site/).
RPOs are drawn but not scored (src/scoring.py).
"""
import json
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brand import HANDLE, SITE, NICK, TAG, BOARD, INK, INK2, GRID, PASS, RUN, use_fonts, pct
from scoring import mark_rpo, NOTE

D = os.environ.get("DATA", "data")
SURE = 0.75  # "we were sure": model at least this confident


def load(season):
    P = pd.read_parquet(f"{D}/preds_v3.parquet")
    P = P[P.season == season].copy()
    P["p"] = P.full_v3.fillna(P.live_v3)
    P["hit"] = (P.p > 0.5) == P.y
    P = mark_rpo(P, season)
    pbp = pd.read_parquet(f"{D}/raw/pbp_{season}.parquet",
                          columns=["game_id", "play_id", "home_team", "away_team", "home_score", "away_score",
                                   "game_seconds_remaining", "time", "yards_gained", "game_date", "qtr"])
    pbp["play_id"] = pbp.play_id.astype(float)
    P = P.merge(pbp.drop(columns="qtr"), on=["game_id", "play_id"], how="left")
    ftn = pd.read_parquet(f"{D}/raw/ftn_{season}.parquet", columns=["nflverse_game_id"])
    return P, set(ftn.nflverse_game_id), pbp


def team_stats(g, league):
    n = g[g.neutral == 1]
    sure = g[(g.p >= SURE) | (g.p <= 1 - SURE)]
    return dict(acc=g.hit.mean(), neutral=n.hit.mean() if len(n) else np.nan, plays=len(g),
                sure=len(sure), sure_missed=int((~sure.hit).sum()), pass_rate=g.y.mean(),
                vs=g.hit.mean() - league)


def surprise(g):
    """Most confident wrong call: (play row, situation line, what happened line)."""
    m = g[~g.hit]
    if m.empty:
        return None
    r = m.loc[(m.p - 0.5).abs().idxmax()]
    sure = r.p if r.p > 0.5 else 1 - r.p
    exp, did = ("pass", "ran") if r.p > 0.5 else ("run", "passed")
    yds = int(r.yards_gained) if pd.notna(r.yards_gained) else 0
    gain = f"for {yds} yd{'s' if yds != 1 else ''}" if yds > 0 else ("for no gain" if yds == 0 else f"for a loss of {-yds}")
    q = f"Q{int(r.qtr)}" if r.qtr <= 4 else "OT"
    clock = str(r.time)[1:] if str(r.time).startswith("0") else str(r.time)
    dn = f"{int(r.down)}{['st', 'nd', 'rd', 'th'][min(int(r.down), 4) - 1]} & {int(r.ydstogo)}"
    return r, f"{q} {clock} · {dn}", f"We were {pct(sure)} sure of a {exp}. They {did} {gain}."


def card(path, gid, G, league, league_n, week):
    use_fonts()
    disp = dict(family="Barlow Condensed")
    away, home = G.away_team.iloc[0], G.home_team.iloc[0]
    last = G.sort_values("play_id").iloc[-1]
    fs = pd.read_parquet(f"{D}/raw/pbp_{G.season.iloc[0]}.parquet", columns=["game_id", "home_score", "away_score"])
    fs = fs[fs.game_id == gid].iloc[-1]
    a_s, h_s = int(fs.away_score), int(fs.home_score)
    fig = plt.figure(figsize=(8, 10), dpi=150, facecolor=BOARD)
    line = lambda y, lw, c=INK: fig.lines.append(plt.Line2D([0.06, 0.94], [y, y], transform=fig.transFigure, color=c, lw=lw))

    fig.text(0.06, 0.952, "TELEGRAPHED", color=INK, fontsize=17, weight="heavy", va="baseline", **disp)
    date = pd.Timestamp(last.game_date).strftime("%a %b %-d")
    fig.text(0.94, 0.952, f"Week {week} · {date}", color=INK2, fontsize=10, ha="right", va="baseline")
    line(0.938, 1.6)
    fig.text(0.06, 0.880, f"{NICK[away]} {a_s}  ·  {NICK[home]} {h_s}", color=INK, fontsize=31, weight="bold", **disp)
    fig.text(0.06, 0.853, "How predictable was each offense? Share of pass/run calls we predicted before the snap.",
             color=INK2, fontsize=10.5)

    # two stat columns
    for x, t in [(0.06, away), (0.52, home)]:
        g = G[(G.posteam == t) & ~G.rpo]
        s = team_stats(g, league)
        fig.text(x, 0.812, NICK[t].upper(), color=INK, fontsize=13, weight="bold", **disp)
        fig.text(x, 0.748, pct(s["acc"]), color=INK, fontsize=44, weight="bold", **disp)
        d = round(100 * s["acc"]) - round(100 * league)   # as displayed, so 79% vs 79% reads "even"
        tag = "📖" if d > 0 else ""
        fig.text(x + 0.165, 0.770, f"{'+' if d > 0 else ''}{d} pt{'s' if abs(d) != 1 else ''} vs NFL avg ({pct(league)})" if d else f"Even with NFL avg ({pct(league)})",
                 color=INK, fontsize=10.5, weight="semibold")
        fig.text(x + 0.165, 0.750, "Open Book" if d > 0 else ("Poker Face" if d < 0 else ""), color=INK2, fontsize=10)
        rows = [("Neutral situations", (pct(s["neutral"]) if pd.notna(s["neutral"]) else "–") + f"  (NFL {pct(league_n)})"),
                ("Plays", f"{s['plays']}"),
                (f"Calls we were {pct(SURE)}+ sure of", f"{s['sure']}  ({s['sure_missed']} wrong)"),
                ("Pass rate", pct(s["pass_rate"]))]
        for i, (k, v) in enumerate(rows):
            y = 0.712 - i * 0.024
            fig.text(x, y, k, color=INK2, fontsize=10)
            fig.text(x + 0.40, y, v, color=INK, fontsize=10, ha="right", weight="semibold")
    line(0.627, 0.8, GRID)

    # how to read it
    ky = 0.604
    fig.text(0.06, ky, "Each dot is a play.", color=INK, fontsize=9.5, va="center", weight="semibold")
    fig.lines.append(plt.Line2D([0.215], [ky], transform=fig.transFigure, marker="o", ms=6.5, color=PASS, lw=0))
    fig.text(0.227, ky, "pass", color=INK2, fontsize=9.5, va="center")
    fig.lines.append(plt.Line2D([0.282], [ky], transform=fig.transFigure, marker="o", ms=6.5, color=RUN, lw=0))
    fig.text(0.294, ky, "run", color=INK2, fontsize=9.5, va="center")
    fig.text(0.345, ky, "Higher = we expected a pass. Solid = we called it.", color=INK2, fontsize=9.5, va="center")
    fig.lines.append(plt.Line2D([0.775], [ky], transform=fig.transFigure, marker="o", ms=6.5, mfc=BOARD, mec=INK2, mew=1.8, lw=0))
    fig.text(0.787, ky, "Hollow = fooled us.", color=INK2, fontsize=9.5, va="center")

    # play-by-play panels, one per offense
    end = max(3600, 3600 - G.game_seconds_remaining.min())
    for i, t in enumerate([away, home]):
        g = G[G.posteam == t].sort_values("play_id")
        top = 0.561 - i * 0.257           # team title baseline; surprise banner below it, then the chart
        ax = fig.add_axes([0.15, top - 0.215, 0.79, 0.160], facecolor=BOARD)
        x = (3600 - g.game_seconds_remaining) / 60
        # zones: above the line we expected a pass, below a run
        ax.axhspan(0.5, 1.06, color=PASS, alpha=0.06, lw=0, zorder=0)
        ax.axhspan(-0.06, 0.5, color=RUN, alpha=0.06, lw=0, zorder=0)
        ax.axhline(0.5, color=INK2, lw=1.0, ls=(0, (4, 4)), zorder=2)
        for q in range(15, int(end / 60) + 1, 15):
            ax.axvline(q, color=GRID, lw=0.9, zorder=1)
        col = np.where(g.y == 1, PASS, RUN)
        hit = g.hit.values & ~g.rpo.values
        miss = ~g.hit.values & ~g.rpo.values
        ax.scatter(x[hit], g.p[hit], s=34, c=col[hit], linewidths=0, zorder=4)
        ax.scatter(x[miss], g.p[miss], s=34, facecolors=BOARD, edgecolors=col[miss], linewidths=1.8, zorder=5)
        rp = g.rpo.values   # unscored: small gray diamonds
        ax.scatter(x[rp], g.p[rp], s=26, marker="D", c="#AEB4BD", linewidths=0, zorder=4.5)
        ax.set_xlim(-1, end / 60 + 1)
        ax.set_ylim(-0.06, 1.06)
        ax.set_yticks([0.78, 0.22])
        ax.set_yticklabels(["Expected\npass", "Expected\nrun"], fontsize=9, weight="semibold", linespacing=1.0)
        for lab, c in zip(ax.get_yticklabels(), [PASS, RUN]):
            lab.set_color(c)
        qs = list(range(0, int(end / 60), 15))
        ax.set_xticks([q + 7.5 for q in qs])
        ax.set_xticklabels([f"Q{k + 1}" if k < 4 else "OT" for k in range(len(qs))])
        ax.tick_params(axis="x", colors=INK2, labelsize=8.5, length=0, pad=5)
        ax.tick_params(axis="y", length=0, pad=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(0.06, top, f"{NICK[t]} offense", color=INK, fontsize=12, weight="bold", **disp)
        su = surprise(g[~g.rpo])
        if su:
            r, when, what = su
            sx = (3600 - r.game_seconds_remaining) / 60
            ax.scatter([sx], [r.p], s=330, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=6)
            # banner outside the chart so it never covers plays; the ring icon matches the ringed dot
            by = top - 0.030
            fig.patches.append(matplotlib.patches.FancyBboxPatch((0.06, by - 0.013), 0.88, 0.026, boxstyle="round,pad=0,rounding_size=0.008",
                                                                 transform=fig.transFigure, fc=INK, ec="none"))
            fig.lines.append(plt.Line2D([0.078], [by], transform=fig.transFigure, marker="o", ms=10, mfc="none", mec=BOARD, mew=1.6, lw=0))
            fig.text(0.095, by, f"BIGGEST SURPRISE  ·  {when}", color=BOARD, fontsize=9.5, weight="bold", va="center")
            fig.text(0.928, by, what, color=BOARD, fontsize=9.5, va="center", ha="right")

    line(0.05, 1.6)
    fig.text(0.06, 0.024, HANDLE, color=INK, fontsize=10.5, weight="semibold")
    fig.lines.append(plt.Line2D([0.30], [0.0285], transform=fig.transFigure, marker="D", ms=5, color="#AEB4BD", lw=0))
    fig.text(0.312, 0.024, NOTE, color=INK2, fontsize=9)
    fig.text(0.94, 0.024, SITE, color=INK2, fontsize=10, ha="right")
    fig.savefig(path, facecolor=BOARD)
    plt.close(fig)
    return away, home, a_s, h_s


def tweet(away, home, a_s, h_s, G, league):
    lines = []
    G = G[~G.rpo]
    for t in sorted([away, home], key=lambda t: -G[G.posteam == t].hit.mean()):
        a = G[G.posteam == t].hit.mean()
        d = round(100 * a) - round(100 * league)
        lines.append(f"{'📖' if d > 0 else '🃏' if d < 0 else '➖'} {NICK[t]} {pct(a)}")
    return (f"{NICK[away]} {a_s}, {NICK[home]} {h_s}\n\nHow predictable was each offense?\n" + "\n".join(lines)
            + f"\nNFL average: {pct(league)}\n\n#NFL {TAG[away]} {TAG[home]}")


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else int(pd.read_parquet(f"{D}/preds_v3.parquet", columns=["season"]).season.max())
    P, charted, _ = load(season)
    # default: charted games from the last 6 days without a card (older ones only by explicit id)
    recent = set(P.game_id[pd.to_datetime(P.game_date) >= pd.Timestamp.now().normalize() - pd.Timedelta(days=6)])
    ids = sys.argv[2:] or sorted(g for g in recent if g in charted and not os.path.exists(f"state/games/{g}.json"))
    os.makedirs("site/games", exist_ok=True)
    os.makedirs("state/games", exist_ok=True)
    for gid in ids:
        G = P[P.game_id == gid]
        week = int(G.week.iloc[0])
        # league average: every charted play this season through this game's week, same model
        L = P[P.game_id.isin(charted) & (P.week <= week) & ~P.rpo]
        league, league_n = L.hit.mean(), L[L.neutral == 1].hit.mean()
        img = f"site/games/{gid}.png"
        away, home, a_s, h_s = card(img, gid, G, league, league_n, week)
        t = tweet(away, home, a_s, h_s, G, league)
        assert len(t) <= 280, (gid, len(t))
        json.dump(dict(season=season, week=week, game_id=gid, kickoff=str(G.game_date.iloc[0]), card=img, tweets=[t],
                       created=int(time.time())), open(f"state/games/{gid}.json", "w"), indent=1)
        print(gid, img, "\n" + t + "\n")


if __name__ == "__main__":
    main()
