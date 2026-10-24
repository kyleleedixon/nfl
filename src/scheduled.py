"""Scheduled posts, written to state/queue/{key}.json for post_x.py (each carries post_at / expires, epoch UTC):

- previews:  night games (kickoff 7pm ET or later) and every playoff game; a card with both offenses' season ranking,
             posted 2 hours before kickoff. refresh.sh builds them a few days out and rebuilds them until posted, so they
             carry the latest rankings (the final regular-season ones in the playoffs).
- surprises: the week's biggest surprises: the 5 most confident wrong calls league-wide (one per offense), posted
             the evening the rankings come out.

usage: scheduled.py previews SEASON           (night and playoff games posting in the next 3 days, latest rankings)
       scheduled.py surprises SEASON WEEK
Results only (read rates, ranks, the model's per-play calls), never features or tells.
"""
import datetime as dt
import json
import os
import sys
import time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brand import HANDLE, SITE, NICK, TAG, BOARD, INK, INK2, GRID, PASS, RUN, use_fonts, pct, logo, logo_w, change, matchup_tag, sure
from game_cards import load, surprise

D = os.environ.get("DATA", "data")
ET = ZoneInfo("America/New_York")
Q = "state/queue"
NIGHT = {"Thursday": "TNF", "Sunday": "SNF", "Monday": "MNF", "Saturday": "Saturday night", "Friday": "Friday night"}


def kickoff(r):
    return dt.datetime.strptime(f"{r.gameday} {r.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def queue(key, kind, card, tweet, post_at, expires):
    os.makedirs(Q, exist_ok=True)
    assert len(tweet) <= 280, (key, len(tweet))
    json.dump(dict(key=key, kind=kind, card=card, tweets=[tweet], post_at=int(post_at), expires=int(expires),
                   created=int(time.time())), open(f"{Q}/{key}.json", "w"), indent=1)
    print(key, "posts", dt.datetime.fromtimestamp(post_at, ET).strftime("%a %b %-d %-I:%M%p ET"), "\n" + tweet + "\n")


def header(fig, right, y=0.93, lw=1.6):
    disp = dict(family="Barlow Condensed")
    fig.text(0.05, y, "TELEGRAPHED", color=INK, fontsize=15, weight="heavy", va="baseline", **disp)
    fig.text(0.95, y, right, color=INK2, fontsize=9.5, ha="right", va="baseline")
    fig.lines.append(plt.Line2D([0.05, 0.95], [y - 0.025, y - 0.025], transform=fig.transFigure, color=INK, lw=lw))


def footer(fig, y=0.05, right=SITE):
    fig.lines.append(plt.Line2D([0.05, 0.95], [y + 0.03, y + 0.03], transform=fig.transFigure, color=INK, lw=1.6))
    fig.text(0.05, y, HANDLE, color=INK, fontsize=10, weight="semibold", va="center")
    fig.text(0.95, y, right, color=INK2, fontsize=9.5, ha="right", va="center")


def previews(season, days=3):
    """Previews for night games (kickoff 7pm ET or later) and every playoff game in the next `days` days, with the latest
    rankings. Re-run freely: a preview is rebuilt (fresher rankings) until it has been posted."""
    W = json.load(open("site/weekly/weekly.json"))
    if W["season"] != season:
        return print(f"no {season} rankings yet; no previews")
    T = {t["abbr"]: t for t in W["teams"]}
    posted = json.load(open("state/x_posted.json")) if os.path.exists("state/x_posted.json") else {}
    sch = pd.read_csv(f"{D}/raw/schedule.csv")
    sch = sch[(sch.season == season) & ((sch.gametime >= "19:00") | (sch.game_type != "REG"))]
    now = time.time()
    for r in sch.itertuples():
        if not isinstance(r.gametime, str) or not isinstance(r.away_team, str):
            continue   # playoff slots before the matchup is set
        ko = kickoff(r)
        key = f"preview_{r.game_id}"
        if not now < ko.timestamp() - 2 * 3600 < now + days * 86400 or posted.get(key, {}).get("complete"):
            continue
        if r.away_team not in T or r.home_team not in T:
            continue
        slot = {"WC": "Wild Card", "DIV": "Divisional round", "CON": "Conference championship", "SB": "Super Bowl"}.get(
            r.game_type, NIGHT.get(r.weekday, "Tonight"))
        card = f"site/previews/{r.game_id}.png"
        os.makedirs("site/previews", exist_ok=True)
        preview_card(card, r, ko, slot, T, W)
        lines = []
        for t in [r.away_team, r.home_team]:
            x = T[t]
            d = round(100 * x["neutral"]) - round(100 * W["league_neutral"])
            lines.append(f"{'📖' if d > 0 else '🃏' if d < 0 else '➖'} {NICK[t]}: {ordinal(x['rank'])} of {len(T)} ({pct(x['neutral'])})")
        season_txt = "Season so far" if r.game_type == "REG" else "Regular season"
        tw = (f"{slot}: {NICK[r.away_team]} at {NICK[r.home_team]}\n\nHow predictable are these offenses? {season_txt}:\n"
              + "\n".join(lines) + f"\nNFL average: {pct(W['league_neutral'])}\n\n#NFL {TAG[r.away_team]} {TAG[r.home_team]} {matchup_tag(r.away_team, r.home_team)}")
        queue(key, "preview", card, tw, ko.timestamp() - 2 * 3600, ko.timestamp())


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def preview_card(path, r, ko, slot, T, W):
    """1600x900: both offenses' season-to-date neutral read rate, rank (and change) and label."""
    use_fonts()
    disp = dict(family="Barlow Condensed")
    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor=BOARD)
    when = ko.strftime("%a %b %-d · %-I:%M%p ET").replace("AM", "am").replace("PM", "pm")
    header(fig, f"{slot} · {when}")
    span = "week 1" if W["week"] == 1 else f"weeks 1–{W['week']}"
    fig.text(0.05, 0.775, f"{NICK[r.away_team]} at {NICK[r.home_team]}", color=INK, fontsize=26, weight="bold", **disp)
    fig.text(0.05, 0.715, f"How predictable is each offense? Share of pass/run calls we predicted before the snap, "
             f"neutral situations, {span}.", color=INK2, fontsize=8.5)
    for x0, t in [(0.05, r.away_team), (0.52, r.home_team)]:
        s = T[t]
        logo(fig, t, (x0 + 0.055, 0.50), 54)
        fig.text(x0 + 0.13, 0.565, NICK[t].upper(), color=INK, fontsize=12, weight="bold", **disp)
        fig.text(x0 + 0.13, 0.40, pct(s["neutral"]), color=INK, fontsize=40, weight="bold", **disp)
        d = round(100 * s["neutral"]) - round(100 * W["league_neutral"])
        lab = "Open Book" if d > 0 else ("Poker Face" if d < 0 else "League average")
        fig.text(x0 + 0.27, 0.49, f"{ordinal(s['rank'])} of {len(T)}", color=INK, fontsize=11, weight="semibold")
        if s.get("move"):
            change(fig, x0 + 0.375, 0.502, s["move"], size=9.5, ha="left")
        fig.text(x0 + 0.27, 0.43, lab, color=INK2, fontsize=10)
    fig.lines.append(plt.Line2D([0.05, 0.95], [0.30, 0.30], transform=fig.transFigure, color=GRID, lw=0.8))
    fig.text(0.05, 0.215, f"NFL average {pct(W['league_neutral'])}. Open Book = easier to read than average; Poker Face = harder. "
             "Rank 1 = most predictable.", color=INK2, fontsize=8.5)
    footer(fig, right=f"{SITE}/weekly")
    fig.savefig(path, facecolor=BOARD)
    plt.close(fig)


def surprises(season, week):
    if os.path.exists(f"{Q}/surprises_{season}_wk{week:02d}.json") and not os.environ.get("REBUILD_SURPRISES"):
        return print(f"week {week} surprises already queued")
    P, charted, _ = load(season)
    G = P[(P.week == week) & ~P.rpo]
    rows = []
    for (gid, t), g in G.groupby(["game_id", "posteam"]):
        su = surprise(g)
        if su:
            r, when, what = su
            rows.append(dict(team=t, opp=r.defteam, conf=max(r.p, 1 - r.p), when=when, what=what, y=int(r.y)))
    top = sorted(rows, key=lambda x: -x["conf"])[:5]
    card = f"site/surprises/{season}_wk{week:02d}.png"
    os.makedirs("site/surprises", exist_ok=True)
    surprise_card(card, season, week, top)
    lines = [f"{i}. {NICK[x['team']]}: {sure(x['conf'])} sure, wrong" for i, x in enumerate(top, 1)]
    tw = (f"Week {week}: the 5 calls that fooled us most\n\n" + "\n".join(lines)
          + f"\n\n#NFL {TAG[top[0]['team']]} {TAG[top[1]['team']]}")
    now = dt.datetime.now(dt.timezone.utc)
    # the evening the rankings come out (Wednesday 6pm CT = 23:00 UTC); later in the week: an hour from now
    wed = (now + dt.timedelta(days=(2 - now.weekday()) % 7)).replace(hour=23, minute=0, second=0, microsecond=0)
    post = wed if now.weekday() <= 2 and wed > now else now + dt.timedelta(hours=1)
    queue(f"surprises_{season}_wk{week:02d}", "surprises", card, tw, post.timestamp(), post.timestamp() + 3 * 86400)


def surprise_card(path, season, week, top):
    """1200x1500: the five most confident wrong calls of the week, one per offense."""
    use_fonts()
    disp = dict(family="Barlow Condensed")
    fig = plt.figure(figsize=(8, 10), dpi=150, facecolor=BOARD)
    header(fig, f"{season} · Week {week}", y=0.952)
    fig.text(0.05, 0.875, "The calls that fooled us", color=INK, fontsize=33, weight="bold", **disp)
    fig.text(0.05, 0.848, f"Week {week}'s five most confident wrong calls, before the snap. One per offense.", color=INK2, fontsize=10.5)
    for i, x in enumerate(top):
        y = 0.765 - i * 0.142
        fig.patches.append(matplotlib.patches.FancyBboxPatch((0.05, y - 0.058), 0.90, 0.116, boxstyle="round,pad=0,rounding_size=0.012",
                                                             transform=fig.transFigure, fc="#EEF2F7", ec="none"))
        fig.text(0.075, y, str(i + 1), color=INK2, fontsize=26, weight="bold", va="center", **disp)
        logo(fig, x["team"], (0.165, y + 0.012), 34)
        fig.text(0.215, y + 0.022, f"{NICK[x['team']]} vs {NICK[x['opp']]}", color=INK, fontsize=15, weight="bold", va="center", **disp)
        fig.text(0.215, y - 0.010, x["when"], color=INK2, fontsize=10, va="center")
        fig.text(0.215, y - 0.036, x["what"], color=INK, fontsize=10.5, va="center")
        fig.text(0.925, y + 0.016, sure(x["conf"]), color=PASS if x["y"] == 0 else RUN, fontsize=30, weight="bold", ha="right", va="center", **disp)
        fig.text(0.925, y - 0.024, "sure, and wrong", color=INK2, fontsize=9.5, ha="right", va="center")
    fig.text(0.05, 0.072, "Blue: we expected a pass. Red: we expected a run. RPOs not scored.", color=INK2, fontsize=9.5)
    footer(fig, y=0.024)
    fig.savefig(path, facecolor=BOARD)
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "previews":
        previews(int(sys.argv[2]))
    else:
        surprises(int(sys.argv[2]), int(sys.argv[3]))
