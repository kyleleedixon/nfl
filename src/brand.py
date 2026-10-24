"""Shared brand constants for the weekly report and game cards: names, hashtags, site color tokens, fonts."""
import json
import os

BRAND = os.environ.get("BRAND", "Telegraphed")
HANDLE = "@TelegraphedNFL"
SITE = os.environ.get("SITE", "telegraphed.app")

NICK = dict(ARI="Cardinals", ATL="Falcons", BAL="Ravens", BUF="Bills", CAR="Panthers", CHI="Bears",
            CIN="Bengals", CLE="Browns", DAL="Cowboys", DEN="Broncos", DET="Lions", GB="Packers",
            HOU="Texans", IND="Colts", JAX="Jaguars", KC="Chiefs", LA="Rams", LAC="Chargers", LV="Raiders",
            MIA="Dolphins", MIN="Vikings", NE="Patriots", NO="Saints", NYG="Giants", NYJ="Jets",
            PHI="Eagles", PIT="Steelers", SF="49ers", SEA="Seahawks", TB="Buccaneers", TEN="Titans",
            WAS="Commanders")

# fan hashtags go in the bot's tweets (allowed); official @handles go only in a reply Kyle posts by hand, because
# X's automation rules forbid unsolicited automated @mentions
TAG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hashtags.json")
TAGS = json.load(open(TAG_FILE))["teams"]
TAG = {t: v["tag"] for t, v in TAGS.items()}


def unchecked_tags(season):
    """Teams whose hashtag nobody has confirmed for this season."""
    return [t for t, v in TAGS.items() if (v["checked"] or 0) < season]


AT = dict(ARI="@AZCardinals", ATL="@AtlantaFalcons", BAL="@Ravens", BUF="@BuffaloBills", CAR="@Panthers",
          CHI="@ChicagoBears", CIN="@Bengals", CLE="@Browns", DAL="@dallascowboys", DEN="@Broncos", DET="@Lions",
          GB="@packers", HOU="@HoustonTexans", IND="@Colts", JAX="@Jaguars", KC="@Chiefs", LA="@RamsNFL",
          LAC="@chargers", LV="@Raiders", MIA="@MiamiDolphins", MIN="@Vikings", NE="@Patriots", NO="@Saints",
          NYG="@Giants", NYJ="@nyjets", PHI="@Eagles", PIT="@steelers", SF="@49ers", SEA="@Seahawks",
          TB="@Buccaneers", TEN="@Titans", WAS="@Commanders")

# site tokens (site/index.html, light mode): cards must look like the site
BOARD, INK, INK2, GRID, PASS, RUN = "#FAFBFA", "#1B1D22", "#5A606B", "#DCE0E5", "#1E5BC6", "#C42B2B"
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")


def use_fonts():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in os.listdir(FONTS):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONTS, f))
    plt.rcParams["font.family"] = ["Barlow"]


LOGOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "logos")   # src/fetch_logos.py


def logo_w(team, size):
    """Width in points of `team`'s logo drawn with longest side `size` points."""
    import matplotlib.pyplot as plt
    path = os.path.join(LOGOS, f"{team}.png")
    if not os.path.exists(path):
        return 0
    h, w = plt.imread(path).shape[:2]
    return size * w / max(h, w)


def logo(target, team, xy, size, coords="figure fraction", offset=(0, 0)):
    """Team logo centered at xy (offset in points), longest side `size` points. target: a Figure or Axes.
    Missing logos are skipped."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    path = os.path.join(LOGOS, f"{team}.png")
    if not os.path.exists(path):
        return
    im = plt.imread(path)
    box = AnnotationBbox(OffsetImage(im, zoom=size / max(im.shape[:2])), xy, xycoords=coords, xybox=offset,
                         boxcoords="offset points", frameon=False, pad=0, annotation_clip=False)
    target.add_artist(box)


def change(fig, x, y, n, suffix="", size=9, color=None, ha="right"):
    """Week-over-week change on a card: a drawn triangle (Barlow has no arrow glyphs) + the number, e.g. ▲ 3 pts.
    x is the right edge (ha="right") or left edge (ha="left") in figure fraction; y is the text's vertical center.
    No change draws a dash. Neutral ink: moving up isn't good or bad."""
    import matplotlib.pyplot as plt
    color = color or INK2
    n = int(round(n))
    if n == 0:
        fig.text(x, y, "–", color="#AEB4BD", fontsize=size, ha=ha, va="center")
        return
    t = fig.text(x, y, f"{abs(n)}{suffix}", color=color, fontsize=size, ha=ha, va="center")
    tw = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / fig.get_figwidth()
    mw = size * 0.75 / 72 / fig.get_figwidth()          # triangle width plus gap, figure fraction
    if ha == "left":
        t.set_x(x + mw)
        mx = x + mw * 0.35
    else:
        mx = x - tw - mw * 0.65
    fig.lines.append(plt.Line2D([mx], [y], transform=fig.transFigure, marker="^" if n > 0 else "v", ms=size * 0.55,
                                color=color, lw=0))


def arrow(n, unit=""):
    """Change since last week: ▲3 / ▼2 / – (neutral ink everywhere: moving up isn't good or bad)."""
    if n is None or n != n:
        return ""
    n = int(round(n))
    return f"▲{n}{unit}" if n > 0 else (f"▼{-n}{unit}" if n < 0 else "–")


def pct(x):
    return f"{100 * x:.0f}%"
