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


def logo(target, team, xy, size, coords="figure fraction", offset=(0, 0)):
    """Team logo centered at xy (offset in points), longest side `size` points. target: a Figure or Axes."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    path = os.path.join(LOGOS, f"{team}.png")
    if not os.path.exists(path):
        return
    im = plt.imread(path)
    box = AnnotationBbox(OffsetImage(im, zoom=size / max(im.shape[:2])), xy, xycoords=coords, xybox=offset,
                         boxcoords="offset points", frameon=False, pad=0, annotation_clip=False)
    (target.add_artist(box))


def pct(x):
    return f"{100 * x:.0f}%"
