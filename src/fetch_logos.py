"""Downloads team logos (nflverse team table -> ESPN 500px PNGs) to assets/logos/{TEAM}.png at 200px for the cards,
and a 64px copy to site/logos/ for the /weekly page.
Re-run after a rebrand. Logos are the teams' trademarks; used only to identify teams on result cards."""
import io
import os
import pandas as pd
import requests
from PIL import Image

URL = "https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.csv"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT, WEB = os.path.join(ROOT, "assets", "logos"), os.path.join(ROOT, "site", "logos")
TEAMS = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
         "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"}

os.makedirs(OUT, exist_ok=True)
os.makedirs(WEB, exist_ok=True)
t = pd.read_csv(URL)
for r in t[t.team_abbr.isin(TEAMS)].itertuples():
    im = Image.open(io.BytesIO(requests.get(r.team_logo_espn, timeout=30).content)).convert("RGBA")
    im = im.crop(im.getbbox())   # trim transparent margin so logos size consistently
    im.thumbnail((200, 200), Image.LANCZOS)
    im.save(f"{OUT}/{r.team_abbr}.png", optimize=True)
    im.thumbnail((64, 64), Image.LANCZOS)
    im.save(f"{WEB}/{r.team_abbr}.png", optimize=True)
missing = TEAMS - {f[:-4] for f in os.listdir(OUT)}
print(f"{32 - len(missing)} logos in assets/logos", f"missing: {missing}" if missing else "")
