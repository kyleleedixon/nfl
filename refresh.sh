#!/usr/bin/env bash
# In-season refresh, run every few hours (GitHub Actions: .github/workflows/refresh.yml) or locally.
# Waits for FTN charting (the live model needs it; nflverse publishes it ~1-2 days after a game), then:
# features -> re-train current season's live model -> a card for each newly charted game -> the weekly rankings once
# FTN covers every game of a week -> push (Vercel deploys site/; the post workflows tweet from state/).
# Exits early when nothing new is charted. WEEK=n FORCE=1 re-runs a week's rankings; NODEPLOY=1 stops before push.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONPATH="$PWD/tools/pyexit"   # avoids a pyarrow hang at interpreter exit
SEASON=${SEASON:-$($PY src/season.py)}   # rolls over each September (src/season.py)
mkdir -p data/raw data/ref data/preds_v3
if [[ -z "${CI:-}" ]]; then
  exec >>data/refresh.log 2>&1
  caffeinate -dimsu -w $$ &
fi
notify() { command -v osascript >/dev/null && osascript -e "display notification \"$1\" with title \"Telegraphed\"" || true; }
trap 'echo "=== $(date "+%F %T") FAILED (line $LINENO)"; notify "Refresh failed"' ERR
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "inputs"
# committed inputs a fresh checkout needs (see artifacts/README.md); never overwrite local copies
cp -n artifacts/preds_v3/*.parquet data/preds_v3/ 2>/dev/null || true
[[ -f data/ablate_v3.json ]] || cp artifacts/ablate_v3.json data/
[[ -f data/ref/players.parquet ]] || cp artifacts/players.parquet data/ref/
REL=https://github.com/nflverse/nflverse-data/releases/download
get() { [[ -f "data/raw/$2" ]] || curl -sfL -o "data/raw/$2" "$REL/$1" || rm -f "data/raw/$2"; }
for y in $(seq 2016 $((SEASON - 1))); do   # past seasons: download once
  get pbp/play_by_play_$y.parquet pbp_$y.parquet
  get pbp_participation/pbp_participation_$y.parquet part_$y.parquet
  if [[ $y -ge 2022 ]]; then get ftn_charting/ftn_charting_$y.parquet ftn_$y.parquet; fi
done
mkdir -p data/raw/new   # current season: always fresh
if ! curl -sfL -o data/raw/new/pbp_$SEASON.parquet $REL/pbp/play_by_play_$SEASON.parquet; then
  echo "no $SEASON play-by-play yet (season not started)"; exit 0
fi
curl -sfL -o data/raw/new/part_$SEASON.parquet $REL/pbp_participation/pbp_participation_$SEASON.parquet || rm -f data/raw/new/part_$SEASON.parquet
if ! curl -sfL -o data/raw/new/ftn_$SEASON.parquet $REL/ftn_charting/ftn_charting_$SEASON.parquet; then
  echo "no $SEASON FTN charting yet"; exit 0
fi
curl -sfL -o data/raw/schedule.csv https://github.com/nflverse/nfldata/raw/master/data/games.csv

# what's new: charted games (in play-by-play and FTN) from the last 6 days without a card, and the latest week
# whose every scheduled game is charted
read -r NEWG WEEKDONE <<<"$($PY - "$SEASON" <<'EOF'
import json, os, sys
import pandas as pd
S = int(sys.argv[1])
pbp = set(pd.read_parquet(f"data/raw/new/pbp_{S}.parquet", columns=["game_id"]).game_id)
ftn = set(pd.read_parquet(f"data/raw/new/ftn_{S}.parquet", columns=["nflverse_game_id"]).nflverse_game_id)
sch = pd.read_csv("data/raw/schedule.csv")
sch = sch[(sch.season == S) & (sch.game_type == "REG")]
charted = pbp & ftn
recent = set(sch.game_id[pd.to_datetime(sch.gameday) >= pd.Timestamp.now().normalize() - pd.Timedelta(days=6)])
new = [g for g in charted & recent if not os.path.exists(f"state/games/{g}.json")]
full = [w for w, g in sch.groupby("week") if set(g.game_id) <= charted]
print(len(new), max(full) if full else 0)
EOF
)"
DONE=$($PY -c "import json,sys; w=json.load(open('site/weekly/weekly.json')); print(w['week'] if w['season']==int(sys.argv[1]) else 0)" $SEASON 2>/dev/null || echo 0)
WEEK=${WEEK:-$WEEKDONE}
echo "new charted games: $NEWG; latest fully charted week: $WEEKDONE; last rankings: week $DONE"
RANK=""
if [[ "$WEEK" -gt "$DONE" || -n "${FORCE:-}" ]] && [[ "$WEEK" -gt 0 ]]; then RANK=1; fi
if [[ "$NEWG" -eq 0 && -z "$RANK" ]]; then echo "nothing new charted, nothing to do"; exit 0; fi
mv data/raw/new/*.parquet data/raw/

step "features"
$PY src/features.py data/raw data/features.parquet
$PY src/features_v2.py data/raw data/features.parquet data/features_v2.parquet
$PY src/features_v3.py data/raw data/features_v2.parquet data/features_v3.parquet

step "re-train $SEASON live model (trained on seasons < $SEASON; only new plays change)"
$PY src/backtest_v3.py live "$SEASON"

step "game cards"
$PY src/game_cards.py "$SEASON"
if [[ -n "$RANK" ]]; then
  step "weekly rankings: week $WEEK"
  $PY src/weekly_report.py "$SEASON" "$WEEK"
  # scheduled posts (state/queue/): the week's biggest surprises, and next week's prime-time previews off these rankings
  $PY src/scheduled.py surprises "$SEASON" "$WEEK" || echo "::warning::surprises card failed"
  $PY src/scheduled.py previews "$SEASON" $((WEEK + 1)) || echo "::warning::prime-time previews failed"
fi
# hashtags nobody has confirmed for this season (src/hashtags.json): a warning on the Actions run, not a failure
$PY -c "import os,sys; sys.path.insert(0,'src'); from brand import unchecked_tags; u=unchecked_tags($SEASON)
if u: print(('::warning::' if os.environ.get('GITHUB_ACTIONS') else '') + f'hashtags not checked for $SEASON: {\", \".join(u)} (src/hashtags.json)')"
$PY src/showcase_data.py >/dev/null
$PY - <<'EOF'
import json
R = json.load(open("site/results.json"))
assert 0.7 < R["headline"]["acc"] < 0.95 and R["live"]["plays"] > 0, R["headline"]
print("site data ok")
EOF

if [[ -n "${NODEPLOY:-}" ]]; then step "NODEPLOY set: skipping commit and push"; exit 0; fi
step "commit + push (Vercel deploys site/ from GitHub on push)"
git add site/results.json site/weekly site/games site/previews site/surprises state
git commit -q -m "Refresh: $SEASON, $NEWG new game card(s)${RANK:+, week $WEEK rankings}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing to commit"
git pull -q --rebase origin main
git push origin main
notify "Refresh done: $NEWG game card(s)${RANK:+, week $WEEK rankings}"
step "done"
