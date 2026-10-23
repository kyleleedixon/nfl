#!/usr/bin/env bash
# Tuesday refresh: new nflverse week -> features -> re-train current season's live model -> report + card -> push.
# Runs in GitHub Actions (.github/workflows/weekly-refresh.yml) or locally. Vercel deploys site/ on push.
# Exits quietly if there's no new week yet, so it can be scheduled more than once on Tuesday.
# WEEK=n overrides the week, FORCE=1 reruns a week already reported, NODEPLOY=1 stops before commit/push.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONPATH="$PWD/tools/pyexit"   # avoids a pyarrow hang at interpreter exit
SEASON=${SEASON:-2026}
mkdir -p data/raw data/ref data/preds_v3
if [[ -z "${CI:-}" ]]; then
  exec >>data/weekly.log 2>&1
  caffeinate -dimsu -w $$ &
fi
notify() { command -v osascript >/dev/null && osascript -e "display notification \"$1\" with title \"Telegraphed\"" || true; }
trap 'echo "=== $(date "+%F %T") FAILED (line $LINENO)"; notify "Weekly refresh failed"' ERR
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
curl -sfL -o data/raw/new/pbp_$SEASON.parquet $REL/pbp/play_by_play_$SEASON.parquet
curl -sfL -o data/raw/new/part_$SEASON.parquet $REL/pbp_participation/pbp_participation_$SEASON.parquet || rm -f data/raw/new/part_$SEASON.parquet
curl -sfL -o data/raw/new/ftn_$SEASON.parquet $REL/ftn_charting/ftn_charting_$SEASON.parquet || rm -f data/raw/new/ftn_$SEASON.parquet

NEW=$($PY -c "import pandas as pd,sys; print(int(pd.read_parquet(sys.argv[1], columns=['week']).week.max()))" data/raw/new/pbp_$SEASON.parquet)
DONE=$($PY -c "import json,sys; w=json.load(open('site/weekly/weekly.json')); print(w['week'] if w['season']==int(sys.argv[1]) else 0)" $SEASON 2>/dev/null || echo 0)
WEEK=${WEEK:-$NEW}
echo "latest week in data: $NEW, last report: $DONE, reporting week: $WEEK"
if [[ "$WEEK" -le "$DONE" && -z "${FORCE:-}" ]]; then echo "no new week, nothing to do"; exit 0; fi
mv data/raw/new/*.parquet data/raw/

step "features"
$PY src/features.py data/raw data/features.parquet
$PY src/features_v2.py data/raw data/features.parquet data/features_v2.parquet
$PY src/features_v3.py data/raw data/features_v2.parquet data/features_v3.parquet

step "re-train $SEASON live model (trained on seasons < $SEASON; only this week's plays are new)"
$PY src/backtest_v3.py live "$SEASON"

step "report"
$PY src/weekly_report.py "$SEASON" "$WEEK"
$PY src/showcase_data.py >/dev/null
$PY - <<'EOF'
import json
R = json.load(open("site/results.json"))
assert 0.7 < R["headline"]["acc"] < 0.95 and R["live_2026"]["plays"] > 0, R["headline"]
print("site data ok")
EOF

if [[ -n "${NODEPLOY:-}" ]]; then step "NODEPLOY set: skipping commit and push"; exit 0; fi
step "commit + push (Vercel deploys site/ from GitHub on push)"
git add site/results.json site/weekly state
git commit -q -m "Weekly refresh: $SEASON week $WEEK

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing to commit"
git pull -q --rebase origin main
git push origin main
notify "Week $WEEK rankings are live"
step "done"
