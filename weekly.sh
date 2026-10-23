#!/usr/bin/env bash
# Tuesday refresh: new nflverse week -> features -> re-score current season -> weekly report + card -> deploy site.
# Exits quietly if there's no new week yet (so it can be scheduled more than once on Tuesday).
# Tweet drafts land in data/reports/; nothing is posted. Log: data/weekly.log.
# WEEK=n overrides the week, FORCE=1 reruns a week already reported, NODEPLOY=1 stops before deploy/push.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONPATH="$PWD/tools/pyexit"   # avoids a pyarrow hang at interpreter exit
SEASON=${SEASON:-2026}
exec >>data/weekly.log 2>&1
caffeinate -dimsu -w $$ &
trap 'echo "=== $(date "+%F %T") FAILED (line $LINENO)"; osascript -e "display notification \"Weekly refresh failed, see data/weekly.log\" with title \"NFL weekly\"" || true' ERR
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "download $SEASON"
REL=https://github.com/nflverse/nflverse-data/releases/download
mkdir -p data/raw/new
curl -sfL -o data/raw/new/pbp_$SEASON.parquet $REL/pbp/play_by_play_$SEASON.parquet
curl -sfL -o data/raw/new/part_$SEASON.parquet $REL/pbp_participation/pbp_participation_$SEASON.parquet || rm -f data/raw/new/part_$SEASON.parquet
curl -sfL -o data/raw/new/ftn_$SEASON.parquet $REL/ftn_charting/ftn_charting_$SEASON.parquet || rm -f data/raw/new/ftn_$SEASON.parquet
week_of() { $PY -c "import pandas as pd,sys; print(int(pd.read_parquet(sys.argv[1], columns=['week']).week.max()))" "$1"; }
NEW=$(week_of data/raw/new/pbp_$SEASON.parquet)
DONE=$(cat data/weekly.week 2>/dev/null || echo 0)
WEEK=${WEEK:-$NEW}
echo "latest week in data: $NEW, last report: $DONE, reporting week: $WEEK"
if [[ "$WEEK" -le "$DONE" && -z "${FORCE:-}" ]]; then echo "no new week, nothing to do"; exit 0; fi
mv data/raw/new/*.parquet data/raw/

step "features"
$PY src/features.py data/raw data/features.parquet
$PY src/features_v2.py data/raw data/features.parquet data/features_v2.parquet
$PY src/features_v3.py data/raw data/features_v2.parquet data/features_v3.parquet

step "re-score $SEASON (models are trained on seasons < $SEASON, so only predictions change)"
rm -f data/preds_v3/${SEASON}_*.parquet
$PY src/backtest_v3.py run

step "report"
$PY src/weekly_report.py "$SEASON" "$WEEK"
$PY src/showcase_data.py >/dev/null
$PY - <<'EOF'
import json
R = json.load(open("site/results.json"))
assert 0.7 < R["headline"]["acc"] < 0.95 and R["live_2026"]["plays"] > 0, R["headline"]
print("site data ok")
EOF

if [[ -n "${NODEPLOY:-}" ]]; then step "NODEPLOY set: skipping deploy and push"; exit 0; fi
step "deploy"
(cd site && vercel deploy --prod --yes)
git add site/results.json site/weekly
git commit -q -m "Weekly refresh: $SEASON week $WEEK

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing to commit"
git push origin main

echo "$WEEK" > data/weekly.week
osascript -e "display notification \"Week $WEEK report ready: data/reports\" with title \"NFL weekly\"" || true
step "done"
