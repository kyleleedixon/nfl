#!/usr/bin/env bash
# Season wrap-up, run monthly March-August (GitHub Actions: preseason-check.yml) or locally. Idempotent.
# For the season just finished (src/season.py; SEASON=n overrides):
#   1. save its final live_v3 predictions to artifacts/ (fresh checkouts rebuild only the current season)
#   2. once nflverse publishes its participation (personnel) data, run the full model on it -> artifacts/, and the
#      site headline grows by a season
#   3. refresh the players table (new rookies' positions) for next season's features
# Exits early when there is nothing to do. NODEPLOY=1 stops before commit/push; ART=dir writes artifacts elsewhere.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONPATH="$PWD/tools/pyexit"
S=${SEASON:-$($PY src/season.py)}
ART=${ART:-artifacts}
REL=https://github.com/nflverse/nflverse-data/releases/download
step() { echo; echo "=== $(date '+%F %T') $*"; }

have_part=""
if curl -sfI -o /dev/null "$REL/pbp_participation/pbp_participation_$S.parquet"; then have_part=1; fi
need=""
[[ -f $ART/preds_v3/${S}_live_v3.parquet ]] || need="live"
[[ -n $have_part && ! -f $ART/preds_v3/${S}_full_v3.parquet ]] && need="$need full"
echo "season $S: participation published: ${have_part:-no}; to do: ${need:-nothing}"
[[ -z $need ]] && exit 0

step "inputs"
mkdir -p data/raw data/ref data/preds_v3
cp -n $ART/preds_v3/*.parquet data/preds_v3/ 2>/dev/null || true
[[ -f data/ablate_v3.json ]] || cp artifacts/ablate_v3.json data/
curl -sfL -o data/ref/players.parquet "$REL/players/players.parquet" || cp artifacts/players.parquet data/ref/
get() { [[ -f "data/raw/$2" ]] || curl -sfL -o "data/raw/$2" "$REL/$1" || rm -f "data/raw/$2"; }
for y in $(seq 2016 $((S - 1))); do
  get pbp/play_by_play_$y.parquet pbp_$y.parquet
  get pbp_participation/pbp_participation_$y.parquet part_$y.parquet
  if [[ $y -ge 2022 ]]; then get ftn_charting/ftn_charting_$y.parquet ftn_$y.parquet; fi
done
# the finished season: always the final published files
curl -sfL -o data/raw/pbp_$S.parquet $REL/pbp/play_by_play_$S.parquet
curl -sfL -o data/raw/ftn_$S.parquet $REL/ftn_charting/ftn_charting_$S.parquet
if [[ -n $have_part ]]; then curl -sfL -o data/raw/part_$S.parquet $REL/pbp_participation/pbp_participation_$S.parquet; fi

step "features"
$PY src/features.py data/raw data/features.parquet
$PY src/features_v2.py data/raw data/features.parquet data/features_v2.parquet
$PY src/features_v3.py data/raw data/features_v2.parquet data/features_v3.parquet

step "models for $S (trained on seasons < $S)"
$PY src/backtest_v3.py close "$S"
mkdir -p $ART/preds_v3
cp data/preds_v3/${S}_live_v3.parquet $ART/preds_v3/
[[ -f data/preds_v3/${S}_full_v3.parquet ]] && cp data/preds_v3/${S}_full_v3.parquet $ART/preds_v3/
cp data/ref/players.parquet $ART/players.parquet
$PY src/showcase_data.py >/dev/null
$PY - <<'PY'
import json
R = json.load(open("site/results.json"))
assert 0.7 < R["headline"]["acc"] < 0.95, R["headline"]
print("site data ok:", R["seasons"], R["headline"]["acc"])
PY

if [[ -n "${NODEPLOY:-}" ]]; then step "NODEPLOY set: skipping commit and push"; exit 0; fi
step "commit + push"
git add $ART site/results.json
git commit -q -m "Season close: $S ($need)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing to commit"
git pull -q --rebase origin main
git push origin main
step "done"
