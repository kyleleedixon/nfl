#!/usr/bin/env bash
# Rebuild everything from public nflverse data. Run from the project root.
set -euo pipefail
PY=${PY:-.venv/bin/python}
mkdir -p data/raw data/ref site

echo "1/6 Downloading play-by-play and participation data"
REL=https://github.com/nflverse/nflverse-data/releases/download
for y in $(seq 2016 2026); do
  curl -sfL -o data/raw/pbp_$y.parquet $REL/pbp/play_by_play_$y.parquet
  curl -sfL -o data/raw/part_$y.parquet $REL/pbp_participation/pbp_participation_$y.parquet || rm -f data/raw/part_$y.parquet
done
curl -sfL -o data/ref/players.parquet $REL/players/players.parquet

echo "2/6 Base features"
$PY src/features.py data/raw data/features.parquet

echo "3/6 v2 features (lineups, substitutions, personnel x formation tendencies)"
$PY src/features_v2.py data/raw data/features.parquet data/features_v2.parquet

if [[ "${TUNE:-0}" == "1" ]]; then
  echo "   tuning (TUNE=1)"; $PY src/backtest_v2.py tune
fi

echo "4/6 Walk-forward backtest 2019-2026 (slowest step)"
$PY src/backtest_v2.py run

echo "5/6 Showcase numbers"
$PY src/showcase_data.py

echo "6/6 Hero drive"
$PY src/pick_drive.py
echo "Done. Preview with: cd site && python3 -m http.server 8000"
