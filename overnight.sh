#!/usr/bin/env bash
# Unattended v3 pipeline: features -> ablation -> tuning -> walk-forward -> site -> deploy -> commit/push.
# Every step is resumable; re-running skips finished work. Log: data/overnight.log
# Start detached: nohup ./overnight.sh >/dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
exec >>data/overnight.log 2>&1
caffeinate -dimsu -w $$ &
rm -f data/overnight.done data/overnight.failed
trap 'echo "=== $(date "+%F %T") FAILED (line $LINENO)"; touch data/overnight.failed' ERR
step() { echo; echo "=== $(date '+%F %T') $*"; }

if [[ ! -f data/features_v3.parquet ]]; then
  step "v3 features"; $PY src/features_v3.py data/raw data/features_v2.parquet data/features_v3.parquet
fi
if ! $PY -c "import json,sys; sys.exit(0 if 'catboost' in json.load(open('data/ablate_v3.json'))['log'] else 1)" 2>/dev/null; then
  step "ablation (2023)"; $PY src/backtest_v3.py ablate
fi
if [[ ! -f src/best_params_v3.json ]]; then
  step "tuning (train <=2021, early-stop 2022)"; $PY src/backtest_v3.py tune 20
fi
step "walk-forward"; $PY src/backtest_v3.py run

step "holdout comparison"
$PY - <<'EOF' | tee data/holdout_summary.txt
import pandas as pd, numpy as np
from sklearn.metrics import log_loss
P = pd.read_parquet("data/preds_v3.parquet")
def row(g, c):
    g = g[g[c].notna()]
    return f"{c:8s} ll {log_loss(g.y, np.clip(g[c], 1e-6, 1 - 1e-6)):.4f}  acc {((g[c] > .5) == g.y).mean():.4f}  n {len(g)}"
for S in (2024, 2025, 2026):
    g = P[P.season == S]
    print(S); [print("  " + row(g, c)) for c in ["full_v3", "full_v2", "live_v3", "live_v2", "xpass"] if g[c].notna().any()]
EOF

step "site data"
$PY src/showcase_data.py
$PY src/pick_drive.py
$PY - <<'EOF'
import json
R = json.load(open("site/results.json")); json.load(open("site/drive.json"))
a = R["headline"]["acc"]
assert 0.7 < a < 0.95 and R["n_plays"] > 200000 and R["live_2026"]["plays"] > 0, (a, R["n_plays"])
print("site data ok: acc", a, "plays", R["n_plays"])
EOF

step "deploy"
(cd site && vercel deploy --prod --yes)

step "commit + push"
git add -A src site/index.html site/results.json site/drive.json run_all.sh overnight.sh requirements.txt CLAUDE.md .gitignore
git commit -q -m "v3: coach decision-bias and pre-snap alignment features, tuned ensemble, published results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing to commit"
git push origin main

touch data/overnight.done
step "done"
