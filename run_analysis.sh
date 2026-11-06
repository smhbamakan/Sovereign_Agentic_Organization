#!/usr/bin/env bash
# Reproduce every table and figure of the manuscript from the decision ledger in data/results.zip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="${SAO_DATA:-$ROOT/data}"
[ -d "$DATA/results" ] || unzip -q "$DATA/results.zip" -d "$DATA"
A="$ROOT/analysis"
mkdir -p "$ROOT/outputs/main" "$ROOT/outputs/followup" "$ROOT/outputs/model" "$ROOT/figures"

echo "[1/4] main study (Case A, frontier replication, Case B)"
cd "$ROOT/outputs/main"
SAO_STUDIES=mainA,frontA,mainB python "$A/build_data.py" "$DATA"        # windows.csv, episodes.csv, items.pkl
python "$A/analyze.py"                   # results_main.json  (mixed models, arrangements, H6, H7)
python "$A/coupling.py" "$DATA"          # coupling.json      (planned gap specification)
python "$A/coupling2.py"                 # coupling2.json     (unrestricted specification, Table 7)
python "$A/spec_an.py"                   # spec_an.json       (specification comparison)
python "$A/recovery.py"                  # recovery.json      (coupling recovery study)
python "$A/probes_an.py" "$DATA"         # probes.json        (error correlations)
python "$A/probes_retest.py" "$DATA"     # probes_retest.json (within-model retest)
python "$A/review_an.py"                 # review_an.json     (ratifier GEE/AUC, beta(tau), numeric floor, majority vote)

echo "[2/4] follow-up experiments (S1-S4, persona probes)"
cd "$ROOT/outputs/followup"
python "$A/build_data.py" "$DATA"
python "$A/supp_an.py"                   # supp.json
python "$A/perm.py"                      # perm.json (within-episode permutation tests)

echo "[3/4] figures"
python "$A/figs_pub.py"                  # Figs. 1-6
python "$A/fig7.py"                      # Fig. 7

echo "[4/4] Monte Carlo verification of the propositions"
cd "$ROOT/outputs/model"
python "$ROOT/model/verify.py" && python "$ROOT/model/verify2.py"
echo "done: outputs/ and figures/"
