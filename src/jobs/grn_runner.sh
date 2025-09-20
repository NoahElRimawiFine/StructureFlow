#!/usr/bin/env bash
# grn_runner.sh – run run_less_simple.py once per dataset
#                 (wild-type   ➜ default flags)
#                 (full data  ➜ --concat_all)
# -----------------------------------------------------------------
set -euo pipefail

# ─── Edit just this array to add/remove datasets ─────────────────
datasets=(dyn-TF dyn-CY dyn-BF dyn-LL dyn-SW)
seeds=(1 2 3 4 5)

ROOT="../../data/Synthetic"        # common parent
SCRIPT="run_less_simple.py"        # python entry-point
PYTHON="python3"
T=5                                # pseudo-time bins

for ds in "${datasets[@]}"; do
  DIR="${ROOT}/${ds}"
  echo -e "\n▸ DATASET: ${DIR}"

  # Automatically add --curated if "Curated" appears in the path
  CURATED=""
  [[ "${DIR}" =~ /Curated/ ]] && CURATED="--curated"

  for seed in "${seeds[@]}"; do
    echo "  • wildtype (seed ${seed})"
    ${PYTHON} ${SCRIPT} "${DIR}" --T "${T}" ${CURATED} --seed "${seed}"

    echo "  • full (seed ${seed})"
    ${PYTHON} ${SCRIPT} "${DIR}" --T "${T}" --concat_all ${CURATED} --seed "${seed}"
  done
done

echo -e "\n✓ All datasets processed."

