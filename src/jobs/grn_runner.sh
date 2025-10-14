#!/usr/bin/env bash

set -euo pipefail
datasets=(HSC)
seeds=(1 2 3 4 5)

ROOT="../../data/Curated"       
SCRIPT="run_models.py"       
PYTHON="python3"
T=5                                

for ds in "${datasets[@]}"; do
  DIR="${ROOT}/${ds}"
  echo -e "\n▸ DATASET: ${DIR}"

  CURATED=""
  [[ "${DIR}" =~ /Curated/ ]] && CURATED="--curated"

  for seed in "${seeds[@]}"; do
    echo "  • wildtype (seed ${seed})"
    ${PYTHON} ${SCRIPT} "${DIR}" --T "${T}" ${CURATED} --seed "${seed}"

    echo "  • full (seed ${seed})"
    ${PYTHON} ${SCRIPT} "${DIR}" --T "${T}" --concat_all ${CURATED} --seed "${seed}"
  done
done

echo -e "\n All datasets processed."

