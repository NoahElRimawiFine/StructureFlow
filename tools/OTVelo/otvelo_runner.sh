#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"                      
SCRIPT="${SCRIPT:-otvelo_baseline.py}"
DATA_ROOT="${DATA_ROOT:-../../data/Synthetic}"       
HSC_PATH="${HSC_PATH:-../../data/Curated/HSC}"        
SEEDS=(1 2 3 4 5)                        
T_BINS=5                                         

backbones=(dyn-BF dyn-TF dyn-SW dyn-CY)

echo
echo "Running OTVelo baseline"
echo "• script     : $SCRIPT"
echo "• data root  : $DATA_ROOT"
echo "• seeds      : ${SEEDS[*]}"
echo "• python     : $PYTHON"
echo

for ds in "${backbones[@]}"; do
  DIR="${DATA_ROOT}/${ds}"
  echo -e "\n▸ DATASET: ${ds}"

  if [[ ! -d "$DIR" ]]; then
      echo "  [WARN] path not found -> $DIR (skipping)"; continue
  fi

  for seed in "${SEEDS[@]}"; do
    echo "  • WT only (seed ${seed})"
    $PYTHON "$SCRIPT" --backbone "${ds}"               \
            --subset wt       \
            --seed "$seed"

    echo "  • WT + KO pooled (seed ${seed})"
    $PYTHON "$SCRIPT" --backbone "${ds}"               \
            --subset all       \
            --seed "$seed"
  done
done

if [[ -d "$HSC_PATH" ]]; then
  echo -e "\n▸ DATASET: HSC (Curated)"

  for seed in "${SEEDS[@]}"; do
    echo "  • WT only (seed ${seed})"
    $PYTHON "$SCRIPT" --backbone "HSC"          \
            --subset wt       \
            --seed "$seed"

    echo "  • WT + KO pooled (seed ${seed})"
    $PYTHON "$SCRIPT" --backbone "HSC"          \
            --subset all     \
            --seed "$seed"
  done
else
  echo -e "\n[WARN] Curated HSC path not found ($HSC_PATH) — skipping"
fi

echo -e "\n✓ All datasets processed."
