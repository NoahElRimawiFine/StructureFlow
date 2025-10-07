#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"                      
SCRIPT="${SCRIPT:-otvelo_baseline.py}"
DATA_ROOT="${DATA_ROOT:-../../data/Synthetic}"       
HSC_PATH="${HSC_PATH:-../../data/Curated}"        
SEEDS=(1 2 3)                        
T_BINS=5                                         

backbones=(dyn-SW dyn-BF dyn-CY dyn-TF)

echo
echo "Running OTVelo baseline"
echo "• script     : $SCRIPT"
echo "• data root  : $DATA_ROOT"
echo "• seeds      : ${SEEDS[*]}"
echo "• python     : $PYTHON"
echo

FAILS=()

for ds in "${backbones[@]}"; do
  DIR="${DATA_ROOT}/${ds}"
  echo -e "\n▸ DATASET: ${ds}"

  if [[ ! -d "$DIR" ]]; then
      echo "  [WARN] path not found -> $DIR (skipping)"; continue
  fi

  for seed in "${SEEDS[@]}"; do
    echo "  • WT + KO pooled (seed ${seed})"
    if ! $PYTHON -u "$SCRIPT" --backbone "${ds}" \
            --subset all       \
            --seed "$seed"; then
        echo "[ERROR] ${ds} seed ${seed} failed"
        FAILS+=("${ds}_seed${seed}")
    fi
  done
done

if [[ -d "$HSC_PATH/HSC" ]]; then
  echo -e "\n▸ DATASET: HSC (Curated)"

  for seed in "${SEEDS[@]}"; do
    echo "  • WT + KO pooled (seed ${seed})"
    if ! $PYTHON "$SCRIPT" --backbone "HSC"          \
            --subset all     \
            --seed "$seed"; then
        echo "[ERROR] HSC seed ${seed} failed"
        FAILS+=("HSC_seed${seed}")
    fi
  done
else
  echo -e "\n[WARN] Curated HSC path not found ($HSC_PATH/HSC) — skipping"
fi

# Aggregate results across seeds
echo -e "\n▸ Aggregating results across seeds..."
if command -v python3 &> /dev/null; then
    python3 aggregate_seeds.py
else
    python aggregate_seeds.py
fi

if [[ ${#FAILS[@]} -ne 0 ]]; then
    echo
    echo "Summary: Failed runs:"
    for F in "${FAILS[@]}"; do
        echo " - $F"
    done
    exit 1
else
    echo -e "\n✓ All datasets processed successfully."
fi