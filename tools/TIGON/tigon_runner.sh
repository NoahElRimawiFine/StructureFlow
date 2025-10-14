#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-tigon_baseline.py}"
DATA_ROOT="${DATA_ROOT:-../../data/Synthetic}"
HSC_PATH="${HSC_PATH:-../../data/Curated}"
SEEDS=(1 2 3)

BACKBONES=("dyn-BF" "dyn-TF" "dyn-SW" "dyn-CY" "dyn-LL")
SUBSET="all"

NITERS=100
TIMEPOINTS="0,0.25,0.5,0.75,1.0"
HIDDENDIM=16

echo
echo "Running TIGON baseline"
echo "• script     : $SCRIPT"
echo "• data root  : $DATA_ROOT"
echo "• seeds      : ${SEEDS[*]}"
echo "• python     : $PYTHON"
echo

FAILS=()

for BACKBONE in "${BACKBONES[@]}"; do
    DIR="${DATA_ROOT}/${BACKBONE}"
    echo -e "\n DATASET: ${BACKBONE}"

    if [[ ! -d "$DIR" ]]; then
        echo "  [WARN] path not found -> $DIR (skipping)"; continue
    fi

    for seed in "${SEEDS[@]}"; do
        echo "  • WT + KO pooled (seed ${seed})"
        CMD=($PYTHON -u "$SCRIPT" \
            --niters "$NITERS" \
            --timepoints "$TIMEPOINTS" \
            --hidden-dim "$HIDDENDIM" \
            --backbone "$BACKBONE" \
            --subset "$SUBSET" \
            --seed "$seed")

        echo "    running: ${CMD[*]}"
        if ! "${CMD[@]}"; then
            echo "[ERROR] $BACKBONE seed $seed failed"
            FAILS+=("${BACKBONE}_seed${seed}")
        fi
    done
done

# Run HSC if available
if [[ -d "$HSC_PATH/HSC" ]]; then
    echo -e "\n▸ DATASET: HSC (Curated)"
    
    for seed in "${SEEDS[@]}"; do
        echo "  • WT + KO pooled (seed ${seed})"
        CMD=($PYTHON -u "$SCRIPT" \
            --niters "$NITERS" \
            --timepoints "$TIMEPOINTS" \
            --hidden-dim "$HIDDENDIM" \
            --backbone "HSC" \
            --subset "all" \
            --seed "$seed")

        echo "    running: ${CMD[*]}"
        if ! "${CMD[@]}"; then
            echo "[ERROR] HSC seed $seed failed"
            FAILS+=("HSC_seed${seed}")
        fi
    done
else
    echo -e "\n[WARN] Curated HSC path not found ($HSC_PATH/HSC) — skipping"
fi

# Aggregate results across seeds
echo -e "\n Aggregating results across seeds..."
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
    echo -e "\n All datasets processed successfully."
fi