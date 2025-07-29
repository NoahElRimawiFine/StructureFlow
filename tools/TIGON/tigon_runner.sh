#!/bin/bash

BACKBONES=("dyn-BF" "dyn-TF" "dyn-SW" "dyn-CY" "dyn-LL")
SUBSET="all"

NITERS=100
TIMEPOINTS="0,0.25,0.5,0.75,1.0"
HIDDENDIM=16

FAILS=()

for BACKBONE in "${BACKBONES[@]}"; do
    echo "=== RUN $BACKBONE subset=$SUBSET ==="
    CMD=(python tigon_baseline.py \
        --niters "$NITERS" \
        --timepoints "$TIMEPOINTS" \
        --hidden-dim "$HIDDENDIM" \
        --backbone "$BACKBONE" \
        --subset "$SUBSET")

    echo "running: ${CMD[*]}"
    "${CMD[@]}"
    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
        echo "[ERROR] $BACKBONE failed with exit code $STATUS"
        FAILS+=("$BACKBONE")
    fi
done

if [[ ${#FAILS[@]} -ne 0 ]]; then
    echo
    echo "Summary: Failed runs:"
    for F in "${FAILS[@]}"; do
        echo " - $F"
    done
else
    echo "All runs finished successfully."
fi
