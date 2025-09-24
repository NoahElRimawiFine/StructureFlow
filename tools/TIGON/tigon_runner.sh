#!/bin/bash

BACKBONES=(dyn-BF dyn-TF dyn-SW dyn-CY dyn-LL)
SUBSET="all"

NITERS=100
TIMEPOINTS="0,0.25,0.5,0.75,1.0"
HIDDENDIM=16
SEEDS=(1 2 3)

declare -A FAILS
mkdir -p logs

for BACKBONE in "${BACKBONES[@]}"; do
  echo "=== RUN $BACKBONE subset=$SUBSET ==="
  for SEED in "${SEEDS[@]}"; do
    echo "== SEED $SEED =="
    log="logs/${BACKBONE}_seed${SEED}.log"
    CMD=(python tigon_baseline.py
      --niters "$NITERS"
      --timepoints "$TIMEPOINTS"
      --hidden-dim "$HIDDENDIM"
      --backbone "$BACKBONE"
      --subset "$SUBSET"
      --seed "$SEED"
    )
    echo "running: ${CMD[*]}"
    # run in background
    (
      "${CMD[@]}" >"$log" 2>&1
      STATUS=$?
      if [[ $STATUS -ne 0 ]]; then
        echo "[ERROR] $BACKBONE (seed $SEED) failed with exit code $STATUS. See $log"
        FAILS["$BACKBONE|$SEED"]=1
      fi
    ) &
  done

  # wait for all seeds of this backbone to finish
  wait
done

if (( ${#FAILS[@]} > 0 )); then
  echo
  echo "Summary: Failed runs:"
  for key in "${!FAILS[@]}"; do
    IFS='|' read -r b s <<<"$key"
    echo " - $b (seed $s)"
  done
else
  echo "All runs finished successfully."
fi