#!/bin/bash
set -euo pipefail

# ----- unique run tag & dirs -----
RUN_TAG="${RUN_TAG:-traj-$(date +%Y%m%d-%H%M%S)}"
ROOT_DIR="runs/${RUN_TAG}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
echo "Run tag: $RUN_TAG"
echo "Logs:    $LOG_DIR"

# ----- config -----
BACKBONES=(dyn-BF dyn-TF dyn-SW dyn-CY dyn-LL)
SUBSET="all"

NITERS=100
TIMEPOINTS="0,0.25,0.5,0.75,1.0"
HIDDENDIM=16
SEEDS=(1 2 3)

declare -A FAILS

# file to capture PIDs of background jobs for THIS run
PIDFILE="${ROOT_DIR}/pids.txt"
: > "$PIDFILE"

for BACKBONE in "${BACKBONES[@]}"; do
  echo "=== RUN $BACKBONE subset=$SUBSET ==="
  for SEED in "${SEEDS[@]}"; do
    echo "== SEED $SEED =="
    log="${LOG_DIR}/${BACKBONE}_seed${SEED}.log"
    CMD=(python3 -u tigon_baseline.py
      --niters "$NITERS"
      --timepoints "$TIMEPOINTS"
      --hidden-dim "$HIDDENDIM"
      --backbone "$BACKBONE"
      --subset "$SUBSET"
      --seed "$SEED"
    )
    echo "running: ${CMD[*]}  (log: $log)"
    (
      "${CMD[@]}" >"$log" 2>&1
      STATUS=$?
      if [[ $STATUS -ne 0 ]]; then
        echo "[ERROR] $BACKBONE (seed $SEED) failed with exit code $STATUS. See $log"
        FAILS["$BACKBONE|$SEED"]=1
      fi
    ) &
    echo $! >> "$PIDFILE"
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

echo "Background PIDs for this run are in: $PIDFILE"
