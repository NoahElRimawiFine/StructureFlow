#!/usr/bin/env bash
set -euo pipefail

# 1) Define the exact order of your 7 “columns” (true plus 6 methods):
methods=( \
  A_true \
  dynGENIE3 \
  Glasso \
  SCODE \
  Sincerities \
  RF \
  StructureFlow \
)

# 2) Build the list of files for the *full‐regime* row in that order:
row_full=()
for m in "${methods[@]}"; do
  if [[ "$m" == "A_true" ]]; then
    row_full+=( "heatmaps/SW/A_true_SW.pdf" )
  else
    row_full+=( "heatmaps/SW/${m}_FULL_SW.pdf" )
  fi
done

# 3) (Optional) append your global colorbar/legend PDF at the end:
row_full+=( "heatmaps/global_colorbar-crop.pdf" )

# 4) Call pdfjam for a single‐row, eight‐column layout:
pdfjam \
  --nup       8x1 \
  --paper     a4paper \
  --landscape \
  --frame     false \
  --delta     "1mm 1mm" \
  --scale     0.95 \
  --outfile   SW_full_row_with_legend.pdf \
  "${row_full[@]}"
