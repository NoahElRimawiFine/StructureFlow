#!/usr/bin/env bash

for dataset in results2/*; do
    if [ -d "$dataset" ]; then
        echo "$dataset"
        python3 metric_reader.py --dataset "$dataset"
    fi
done
