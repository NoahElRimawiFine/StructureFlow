#!/usr/bin/env python
# aggregate_metrics.py ---------------------------------------------------------
"""
Collect adjacency matrices produced by different seeds and compute
mean ± std AP / AUC per dataset × regime × method.

Specifically handles files like:
results/{dataset}/{regime}/RF_{regime}_{seed}_matrix.csv

Usage
-----
python aggregate_metrics.py --results-dir results

The script produces metrics_summary.csv next to itself.
"""
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score)

# -----------------------------------------------------------------------------


def maskdiag(A: np.ndarray) -> np.ndarray:
    """Zero the diagonal – we never score self‑edges."""
    return A * (1 - np.eye(A.shape[0]))


def compute_scores(y_true, A_pred):
    """Return AP, AUC for a single adjacency matrix."""
    y_score = np.abs(maskdiag(A_pred)).flatten()
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return ap, auc


# ──────────────────────────────────────────────────────────────────────────────
SEED_RX = re.compile(r"_([0-9]+)[^0-9]*")  # captures the first integer chunk


def collect_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        print(dataset_name)

        for regime_dir in sorted(dataset_dir.iterdir()):
            if not regime_dir.is_dir():
                continue
            regime = regime_dir.name  # e.g. "full" or "wt"
            print(regime)

            # ground truth: expect at top dataset level
            true_fp = dataset_dir / "A_true.csv"
            if not true_fp.exists():
                print(f"[warn] A_true.csv missing for {dataset_name}", file=sys.stderr)
                continue
            y_true = np.abs(maskdiag(np.loadtxt(true_fp, delimiter=","))).astype(int).flatten()

            # find all *_matrix.csv files
            matrix_files = list(regime_dir.glob("*_matrix.csv"))
            if not matrix_files:
                continue

            for fp in matrix_files:
                name_parts = fp.stem.split("_")  # e.g. ["RF", "full", "0", "matrix"]
                print(name_parts)
                if len(name_parts) < 3:
                    continue

                method = name_parts[0]
                seed_m = SEED_RX.search(fp.stem)
                seed_id = int(seed_m.group(1)) if seed_m else -1

                A_pred = np.loadtxt(fp, delimiter=",")
                ap, auc = compute_scores(y_true, A_pred)

                rows.append({
                    "dataset": dataset_name,
                    "regime":  regime,
                    "method":  method,
                    "seed":    seed_id,
                    "AP":      ap,
                    "AUC":     auc,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # aggregate mean ± std across seeds
    summary = (df.groupby(["dataset", "regime", "method"])
                 .agg(AP_mean=("AP", "mean"),
                      AP_std=("AP", "std"),
                      AUC_mean=("AUC", "mean"),
                      AUC_std=("AUC", "std"),
                      n_seeds=("seed", "count"))
                 .reset_index())

    return summary


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="root folder with dataset/regime/matrix.csv files")
    args = parser.parse_args()

    df = collect_metrics(args.results_dir)

    if df.empty:
        print("No metrics collected – check paths / filenames.", file=sys.stderr)
        sys.exit(1)

    out_path = Path("metrics_summary-rf.csv")
    df.to_csv(out_path, index=False)
    print("\n=== averaged metrics ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
