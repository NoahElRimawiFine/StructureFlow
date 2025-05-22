#!/usr/bin/env python
# aggregate_rf_metrics.py ------------------------------------------------------
"""
Aggregate per‑seed GRN metrics for reference‑fitting (RF) experiments.

Folder layout expected
----------------------
results/
  dyn-BF/
    A_true.csv
    full/
      RF_full_1_matrix.csv
      RF_full_2_matrix.csv
      ...
    wt/
      RF_wt_1_matrix.csv
      ...
  dyn-TF/
    A_true.csv
    full/
    wt/
    ...

Run:
    python aggregate_rf_metrics.py --results-dir results
Produces:
    metrics_summary.csv   # one row = dataset × regime  (mean ± std)
"""

import argparse, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    auc as sk_auc,
)

# -----------------------------------------------------------------------------


def maskdiag(A: np.ndarray) -> np.ndarray:
    """Remove self‑loops from an adjacency matrix."""
    return A * (1 - np.eye(A.shape[0]))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (AUPR, ROC‑AUC, AP)."""
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    aupr   = sk_auc(rec, prec)                # area under PR curve
    ap     = average_precision_score(y_true, y_pred)
    aucroc = roc_auc_score(y_true, y_pred)
    return aupr, aucroc, ap


SEED_RX = re.compile(r"_([0-9]+)[^0-9]*")  # first integer chunk after "_"


def collect_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        # --------------------------------------------------- ground truth -----
        gt_fp = dataset_dir / "A_true.csv"
        if not gt_fp.exists():
            print(f"[warn] {gt_fp} missing – skip {dataset}", file=sys.stderr)
            continue
        y_true = np.abs(maskdiag(np.loadtxt(gt_fp, delimiter=","))).astype(int).flatten()

        # --------------------------------------------------- each regime ------
        for regime_dir in sorted(dataset_dir.iterdir()):
            if not regime_dir.is_dir() or regime_dir.name == "A_true.csv":
                continue
            regime = regime_dir.name        # e.g. "full", "wt"

            seed_metrics = []

            for fp in regime_dir.glob("RF_*_matrix.csv"):
                # Expected stem: RF_<regime>_<seed>_matrix
                seed_m = SEED_RX.search(fp.stem)
                seed_id = int(seed_m.group(1)) if seed_m else -1

                A_pred = np.loadtxt(fp, delimiter=",")
                y_pred = np.abs(maskdiag(A_pred)).flatten()

                aupr, aucroc, ap = compute_metrics(y_true, y_pred)
                seed_metrics.append((seed_id, aupr, aucroc, ap))

            if not seed_metrics:
                continue

            # ----------------------- aggregate across seeds ------------------
            _, auprs, aucrocs, aps = zip(*seed_metrics)

            rows.append({
                "dataset": dataset,
                "regime":  regime,
                "n_seeds": len(auprs),
                "AUPR_mean":  np.mean(auprs),
                "AUPR_std":   np.std(auprs, ddof=1),
                "ROC_AUC_mean":  np.mean(aucrocs),
                "ROC_AUC_std":   np.std(aucrocs, ddof=1),
                "AP_mean":    np.mean(aps),
                "AP_std":     np.std(aps, ddof=1),
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results-curated"),
                        help="Root folder with dataset/regime/RF_*_matrix.csv")
    args = parser.parse_args()

    df = collect_metrics(args.results_dir)

    if df.empty:
        print("No metrics collected – check folder structure.", file=sys.stderr)
        sys.exit(1)

    out_path = Path("metrics_summary_rf1.csv")
    df.to_csv(out_path, index=False)
    print("\n=== RF metrics (mean ± std across seeds) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
