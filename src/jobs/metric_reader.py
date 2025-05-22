import pandas as pd
import glob
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="results-curated",
                    help="root directory holding dataset folders")
args = parser.parse_args()

results_dir = pathlib.Path(args.results_dir)
all_summaries = []

for dataset_dir in sorted(results_dir.glob("HSC")):
    dataset_name = dataset_dir.name
    print(f"Processing → {dataset_dir}")

    combined_dfs = []

    # patterns for FULL
    full_patterns = [
        f"{dataset_dir}/full/metrics_seed*.csv",
        f"{dataset_dir}/metrics_seed*.csv",
    ]
    full_files = []
    for pat in full_patterns:
        found = glob.glob(pat)
        if found:
            full_files.extend(found)
    print(" full files:", full_files)

    for fp in full_files:
        df = pd.read_csv(fp)
        if "AUC_ROC" in df.columns:
            df = df.rename(columns={"AUC_ROC": "AUC"})
        combined_dfs.append(df)

    # patterns for WT
    wt_patterns = [
        f"{dataset_dir}/wt/metrics_seed*.csv",
        f"{dataset_dir}/metrics_seed*.csv",
    ]
    wt_files = []
    for pat in wt_patterns:
        found = glob.glob(pat)
        if found:
            wt_files.extend(found)
    print(" wt  files:", wt_files)

    for fp in wt_files:
        df = pd.read_csv(fp)
        if "AUC_ROC" in df.columns:
            df = df.rename(columns={"AUC_ROC": "AUC"})
        combined_dfs.append(df)

    if not combined_dfs:
        print(f"  [warn] no metric CSVs found in {dataset_dir}")
        continue

    combined = pd.concat(combined_dfs, ignore_index=True)

    summary = (
        combined
        .groupby(["dataset", "regime", "method"])
        .agg(
            n_seeds   = ("seed", "count"),
            AP_mean   = ("AP", "mean"),
            AP_std    = ("AP", "std"),
            AUPR_mean = ("AUPR", "mean"),
            AUPR_std  = ("AUPR", "std"),
            AUC_mean  = ("AUC", "mean"),
            AUC_std   = ("AUC", "std"),
        )
        .reset_index()
    )

    out_path = dataset_dir / "metrics_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f" Saved summary → {out_path}")
    all_summaries.append(summary)

# combine across all datasets
if all_summaries:
    final_summary = pd.concat(all_summaries, ignore_index=True)
    dst = results_dir / "all_metrics_summary.csv"
    final_summary.to_csv(dst, index=False)
    print(f"\nSaved combined summary → {dst}")
else:
    print("No summaries generated!")
