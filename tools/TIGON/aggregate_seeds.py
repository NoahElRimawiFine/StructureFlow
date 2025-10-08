#!/usr/bin/env python3
"""
Aggregate TIGON trajectory inference results across multiple seeds.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob


def aggregate_results(results_dir="tigon_results"):
    """
    Aggregate results across seeds for each backbone+subset combination.

    Expected directory structure:
        tigon_results/
            {backbone}_{subset}_seed{seed}/
                trajectory_inference_results.csv
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[WARN] Results directory not found: {results_path}")
        return

    # Find all result CSV files
    csv_files = list(results_path.glob("*/trajectory_inference_results.csv"))

    if not csv_files:
        print(
            f"[WARN] No trajectory_inference_results.csv files found in {results_path}"
        )
        return

    print(f"Found {len(csv_files)} result files")

    # Parse directory names to extract backbone, subset, seed
    results = []
    for csv_file in csv_files:
        dir_name = csv_file.parent.name
        parts = dir_name.rsplit("_seed", 1)

        if len(parts) != 2:
            print(f"[WARN] Unexpected directory name format: {dir_name}")
            continue

        base_name, seed_str = parts
        try:
            seed = int(seed_str)
        except ValueError:
            print(f"[WARN] Could not parse seed from: {seed_str}")
            continue

        # Parse base_name to get backbone and subset
        # Format: {backbone}_{subset}
        if "_wt_" in base_name or base_name.endswith("_wt"):
            backbone = base_name.replace("_wt", "")
            subset = "wt"
        elif "_ko_" in base_name or base_name.endswith("_ko"):
            backbone = base_name.replace("_ko", "")
            subset = "ko"
        elif "_all_" in base_name or base_name.endswith("_all"):
            backbone = base_name.replace("_all", "")
            subset = "all"
        else:
            # Default: assume last part is subset
            parts = base_name.rsplit("_", 1)
            if len(parts) == 2:
                backbone, subset = parts
            else:
                backbone = base_name
                subset = "unknown"

        # Read the CSV
        try:
            df = pd.read_csv(csv_file)
            df["backbone"] = backbone
            df["subset"] = subset
            df["seed"] = seed
            results.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_file}: {e}")
            continue

    if not results:
        print("[WARN] No valid results to aggregate")
        return

    # Combine all results
    all_results = pd.concat(results, ignore_index=True)

    # Group by backbone, subset, and t_star to compute statistics
    grouped = all_results.groupby(["backbone", "subset", "t_star"])

    aggregated = grouped.agg(
        {
            "WD2": ["mean", "std", "count"],
            "MMD": ["mean", "std"],
            "src": "first",  # src should be the same for all seeds
        }
    ).reset_index()

    # Flatten multi-level column names
    aggregated.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in aggregated.columns.values
    ]

    # Save aggregated results
    output_file = results_path / "aggregated_trajectory_results.csv"
    aggregated.to_csv(output_file, index=False)
    print(f"\n✓ Aggregated results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary of aggregated results:")
    print("=" * 80)
    for (backbone, subset), group in aggregated.groupby(["backbone", "subset"]):
        print(f"\n{backbone} ({subset}):")
        print(
            f"  Mean WD2: {group['WD2_mean'].mean():.4f} ± {group['WD2_std'].mean():.4f}"
        )
        print(
            f"  Mean MMD: {group['MMD_mean'].mean():.4f} ± {group['MMD_std'].mean():.4f}"
        )
        print(f"  Seeds: {int(group['WD2_count'].iloc[0])}")

    return aggregated


if __name__ == "__main__":
    aggregate_results()
