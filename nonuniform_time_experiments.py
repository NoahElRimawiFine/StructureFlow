#!/usr/bin/env python3
"""
Runner for non-uniform time spacing experiments.

Runs SF2M and RF on the dyn-TF system with multiple seeds for:
1. GRN inference (using train.py with full data) - for AP/AUROC metrics
2. Leave-one-out trajectory inference - for W2/MMD/ED metrics

Uses non-uniform time bins with time_jitter=0.3.
Aggregates all results into a single CSV file with std calculated across seeds.
"""

import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import glob

RESULTS_BASE_DIR = "nonuniform_time_results"
DATASET = "dyn-TF"
DATASET_TYPE = "Synthetic"
SEEDS = [1]  # TODO: Change to [1, 2, 3] for full experiment
TIME_JITTER = 0.3
MODEL_TYPES = ["sf2m", "rf"]

N_STEPS = 15000
BATCH_SIZE = 64
LR = 3e-3
ALPHA = 0.1
REG = 5e-6
GL_REG = 0.04


def run_grn_inference_sf2m(seed, results_dir):
    """Run GRN inference using train.py for SF2M."""
    cmd = [
        "python",
        "src/train.py",
        "data=boolode",
        "model=sf2m",
        f"seed={seed}",
        f"data.dataset={DATASET}",
        f"data.dataset_type={DATASET_TYPE}",
        "+data.use_nonuniform_time=true",
        f"+data.time_jitter={TIME_JITTER}",
        f"+data.time_seed={seed}",
        f"model.n_steps={N_STEPS}",
        f"model.batch_size={BATCH_SIZE}",
        f"model.lr={LR}",
        f"model.alpha={ALPHA}",
        f"model.reg={REG}",
        f"model.GL_reg={GL_REG}",
        "trainer.max_steps=" + str(N_STEPS),
        "trainer.max_epochs=-1",
        f"paths.output_dir={results_dir}/sf2m_seed{seed}",
        "logger=csv",
        "callbacks=none",
    ]

    print(f"\n{'='*60}")
    print(f"Running GRN inference (train.py): sf2m, seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_grn_inference_rf(seed, results_dir):
    """Run GRN inference using train.py for RF."""
    cmd = [
        "python",
        "src/train.py",
        "data=boolode",
        "model=rf",
        f"seed={seed}",
        f"data.dataset={DATASET}",
        f"data.dataset_type={DATASET_TYPE}",
        "+data.use_nonuniform_time=true",
        f"+data.time_jitter={TIME_JITTER}",
        f"+data.time_seed={seed}",
        f"paths.output_dir={results_dir}/rf_seed{seed}",
        "logger=csv",
        "callbacks=none",
        "trainer.max_epochs=1",
    ]

    print(f"\n{'='*60}")
    print(f"Running GRN inference (train.py): rf, seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_leave_one_out(model_type, seed, results_dir):
    """Run leave-one-out trajectory inference experiment."""
    cmd = [
        "python",
        "-m",
        "src.leave_one_out",
        "--model_type",
        model_type,
        "--dataset_type",
        DATASET_TYPE,
        "--dataset",
        DATASET,
        "--seed",
        str(seed),
        "--results_dir",
        results_dir,
        "--use_nonuniform_time",
        "--time_jitter",
        str(TIME_JITTER),
        "--time_seed",
        str(seed),
        "--n_steps_per_fold",
        str(N_STEPS),
        "--batch_size",
        str(BATCH_SIZE),
        "--lr",
        str(LR),
        "--alpha",
        str(ALPHA),
        "--reg",
        str(REG),
        "--gl_reg",
        str(GL_REG),
    ]

    print(f"\n{'='*60}")
    print(f"Running trajectory inference: {model_type}, seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def extract_grn_metrics_from_logs(results_dir):
    """Extract GRN metrics from train.py CSV logs."""
    all_results = []

    for model_type in MODEL_TYPES:
        for seed in SEEDS:
            log_dir = os.path.join(results_dir, f"{model_type}_seed{seed}", "csv")

            metrics_files = glob.glob(
                os.path.join(log_dir, "**", "metrics.csv"), recursive=True
            )

            if metrics_files:
                df = pd.read_csv(metrics_files[0])

                last_row = df.iloc[-1]

                result = {
                    "model_type": model_type,
                    "seed": seed,
                    "jacobian_ap": last_row.get("grn/jacobian_ap", np.nan),
                    "jacobian_auroc": last_row.get("grn/jacobian_auroc", np.nan),
                    "graph_ap": last_row.get("grn/graph_ap", np.nan),
                    "graph_auroc": last_row.get("grn/graph_auroc", np.nan),
                }
                all_results.append(result)
            else:
                print(f"Warning: No metrics.csv found for {model_type} seed {seed}")

    if all_results:
        return pd.DataFrame(all_results)
    return None


def aggregate_trajectory_results(results_dir):
    """Aggregate trajectory inference results from all seeds."""
    all_results = []

    for model_type in MODEL_TYPES:
        for seed in SEEDS:
            folder_name = f"{DATASET_TYPE}_{model_type}__{DATASET}_seed{seed}"
            summary_file = os.path.join(
                results_dir, folder_name, f"loo_summary_{model_type}_seed{seed}.csv"
            )

            if os.path.exists(summary_file):
                df = pd.read_csv(summary_file)
                df["seed"] = seed
                all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return None


def compute_final_statistics(grn_df, traj_df):
    """Compute final statistics with std across seeds (not timepoints)."""
    results = []

    for model_type in MODEL_TYPES:
        row = {"model_type": model_type, "n_seeds": len(SEEDS)}

        # GRN metrics - aggregate across seeds
        if grn_df is not None:
            model_grn = grn_df[grn_df["model_type"] == model_type]
            if len(model_grn) > 0:
                row["jacobian_ap_mean"] = model_grn["jacobian_ap"].mean()
                row["jacobian_ap_std"] = (
                    model_grn["jacobian_ap"].std() if len(SEEDS) > 1 else 0.0
                )
                row["jacobian_auroc_mean"] = model_grn["jacobian_auroc"].mean()
                row["jacobian_auroc_std"] = (
                    model_grn["jacobian_auroc"].std() if len(SEEDS) > 1 else 0.0
                )
                row["graph_ap_mean"] = model_grn["graph_ap"].mean()
                row["graph_ap_std"] = (
                    model_grn["graph_ap"].std() if len(SEEDS) > 1 else 0.0
                )
                row["graph_auroc_mean"] = model_grn["graph_auroc"].mean()
                row["graph_auroc_std"] = (
                    model_grn["graph_auroc"].std() if len(SEEDS) > 1 else 0.0
                )

        # Trajectory metrics - first average over timepoints per seed, then aggregate across seeds
        if traj_df is not None:
            model_traj = traj_df[traj_df["model_type"] == model_type]
            if len(model_traj) > 0:
                seed_averages = (
                    model_traj.groupby("seed")
                    .agg(
                        {
                            "avg_ode_distance": "mean",
                            "avg_mmd2_ode": "mean",
                            "avg_ed_ode": "mean",
                        }
                    )
                    .reset_index()
                )

                row["w2_mean"] = seed_averages["avg_ode_distance"].mean()
                row["w2_std"] = (
                    seed_averages["avg_ode_distance"].std() if len(SEEDS) > 1 else 0.0
                )
                row["mmd2_mean"] = seed_averages["avg_mmd2_ode"].mean()
                row["mmd2_std"] = (
                    seed_averages["avg_mmd2_ode"].std() if len(SEEDS) > 1 else 0.0
                )
                row["ed_mean"] = seed_averages["avg_ed_ode"].mean()
                row["ed_std"] = (
                    seed_averages["avg_ed_ode"].std() if len(SEEDS) > 1 else 0.0
                )

        results.append(row)

    return pd.DataFrame(results)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_BASE_DIR, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Non-Uniform Time Experiments")
    print(f"{'='*60}")
    print(f"Dataset: {DATASET}")
    print(f"Dataset Type: {DATASET_TYPE}")
    print(f"Seeds: {SEEDS}")
    print(f"Time Jitter: {TIME_JITTER}")
    print(f"Models: {MODEL_TYPES}")
    print(f"Results Directory: {results_dir}")
    print(f"{'='*60}\n")

    grn_results_dir = os.path.join(results_dir, "grn_inference")
    traj_results_dir = os.path.join(results_dir, "trajectory_inference")
    os.makedirs(grn_results_dir, exist_ok=True)
    os.makedirs(traj_results_dir, exist_ok=True)

    # Run GRN inference experiments
    print("\n" + "=" * 60)
    print("PHASE 1: GRN INFERENCE (train.py - full training)")
    print("=" * 60)

    grn_success = 0
    for seed in SEEDS:
        if run_grn_inference_sf2m(seed, grn_results_dir):
            grn_success += 1
            print(f"✓ GRN: sf2m, seed={seed}")
        else:
            print(f"✗ GRN: sf2m, seed={seed}")

        if run_grn_inference_rf(seed, grn_results_dir):
            grn_success += 1
            print(f"✓ GRN: rf, seed={seed}")
        else:
            print(f"✗ GRN: rf, seed={seed}")

    # Run trajectory inference experiments
    print("\n" + "=" * 60)
    print("PHASE 2: TRAJECTORY INFERENCE (leave-one-out)")
    print("=" * 60)

    traj_success = 0
    for model_type in MODEL_TYPES:
        for seed in SEEDS:
            if run_leave_one_out(model_type, seed, traj_results_dir):
                traj_success += 1
                print(f"✓ Trajectory: {model_type}, seed={seed}")
            else:
                print(f"✗ Trajectory: {model_type}, seed={seed}")

    total_runs = len(MODEL_TYPES) * len(SEEDS)
    print(f"\n{'='*60}")
    print(f"GRN Inference: {grn_success}/{total_runs} successful")
    print(f"Trajectory Inference: {traj_success}/{total_runs} successful")
    print(f"{'='*60}\n")

    # Aggregate results
    print("Aggregating results...")
    grn_df = extract_grn_metrics_from_logs(grn_results_dir)
    traj_df = aggregate_trajectory_results(traj_results_dir)

    if grn_df is not None:
        grn_df.to_csv(os.path.join(results_dir, "all_grn_results.csv"), index=False)
        print("Saved all GRN results to all_grn_results.csv")

    if traj_df is not None:
        traj_df.to_csv(
            os.path.join(results_dir, "all_trajectory_results.csv"), index=False
        )
        print("Saved all trajectory results to all_trajectory_results.csv")

    final_stats = compute_final_statistics(grn_df, traj_df)
    final_stats.to_csv(os.path.join(results_dir, "final_statistics.csv"), index=False)
    print("Saved final statistics to final_statistics.csv")

    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS (mean ± std across seeds)")
    print(f"{'='*60}")

    for _, row in final_stats.iterrows():
        print(f"\n{row['model_type'].upper()}")
        print("-" * 40)

        if "jacobian_ap_mean" in row and not pd.isna(row.get("jacobian_ap_mean")):
            print(
                f"  GRN Jacobian AP:    {row['jacobian_ap_mean']:.4f} ± {row['jacobian_ap_std']:.4f}"
            )
            print(
                f"  GRN Jacobian AUROC: {row['jacobian_auroc_mean']:.4f} ± {row['jacobian_auroc_std']:.4f}"
            )
            print(
                f"  GRN Graph AP:       {row['graph_ap_mean']:.4f} ± {row['graph_ap_std']:.4f}"
            )
            print(
                f"  GRN Graph AUROC:    {row['graph_auroc_mean']:.4f} ± {row['graph_auroc_std']:.4f}"
            )

        if "w2_mean" in row and not pd.isna(row.get("w2_mean")):
            print(f"  Trajectory W2:      {row['w2_mean']:.4f} ± {row['w2_std']:.4f}")
            print(
                f"  Trajectory MMD2:    {row['mmd2_mean']:.4f} ± {row['mmd2_std']:.4f}"
            )
            print(f"  Trajectory ED:      {row['ed_mean']:.4f} ± {row['ed_std']:.4f}")

    config_info = {
        "dataset": DATASET,
        "dataset_type": DATASET_TYPE,
        "seeds": str(SEEDS),
        "time_jitter": TIME_JITTER,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "alpha": ALPHA,
        "reg": REG,
        "gl_reg": GL_REG,
        "model_types": str(MODEL_TYPES),
        "timestamp": timestamp,
    }
    pd.DataFrame([config_info]).to_csv(
        os.path.join(results_dir, "experiment_config.csv"), index=False
    )

    print(f"\n{'='*60}")
    print(f"All results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
