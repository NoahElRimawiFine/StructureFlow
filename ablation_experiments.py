#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def run_experiment(
    gl_reg, alpha, knockout_hidden, seed, base_results_dir="ablation_results"
):
    """Run a single ablation experiment with specific configuration."""
    results_subdir = os.path.join(
        base_results_dir, f"gl{gl_reg}_alpha{alpha}_hidden{knockout_hidden}_seed{seed}"
    )

    cmd = [
        "python3",
        "-m",
        "src.leave_one_out",
        "--dataset_type",
        "Synthetic",
        "--dataset",
        "dyn-TF",
        "--model_type",
        "sf2m",
        "--seed",
        str(seed),
        "--gl_reg",
        str(gl_reg),
        "--alpha",
        str(alpha),
        "--knockout_hidden",
        str(knockout_hidden),
        "--results_dir",
        results_subdir,
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"success": True, "cmd": " ".join(cmd), "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "cmd": " ".join(cmd),
            "error": str(e),
            "stderr": e.stderr,
        }


def run_experiment_wrapper(experiment_config):
    """Wrapper function for multiprocessing that unpacks experiment configuration."""
    gl_reg, alpha, knockout_hidden, seed, base_results_dir = experiment_config
    return run_experiment(gl_reg, alpha, knockout_hidden, seed, base_results_dir)


def run_experiments_parallel(experiment_configs, num_workers=None):
    """Run multiple experiments in parallel using ProcessPoolExecutor."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(experiment_configs))

    print(
        f"Running {len(experiment_configs)} experiments using {num_workers} parallel workers"
    )

    failed_experiments = []
    completed_runs = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_config = {
            executor.submit(run_experiment_wrapper, config): config
            for config in experiment_configs
        }

        for future in as_completed(future_to_config):
            config = future_to_config[future]
            gl_reg, alpha, knockout_hidden, seed, base_results_dir = config

            try:
                result = future.result()
                completed_runs += 1

                elapsed_time = time.time() - start_time
                avg_time_per_run = elapsed_time / completed_runs
                remaining_runs = len(experiment_configs) - completed_runs
                est_remaining_time = avg_time_per_run * remaining_runs

                if result["success"]:
                    print(
                        f"✓ [{completed_runs}/{len(experiment_configs)}] Completed GL_REG={gl_reg}, alpha={alpha}, hidden={knockout_hidden}, seed={seed}"
                    )
                else:
                    print(
                        f"✗ [{completed_runs}/{len(experiment_configs)}] Failed GL_REG={gl_reg}, alpha={alpha}, hidden={knockout_hidden}, seed={seed}: {result['error']}"
                    )
                    failed_experiments.append((config, result["error"]))

                print(
                    f"   Progress: {completed_runs}/{len(experiment_configs)} | "
                    f"Elapsed: {elapsed_time:.1f}s | "
                    f"Est. remaining: {est_remaining_time:.1f}s"
                )

            except Exception as exc:
                print(
                    f"✗ [{completed_runs+1}/{len(experiment_configs)}] Exception for GL_REG={gl_reg}, alpha={alpha}, hidden={knockout_hidden}, seed={seed}: {exc}"
                )
                failed_experiments.append((config, str(exc)))
                completed_runs += 1

    return failed_experiments


def aggregate_results(
    base_results_dir, seeds, gl_reg_values, alpha_values, hidden_values
):
    """Aggregate results across seeds for each ablation configuration."""
    print("\nAggregating ablation results across seeds...")

    agg_dir = os.path.join(base_results_dir, "aggregate_results")
    os.makedirs(agg_dir, exist_ok=True)

    default_gl_reg = 0.04
    default_alpha = 0.1
    default_hidden = 100

    all_summary_rows = []

    def aggregate_param_sweep(
        param_name, param_values, default_gl, default_a, default_h
    ):
        """Helper function to aggregate results for a specific parameter sweep."""
        print(f"\n--- Aggregating {param_name} sweep ---")
        sweep_results = []

        for param_val in param_values:
            if param_name == "GL_REG":
                gl_reg, alpha, hidden = param_val, default_a, default_h
            elif param_name == "alpha":
                gl_reg, alpha, hidden = default_gl, param_val, default_h
            else:
                gl_reg, alpha, hidden = default_gl, default_a, param_val

            all_seed_summaries = []

            for seed in seeds:
                results_subdir = f"gl{gl_reg}_alpha{alpha}_hidden{hidden}_seed{seed}"
                summary_path = os.path.join(
                    base_results_dir,
                    results_subdir,
                    "Synthetic_sf2m__dyn-TF_seed" + str(seed),
                    f"loo_summary_sf2m_seed{seed}.csv",
                )

                if os.path.exists(summary_path):
                    summary_df = pd.read_csv(summary_path)
                    all_seed_summaries.append(summary_df)
                else:
                    print(f"  Warning: No results found at {summary_path}")

            if not all_seed_summaries:
                print(f"  No results found for {param_name}={param_val}")
                continue

            seed_overall_means = []
            for seed_df in all_seed_summaries:
                seed_overall_means.append(
                    {
                        "avg_ode_distance": seed_df["avg_ode_distance"].mean(),
                        "avg_sde_distance": seed_df["avg_sde_distance"].mean(),
                        "avg_mmd2_ode": seed_df["avg_mmd2_ode"].mean(),
                        "avg_mmd2_sde": seed_df["avg_mmd2_sde"].mean(),
                        "avg_ed_ode": seed_df["avg_ed_ode"].mean(),
                        "avg_ed_sde": seed_df["avg_ed_sde"].mean(),
                        "jacobian_ap": (
                            seed_df["jacobian_ap"].mean()
                            if "jacobian_ap" in seed_df.columns
                            else np.nan
                        ),
                        "jacobian_auroc": (
                            seed_df["jacobian_auroc"].mean()
                            if "jacobian_auroc" in seed_df.columns
                            else np.nan
                        ),
                        "causal_graph_ap": (
                            seed_df["causal_graph_ap"].mean()
                            if "causal_graph_ap" in seed_df.columns
                            else np.nan
                        ),
                        "causal_graph_auroc": (
                            seed_df["causal_graph_auroc"].mean()
                            if "causal_graph_auroc" in seed_df.columns
                            else np.nan
                        ),
                    }
                )

            seed_means_df = pd.DataFrame(seed_overall_means)

            avg_ode = seed_means_df["avg_ode_distance"].mean()
            std_ode = seed_means_df["avg_ode_distance"].std()
            avg_sde = seed_means_df["avg_sde_distance"].mean()
            std_sde = seed_means_df["avg_sde_distance"].std()
            avg_mmd2_ode = seed_means_df["avg_mmd2_ode"].mean()
            std_mmd2_ode = seed_means_df["avg_mmd2_ode"].std()
            avg_mmd2_sde = seed_means_df["avg_mmd2_sde"].mean()
            std_mmd2_sde = seed_means_df["avg_mmd2_sde"].std()
            avg_ed_ode = seed_means_df["avg_ed_ode"].mean()
            std_ed_ode = seed_means_df["avg_ed_ode"].std()
            avg_ed_sde = seed_means_df["avg_ed_sde"].mean()
            std_ed_sde = seed_means_df["avg_ed_sde"].std()

            avg_jac_ap = seed_means_df["jacobian_ap"].mean()
            std_jac_ap = seed_means_df["jacobian_ap"].std()
            avg_jac_auroc = seed_means_df["jacobian_auroc"].mean()
            std_jac_auroc = seed_means_df["jacobian_auroc"].std()
            avg_cg_ap = seed_means_df["causal_graph_ap"].mean()
            std_cg_ap = seed_means_df["causal_graph_ap"].std()
            avg_cg_auroc = seed_means_df["causal_graph_auroc"].mean()
            std_cg_auroc = seed_means_df["causal_graph_auroc"].std()

            sweep_results.append(
                {
                    param_name: param_val,
                    "W-Dist (ODE) Mean": avg_ode,
                    "W-Dist (ODE) Std": std_ode,
                    "W-Dist (SDE) Mean": avg_sde,
                    "W-Dist (SDE) Std": std_sde,
                    "MMD2 (ODE) Mean": avg_mmd2_ode,
                    "MMD2 (ODE) Std": std_mmd2_ode,
                    "MMD2 (SDE) Mean": avg_mmd2_sde,
                    "MMD2 (SDE) Std": std_mmd2_sde,
                    "ED (ODE) Mean": avg_ed_ode,
                    "ED (ODE) Std": std_ed_ode,
                    "ED (SDE) Mean": avg_ed_sde,
                    "ED (SDE) Std": std_ed_sde,
                    "Jacobian AP Mean": avg_jac_ap,
                    "Jacobian AP Std": std_jac_ap,
                    "Jacobian AUROC Mean": avg_jac_auroc,
                    "Jacobian AUROC Std": std_jac_auroc,
                    "Causal Graph AP Mean": avg_cg_ap,
                    "Causal Graph AP Std": std_cg_ap,
                    "Causal Graph AUROC Mean": avg_cg_auroc,
                    "Causal Graph AUROC Std": std_cg_auroc,
                }
            )

        if sweep_results:
            sweep_df = pd.DataFrame(sweep_results)
            sweep_path = os.path.join(agg_dir, f"{param_name}_sweep_results.csv")
            sweep_df.to_csv(sweep_path, index=False)
            print(f"  Saved {param_name} sweep results to {sweep_path}")

            print(f"\n{param_name} Sweep Results:")
            print(sweep_df.to_string(index=False))

            return sweep_df
        return None

    gl_reg_df = aggregate_param_sweep(
        "GL_REG", gl_reg_values, default_gl_reg, default_alpha, default_hidden
    )
    alpha_df = aggregate_param_sweep(
        "alpha", alpha_values, default_gl_reg, default_alpha, default_hidden
    )
    hidden_df = aggregate_param_sweep(
        "KNOCKOUT_HIDDEN", hidden_values, default_gl_reg, default_alpha, default_hidden
    )

    print(f"\nAll ablation results saved to {agg_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for SF2M on dyn-TF"
    )
    parser.add_argument(
        "--base_results_dir",
        type=str,
        default="ablation_results",
        help="Base directory to save results",
    )
    parser.add_argument(
        "--only_aggregate",
        action="store_true",
        help="Only aggregate existing results without running new experiments",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of in parallel",
    )
    args = parser.parse_args()

    default_gl_reg = 0.04
    default_alpha = 0.1
    default_hidden = 100

    gl_reg_values = [0, 0.0001, 0.001, 0.01, 0.04, 0.1, 0.2, 0.5]
    alpha_values = [0.1, 0.2, 0.5, 0.8]
    hidden_values = [10, 50, 100, 200, 256]
    seeds = [1, 2, 3]

    os.makedirs(args.base_results_dir, exist_ok=True)

    start_time = datetime.now()
    print(
        f"Starting ablation experiments at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"\nAblation Configuration:")
    print(f"  GL_REG values: {gl_reg_values}")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Model size (KNOCKOUT_HIDDEN) values: {hidden_values}")
    print(f"  Seeds: {seeds}")
    print(f"\nDefaults (for non-varied parameters):")
    print(f"  GL_REG: {default_gl_reg}")
    print(f"  Alpha: {default_alpha}")
    print(f"  KNOCKOUT_HIDDEN: {default_hidden}")

    if not args.only_aggregate:
        experiment_configs = []

        for seed in seeds:
            for gl_reg in gl_reg_values:
                experiment_configs.append(
                    (gl_reg, default_alpha, default_hidden, seed, args.base_results_dir)
                )

            for alpha in alpha_values:
                experiment_configs.append(
                    (default_gl_reg, alpha, default_hidden, seed, args.base_results_dir)
                )

            for hidden in hidden_values:
                experiment_configs.append(
                    (default_gl_reg, default_alpha, hidden, seed, args.base_results_dir)
                )

        total_runs = len(experiment_configs)
        print(f"\nTotal experiments to run: {total_runs}")
        print(f"  - GL_REG sweep: {len(gl_reg_values) * len(seeds)} runs")
        print(f"  - Alpha sweep: {len(alpha_values) * len(seeds)} runs")
        print(f"  - Model size sweep: {len(hidden_values) * len(seeds)} runs")

        if args.sequential:
            print("\nRunning experiments sequentially...")
            failed_experiments = []
            for i, config in enumerate(experiment_configs):
                gl_reg, alpha, knockout_hidden, seed, base_results_dir = config
                print(
                    f"\n[{i+1}/{total_runs}] Running GL_REG={gl_reg}, alpha={alpha}, hidden={knockout_hidden}, seed={seed}"
                )

                result = run_experiment_wrapper(config)
                if not result["success"]:
                    failed_experiments.append((config, result["error"]))
                    print(f"✗ Failed: {result['error']}")
                else:
                    print("✓ Completed successfully")
        else:
            print(
                f"\nRunning experiments in parallel with {args.num_workers} workers..."
            )
            failed_experiments = run_experiments_parallel(
                experiment_configs, args.num_workers
            )

        if failed_experiments:
            print(f"\n⚠️  {len(failed_experiments)} experiments failed:")
            for config, error in failed_experiments:
                gl_reg, alpha, knockout_hidden, seed, _ = config
                print(
                    f"  - GL_REG={gl_reg}, alpha={alpha}, hidden={knockout_hidden}, seed={seed}: {error}"
                )
        else:
            print("\n✅ All experiments completed successfully!")

    aggregate_results(
        args.base_results_dir, seeds, gl_reg_values, alpha_values, hidden_values
    )

    total_runtime = datetime.now() - start_time
    print(f"\nTotal runtime: {total_runtime}")
    print("All ablation experiments completed successfully.")


if __name__ == "__main__":
    main()
