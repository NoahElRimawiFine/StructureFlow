#!/usr/bin/env python3
"""
Standalone RF hyperparameter sweep for causal discovery.
Focuses on finding optimal hyperparameters for Reference Fitting on N=10 systems.
"""

import itertools
import random
import pandas as pd
import time
from typing import List, Dict, Any
import os

# Import from our scaling experiment
from causal_discovery_experiment import (
    ReferenceFittingMethod,
    CorrelationBasedMethod,
    run_single_experiment_silent,
    set_fixed_cores,
)


def create_rf_hparam_configs() -> List[Dict[str, Any]]:
    """
    Create list of RF hyperparameter configurations to test.
    Returns list of dictionaries, each containing one RF hyperparameter combination.
    """

    # Define RF hyperparameter search space based on ReferenceFittingModule options
    rf_hparam_space = {
        "lr": [0.05, 0.1, 0.2],  # Learning rate for optimizer
        "iter": [50, 100, 150, 200, 300],  # Number of iterations
        "reg_sinkhorn": [0.01, 0.05, 0.1, 0.2],  # Sinkhorn regularization
        "reg_A": [1e-5, 1e-4, 1e-3, 5e-3, 1e-2],  # Adjacency matrix regularization
        "reg_A_elastic": [0],  # Keep elastic regularization at 0 for now
    }

    print(f"RF Hyperparameter search space:")
    for param, values in rf_hparam_space.items():
        print(f"  {param}: {values}")

    # Generate all combinations
    param_names = list(rf_hparam_space.keys())
    param_values = list(rf_hparam_space.values())

    configs = []
    for combination in itertools.product(*param_values):
        config_dict = dict(zip(param_names, combination))
        # Add fixed parameters
        config_dict.update(
            {
                "ot_coupling": True,
                "n_pca_components": 10,
                "device": "cpu",
            }
        )
        configs.append(config_dict)

    print(f"\nTotal RF hyperparameter combinations: {len(configs)}")
    return configs


def create_rf_method_from_hparams(hparams: Dict[str, Any]) -> ReferenceFittingMethod:
    """Create RF method with specific hyperparameters."""
    return ReferenceFittingMethod(hparams, silent=True)


def run_rf_hparam_sweep(
    hparam_configs: List[Dict[str, Any]],
    system_sizes: List[int] = [10],
    seeds: List[int] = [42, 123, 456, 789, 999],
    num_cores: int = 64,
    include_baseline: bool = True,
    sparsity: float = 0.2,
) -> pd.DataFrame:
    """
    Run RF hyperparameter sweep across multiple configurations, system sizes, and seeds.

    Args:
        hparam_configs: List of RF hyperparameter dictionaries
        system_sizes: List of system sizes to test
        seeds: List of random seeds
        num_cores: Number of cores for reproducible timing
        include_baseline: Whether to include correlation baseline
        sparsity: Edge probability for graph generation

    Returns:
        results_df: DataFrame with all experimental results
    """

    # Set fixed cores for reproducible timing
    set_fixed_cores(num_cores)

    all_results = []
    total_experiments = len(hparam_configs) * len(system_sizes) * len(seeds)
    current_exp = 0

    print(f"Starting RF hyperparameter sweep:")
    print(f"  Configurations: {len(hparam_configs)}")
    print(f"  System sizes: {system_sizes}")
    print(f"  Seeds: {seeds}")
    print(f"  Sparsity: {sparsity}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Using {num_cores} cores\n")

    start_time = time.time()

    for config_idx, hparams in enumerate(hparam_configs):
        for num_vars in system_sizes:
            for seed in seeds:
                current_exp += 1

                print(
                    f"[{current_exp}/{total_experiments}] RF Config {config_idx+1}/{len(hparam_configs)}, N={num_vars}, seed={seed}"
                )
                print(
                    f"  RF params: lr={hparams['lr']}, iter={hparams['iter']}, reg_A={hparams['reg_A']}, reg_sinkhorn={hparams['reg_sinkhorn']}"
                )

                # Create methods for this configuration
                methods = []

                # Add baseline if requested
                if include_baseline:
                    methods.append(CorrelationBasedMethod("pearson"))

                # Add RF with current hyperparameters
                rf_method = create_rf_method_from_hparams(hparams)
                methods.append(rf_method)

                # Run single experiment
                try:
                    result = run_single_experiment_silent(
                        num_vars, methods, seed, sparsity=sparsity
                    )

                    # Add hyperparameter information to results
                    for param_name, param_value in hparams.items():
                        result[f"rf_hparam_{param_name}"] = param_value

                    result["config_id"] = config_idx
                    result["experiment_id"] = current_exp
                    result["total_experiments"] = total_experiments

                    all_results.append(result)

                    # Print RF performance
                    if "ReferenceFitting_AUROC" in result:
                        print(
                            f"    RF: AUROC={result['ReferenceFitting_AUROC']:.4f}, AUPRC={result['ReferenceFitting_AUPRC']:.4f}"
                        )

                except Exception as e:
                    print(f"  ERROR in RF experiment: {e}")
                    # Still add a result with NaN values so we track failed configs
                    result = {
                        "num_vars": num_vars,
                        "seed": seed,
                        "config_id": config_idx,
                        "experiment_id": current_exp,
                        "total_experiments": total_experiments,
                        "ReferenceFitting_AUROC": float("nan"),
                        "ReferenceFitting_AUPRC": float("nan"),
                        "ReferenceFitting_training_time": float("nan"),
                        "error": str(e),
                    }

                    # Add hyperparameter information
                    for param_name, param_value in hparams.items():
                        result[f"rf_hparam_{param_name}"] = param_value

                    all_results.append(result)

                # Print minimal progress every 25 experiments
                if current_exp % 25 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_exp = elapsed / current_exp
                    remaining_time = avg_time_per_exp * (
                        total_experiments - current_exp
                    )

                    print(
                        f"  Progress: {current_exp}/{total_experiments} ({100*current_exp/total_experiments:.1f}%) | "
                        f"Elapsed: {elapsed:.0f}s | Est. remaining: {remaining_time:.0f}s"
                    )

    total_time = time.time() - start_time
    print(f"RF hyperparameter sweep completed in {total_time:.2f}s")

    results_df = pd.DataFrame(all_results)
    return results_df


def analyze_rf_hparam_results(
    results_df: pd.DataFrame, output_file: str = "rf_hparam_sweep_results.csv"
):
    """Analyze and save RF hyperparameter sweep results."""

    # Save full results
    results_df.to_csv(output_file, index=False)
    print(f"Full RF results saved to {output_file}")

    # Filter to RF results only for analysis
    rf_results = results_df.dropna(subset=["ReferenceFitting_AUROC"])

    if len(rf_results) == 0:
        print("No successful RF results to analyze!")
        return

    print(f"\nAnalyzing {len(rf_results)} successful RF experiments...")

    # Group by hyperparameters and compute mean performance
    rf_hparam_cols = [col for col in results_df.columns if col.startswith("rf_hparam_")]

    # Performance by configuration
    config_performance = (
        rf_results.groupby(rf_hparam_cols)
        .agg(
            {
                "ReferenceFitting_AUROC": ["mean", "std", "count"],
                "ReferenceFitting_AUPRC": ["mean", "std"],
                "ReferenceFitting_training_time": ["mean", "std"],
            }
        )
        .round(4)
    )

    print("\n=== TOP 15 RF CONFIGURATIONS BY AUPRC ===")
    top_configs = config_performance.sort_values(
        ("ReferenceFitting_AUPRC", "mean"), ascending=False
    ).head(15)
    print(top_configs)

    print("\n=== TOP 15 RF CONFIGURATIONS BY AUROC ===")
    top_configs_auroc = config_performance.sort_values(
        ("ReferenceFitting_AUROC", "mean"), ascending=False
    ).head(15)
    print(top_configs_auroc)

    # Performance by individual hyperparameters
    print("\n=== RF HYPERPARAMETER IMPACT ANALYSIS ===")
    for hparam in [
        "rf_hparam_lr",
        "rf_hparam_iter",
        "rf_hparam_reg_sinkhorn",
        "rf_hparam_reg_A",
        "rf_hparam_reg_A_elastic",
    ]:
        if hparam in rf_results.columns:
            hparam_impact = (
                rf_results.groupby(hparam)
                .agg(
                    {
                        "ReferenceFitting_AUPRC": ["mean", "std", "count"],
                        "ReferenceFitting_AUROC": ["mean", "std"],
                        "ReferenceFitting_training_time": "mean",
                    }
                )
                .round(4)
            )

            print(f"\n{hparam}:")
            print(hparam_impact)

    # Compare with baseline if available
    if "Correlation-pearson_AUPRC" in rf_results.columns:
        print("\n=== RF vs CORRELATION BASELINE ===")
        rf_mean_auprc = rf_results["ReferenceFitting_AUPRC"].mean()
        corr_mean_auprc = rf_results["Correlation-pearson_AUPRC"].mean()
        rf_mean_auroc = rf_results["ReferenceFitting_AUROC"].mean()
        corr_mean_auroc = rf_results["Correlation-pearson_AUROC"].mean()

        print(
            f"RF AUPRC: {rf_mean_auprc:.4f} vs Correlation AUPRC: {corr_mean_auprc:.4f}"
        )
        print(
            f"RF AUROC: {rf_mean_auroc:.4f} vs Correlation AUROC: {corr_mean_auroc:.4f}"
        )

        if rf_mean_auprc > corr_mean_auprc:
            print(
                f"✓ RF outperforms Correlation by {rf_mean_auprc - corr_mean_auprc:.4f} AUPRC"
            )
        else:
            print(
                f"✗ RF underperforms Correlation by {corr_mean_auprc - rf_mean_auprc:.4f} AUPRC"
            )

    # Save top configurations (ranked by AUPRC)
    top_configs_file = output_file.replace(".csv", "_top_configs_by_auprc.csv")
    top_configs.to_csv(top_configs_file)
    print(f"\nTop RF configurations (by AUPRC) saved to {top_configs_file}")

    return top_configs


def main():
    """Main RF hyperparameter sweep runner."""

    print("=" * 60)
    print("REFERENCE FITTING HYPERPARAMETER SWEEP")
    print("=" * 60)

    # Create RF hyperparameter configurations
    rf_hparam_configs = create_rf_hparam_configs()

    # Run RF hyperparameter sweep
    results_df = run_rf_hparam_sweep(
        hparam_configs=rf_hparam_configs,
        system_sizes=[10],  # Focus on N=10 as requested
        seeds=[123],  # 3 seeds for faster iteration
        num_cores=64,
        include_baseline=True,
        sparsity=0.2,
    )

    # Analyze results
    top_configs = analyze_rf_hparam_results(
        results_df, "rf_hparam_sweep_n10_results.csv"
    )

    print("\n" + "=" * 60)
    print("RF HYPERPARAMETER SWEEP COMPLETE")
    print("=" * 60)
    print("Check the output files for detailed results and top configurations.")
    print("Use the best hyperparameters found here in your scaling experiments.")


if __name__ == "__main__":
    main()
