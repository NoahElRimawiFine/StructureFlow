import itertools
import random
import pandas as pd
import time
from typing import List, Dict, Any
import os

# Import from our scaling experiment
from causal_discovery_experiment import (
    SF2MConfig,
    StructureFlowMethod,
    CorrelationBasedMethod,
    run_single_experiment,
    set_fixed_cores,
    run_single_experiment_silent,
)


def create_hparam_configs() -> List[Dict[str, Any]]:
    """
    Create list of hyperparameter configurations to test.
    Returns list of dictionaries, each containing one hyperparameter combination.
    """

    # Define hyperparameter search space (dummy values - user will replace)
    hparam_space = {
        "n_steps": [1000, 2000, 5000],
        "batch_size": [64, 128],
        "reg": [1e-6, 1e-5, 1e-4],
        "alpha": [0.1, 0.3],
        "lr": [1e-3, 3e-3, 5e-3],
        "knockout_hidden": [128, 256],
    }

    print(f"Hyperparameter search space:")
    for param, values in hparam_space.items():
        print(f"  {param}: {values}")

    # Generate all combinations
    param_names = list(hparam_space.keys())
    param_values = list(hparam_space.values())

    configs = []
    for combination in itertools.product(*param_values):
        config_dict = dict(zip(param_names, combination))
        configs.append(config_dict)

    print(f"\nTotal hyperparameter combinations: {len(configs)}")
    return configs


def create_sf2m_method_from_hparams(hparams: Dict[str, Any]) -> StructureFlowMethod:
    """Create SF2M method with specific hyperparameters."""

    # Create SF2M config with the hyperparameters
    config = SF2MConfig(
        base_n_steps=hparams["n_steps"],
        base_lr=hparams["lr"],
        base_alpha=hparams["alpha"],
        base_reg=hparams["reg"],
        base_gl_reg=0.02,  # Keep fixed for now
        base_knockout_hidden=hparams["knockout_hidden"],
        base_score_hidden=[64, 64],  # Keep fixed for now
        base_correction_hidden=[32, 32],  # Keep fixed for now
        base_batch_size=hparams["batch_size"],
        sigma=1.0,
        device="cpu",
    )

    # Get the actual hyperparameter dictionary for a system size of 20
    # (since we're testing on system_sizes=[20] in the main function)
    hyperparams_dict = config.get_scaled_config(num_vars=20)

    return StructureFlowMethod(hyperparams_dict, silent=True)  # Enable silent mode


def run_hparam_sweep(
    hparam_configs: List[Dict[str, Any]],
    system_sizes: List[int] = [10, 20],
    seeds: List[int] = [42, 123, 456, 789, 999],
    num_cores: int = 4,
    include_baseline: bool = True,
    sparsity: float = 0.2,
) -> pd.DataFrame:
    """
    Run hyperparameter sweep across multiple configurations, system sizes, and seeds.

    Args:
        hparam_configs: List of hyperparameter dictionaries
        system_sizes: List of system sizes to test
        seeds: List of random seeds
        num_cores: Number of cores for reproducible timing
        include_baseline: Whether to include correlation baseline

    Returns:
        results_df: DataFrame with all experimental results
    """

    # Set fixed cores for reproducible timing
    set_fixed_cores(num_cores)

    all_results = []
    total_experiments = len(hparam_configs) * len(system_sizes) * len(seeds)
    current_exp = 0

    print(f"Starting hyperparameter sweep:")
    print(f"  Configurations: {len(hparam_configs)}")
    print(f"  System sizes: {system_sizes}")
    print(f"  Seeds: {seeds}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Using {num_cores} cores\n")

    start_time = time.time()

    for config_idx, hparams in enumerate(hparam_configs):
        for num_vars in system_sizes:
            for seed in seeds:
                current_exp += 1

                print(
                    f"[{current_exp}/{total_experiments}] Config {config_idx+1}/{len(hparam_configs)}, N={num_vars}, seed={seed}"
                )

                # Create methods for this configuration
                methods = []

                # Add baseline if requested
                if include_baseline:
                    methods.append(CorrelationBasedMethod("pearson"))

                # Add SF2M with current hyperparameters
                sf2m_method = create_sf2m_method_from_hparams(hparams)
                methods.append(sf2m_method)

                # Run single experiment
                try:
                    # Import the run_single_experiment_silent function
                    result = run_single_experiment_silent(
                        num_vars, methods, seed, sparsity=sparsity
                    )

                    # Add hyperparameter information to results
                    for param_name, param_value in hparams.items():
                        result[f"hparam_{param_name}"] = param_value

                    result["config_id"] = config_idx
                    result["experiment_id"] = current_exp
                    result["total_experiments"] = total_experiments

                    all_results.append(result)

                except Exception as e:
                    print(f"  ERROR in experiment: {e}")
                    # Still add a result with NaN values so we track failed configs
                    result = {
                        "num_vars": num_vars,
                        "seed": seed,
                        "config_id": config_idx,
                        "experiment_id": current_exp,
                        "total_experiments": total_experiments,
                        "StructureFlow_AUROC": float("nan"),
                        "StructureFlow_AUPRC": float("nan"),
                        "StructureFlow_training_time": float("nan"),
                        "error": str(e),
                    }

                    # Add hyperparameter information
                    for param_name, param_value in hparams.items():
                        result[f"hparam_{param_name}"] = param_value

                    all_results.append(result)

                # Print minimal progress every 10 experiments
                if current_exp % 10 == 0:
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
    print(f"Hyperparameter sweep completed in {total_time:.2f}s")

    results_df = pd.DataFrame(all_results)
    return results_df


def analyze_hparam_results(
    results_df: pd.DataFrame, output_file: str = "hparam_sweep_results.csv"
):
    """Analyze and save hyperparameter sweep results."""

    # Save full results
    results_df.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

    # Filter to SF2M results only for analysis
    sf2m_results = results_df.dropna(subset=["StructureFlow_AUROC"])

    if len(sf2m_results) == 0:
        print("No successful SF2M results to analyze!")
        return

    print(f"\nAnalyzing {len(sf2m_results)} successful SF2M experiments...")

    # Group by hyperparameters and compute mean performance
    hparam_cols = [col for col in results_df.columns if col.startswith("hparam_")]

    # Performance by configuration
    config_performance = (
        sf2m_results.groupby(hparam_cols)
        .agg(
            {
                "StructureFlow_AUROC": ["mean", "std", "count"],
                "StructureFlow_AUPRC": ["mean", "std"],
                "StructureFlow_training_time": ["mean", "std"],
            }
        )
        .round(4)
    )

    print("\n=== TOP 10 CONFIGURATIONS BY AUPRC ===")
    top_configs = config_performance.sort_values(
        ("StructureFlow_AUPRC", "mean"), ascending=False
    ).head(10)
    print(top_configs)

    # Performance by individual hyperparameters
    print("\n=== HYPERPARAMETER IMPACT ANALYSIS ===")
    for hparam in [
        "hparam_n_steps",
        "hparam_batch_size",
        "hparam_reg",
        "hparam_alpha",
        "hparam_lr",
        "hparam_knockout_hidden",
    ]:
        if hparam in sf2m_results.columns:
            hparam_impact = (
                sf2m_results.groupby(hparam)
                .agg(
                    {
                        "StructureFlow_AUPRC": ["mean", "std", "count"],
                        "StructureFlow_AUROC": ["mean", "std"],
                        "StructureFlow_training_time": "mean",
                    }
                )
                .round(4)
            )

            print(f"\n{hparam}:")
            print(hparam_impact)

    # System size analysis
    print("\n=== PERFORMANCE BY SYSTEM SIZE ===")
    size_performance = (
        sf2m_results.groupby("num_vars")
        .agg(
            {
                "StructureFlow_AUPRC": ["mean", "std", "count"],
                "StructureFlow_AUROC": ["mean", "std"],
                "StructureFlow_training_time": ["mean", "std"],
            }
        )
        .round(4)
    )
    print(size_performance)

    # Save top configurations (ranked by AUPRC)
    top_configs_file = output_file.replace(".csv", "_top_configs_by_auprc.csv")
    top_configs.to_csv(top_configs_file)
    print(f"\nTop configurations (by AUPRC) saved to {top_configs_file}")


def main():
    """Main hyperparameter sweep runner."""

    print("=" * 60)
    print("SF2M HYPERPARAMETER SWEEP")
    print("=" * 60)

    # Create hyperparameter configurations
    hparam_configs = create_hparam_configs()

    # Run hyperparameter sweep
    results_df = run_hparam_sweep(
        hparam_configs=hparam_configs,
        system_sizes=[20],
        seeds=[random.randint(0, 10000) for _ in range(3)],
        num_cores=32,
        include_baseline=False,
        sparsity=0.05,
    )

    # Analyze results
    analyze_hparam_results(results_df, "hparam_sweep_results.csv")


if __name__ == "__main__":
    main()
