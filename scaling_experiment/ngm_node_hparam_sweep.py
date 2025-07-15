import itertools
import random
import pandas as pd
import time
from typing import List, Dict, Any
import os

from causal_discovery_experiment import (
    NGMNodeMethod,
    CorrelationBasedMethod,
    run_single_experiment_silent,
    set_fixed_cores,
)


def create_ngm_hparam_configs() -> List[Dict[str, Any]]:
    """
    Create list of hyperparameter configurations to test for NGM-NODE.
    Returns list of dictionaries, each containing one hyperparameter combination.
    """

    hparam_space = {
        "n_steps": [2000, 4000, 6000],
        "lr": [0.001, 0.005, 0.01],
        "gl_reg": [0.05],
        "hidden_dim": [128],
        "batch_size": [64],
    }

    print(f"NGM-NODE hyperparameter search space:")
    for param, values in hparam_space.items():
        print(f"  {param}: {values}")

    param_names = list(hparam_space.keys())
    param_values = list(hparam_space.values())

    configs = []
    for combination in itertools.product(*param_values):
        config_dict = dict(zip(param_names, combination))
        configs.append(config_dict)

    print(f"\nTotal hyperparameter combinations: {len(configs)}")
    return configs


def create_ngm_method_from_hparams(hparams: Dict[str, Any]) -> NGMNodeMethod:
    """Create NGM-NODE method with specific hyperparameters."""

    config = {
        "n_steps": hparams["n_steps"],
        "lr": hparams["lr"],
        "gl_reg": hparams["gl_reg"],
        "hidden_dim": hparams["hidden_dim"],
        "batch_size": hparams["batch_size"],
        "l2_reg": 0.01,
        "l1_reg": 0.0,
        "device": "cpu",
    }

    return NGMNodeMethod(config, silent=True)


def run_ngm_hparam_sweep(
    hparam_configs: List[Dict[str, Any]],
    system_sizes: List[int] = [10],
    seeds: List[int] = [42, 123],
    num_cores: int = 4,
    include_baseline: bool = True,
) -> pd.DataFrame:
    """
    Run hyperparameter sweep for NGM-NODE across multiple configurations, system sizes, and seeds.

    Args:
        hparam_configs: List of hyperparameter dictionaries
        system_sizes: List of system sizes to test
        seeds: List of random seeds
        num_cores: Number of cores for reproducible timing
        include_baseline: Whether to include correlation baseline

    Returns:
        results_df: DataFrame with all experimental results
    """

    set_fixed_cores(num_cores)

    all_results = []
    total_experiments = len(hparam_configs) * len(system_sizes) * len(seeds)
    current_exp = 0

    print(f"Starting NGM-NODE hyperparameter sweep:")
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

                methods = []

                if include_baseline:
                    methods.append(CorrelationBasedMethod("pearson"))

                ngm_method = create_ngm_method_from_hparams(hparams)
                methods.append(ngm_method)

                try:
                    result = run_single_experiment_silent(num_vars, methods, seed)

                    for param_name, param_value in hparams.items():
                        result[f"hparam_{param_name}"] = param_value

                    result["config_id"] = config_idx
                    result["experiment_id"] = current_exp
                    result["total_experiments"] = total_experiments

                    all_results.append(result)

                except Exception as e:
                    print(f"  ERROR in experiment: {e}")
                    result = {
                        "num_vars": num_vars,
                        "seed": seed,
                        "config_id": config_idx,
                        "experiment_id": current_exp,
                        "total_experiments": total_experiments,
                        "NGM-NODE_AUROC": float("nan"),
                        "NGM-NODE_AUPRC": float("nan"),
                        "NGM-NODE_training_time": float("nan"),
                        "error": str(e),
                    }

                    for param_name, param_value in hparams.items():
                        result[f"hparam_{param_name}"] = param_value

                    all_results.append(result)

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
    print(f"NGM-NODE hyperparameter sweep completed in {total_time:.2f}s")

    results_df = pd.DataFrame(all_results)
    return results_df


def analyze_ngm_hparam_results(
    results_df: pd.DataFrame, output_file: str = "ngm_node_hparam_sweep_results.csv"
):
    """Analyze and save NGM-NODE hyperparameter sweep results."""

    results_df.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

    ngm_results = results_df.dropna(subset=["NGM-NODE_AUROC"])

    if len(ngm_results) == 0:
        print("No successful NGM-NODE results to analyze!")
        return

    print(f"\nAnalyzing {len(ngm_results)} successful NGM-NODE experiments...")

    hparam_cols = [col for col in results_df.columns if col.startswith("hparam_")]

    config_performance = (
        ngm_results.groupby(hparam_cols)
        .agg(
            {
                "NGM-NODE_AUROC": ["mean", "std", "count"],
                "NGM-NODE_AUPRC": ["mean", "std"],
                "NGM-NODE_training_time": ["mean", "std"],
            }
        )
        .round(4)
    )

    print("\n=== TOP 10 CONFIGURATIONS BY AUPRC ===")
    top_configs = config_performance.sort_values(
        ("NGM-NODE_AUPRC", "mean"), ascending=False
    ).head(10)
    print(top_configs)

    print("\n=== HYPERPARAMETER IMPACT ANALYSIS ===")
    for hparam in [
        "hparam_n_steps",
        "hparam_lr",
        "hparam_gl_reg",
        "hparam_hidden_dim",
        "hparam_batch_size",
        "hparam_l2_reg",
    ]:
        if hparam in ngm_results.columns:
            hparam_impact = (
                ngm_results.groupby(hparam)
                .agg(
                    {
                        "NGM-NODE_AUPRC": ["mean", "std", "count"],
                        "NGM-NODE_AUROC": ["mean", "std"],
                        "NGM-NODE_training_time": "mean",
                    }
                )
                .round(4)
            )

            print(f"\n{hparam}:")
            print(hparam_impact)

    print("\n=== PERFORMANCE BY SYSTEM SIZE ===")
    size_performance = (
        ngm_results.groupby("num_vars")
        .agg(
            {
                "NGM-NODE_AUPRC": ["mean", "std", "count"],
                "NGM-NODE_AUROC": ["mean", "std"],
                "NGM-NODE_training_time": ["mean", "std"],
            }
        )
        .round(4)
    )
    print(size_performance)

    top_configs_file = output_file.replace(".csv", "_top_configs_by_auprc.csv")
    top_configs.to_csv(top_configs_file)
    print(f"\nTop configurations (by AUPRC) saved to {top_configs_file}")


def main():
    """Main NGM-NODE hyperparameter sweep runner."""

    print("=" * 60)
    print("NGM-NODE HYPERPARAMETER SWEEP")
    print("=" * 60)

    hparam_configs = create_ngm_hparam_configs()

    results_df = run_ngm_hparam_sweep(
        hparam_configs=hparam_configs,
        system_sizes=[10],
        seeds=[
            random.randint(0, 1000),
            random.randint(0, 1000),
            random.randint(0, 1000),
        ],
        num_cores=4,
        include_baseline=False,
    )

    analyze_ngm_hparam_results(results_df, "ngm_node_hparam_sweep_results.csv")


if __name__ == "__main__":
    main()
