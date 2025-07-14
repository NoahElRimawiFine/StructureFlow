import random
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from causal_discovery_experiment import (
    SF2MConfig,
    DirectSF2MMethod,
    CorrelationBasedMethod,
    generate_causal_system,
    simulate_time_series,
    evaluate_causal_discovery,
    set_fixed_cores,
)


def generate_causal_system_with_sparsity(
    num_vars: int, edge_prob: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random causal system with specified edge probability (sparsity).

    Args:
        num_vars: Number of variables in the system
        edge_prob: Probability of edge existence (controls sparsity)
        seed: Random seed

    Returns:
        adjacency_matrix: True causal adjacency matrix
        dynamics_matrix: System dynamics matrix for simulation
    """
    np.random.seed(seed)
    random.seed(seed)

    # Create adjacency matrix with specified sparsity
    adjacency_matrix = np.zeros((num_vars, num_vars))

    for i in range(num_vars):
        for j in range(num_vars):
            if i != j and np.random.random() < edge_prob:
                # Random weight between -1 and 1, excluding small values
                weight = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0)
                adjacency_matrix[i, j] = weight

    # Dynamics matrix is transpose
    dynamics_matrix = adjacency_matrix.T

    return adjacency_matrix, dynamics_matrix


def run_sparsity_experiment(
    sparsity_levels: List[float],
    num_vars: int = 20,
    seeds: List[int] = None,
    num_cores: int = 4,
) -> pd.DataFrame:
    """
    Run sparsity ablation experiment with SF2M and baseline methods.

    Args:
        sparsity_levels: List of edge probabilities to test
        num_vars: Number of variables (fixed at 20)
        seeds: List of random seeds for averaging
        num_cores: Number of cores for reproducible timing

    Returns:
        results_df: DataFrame containing all experimental results
    """
    if seeds is None:
        seeds = [random.randint(0, 10000) for _ in range(5)]

    set_fixed_cores(num_cores)

    all_results = []
    total_experiments = len(sparsity_levels) * len(seeds)
    current_exp = 0

    print(f"Starting sparsity ablation experiment:")
    print(f"  Fixed dimensionality: N={num_vars}")
    print(f"  Sparsity levels: {sparsity_levels}")
    print(f"  Seeds: {seeds}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Using {num_cores} cores\n")

    # SF2M configuration optimized for N=20
    sf2m_config = {
        "n_steps": 4000,
        "lr": 5e-4,
        "alpha": 0.3,
        "reg": 5e-7,
        "gl_reg": 0.02,
        "knockout_hidden": 128,
        "score_hidden": [64, 64],
        "correction_hidden": [32, 32],
        "batch_size": 64,
        "sigma": 1.0,
        "device": "cpu",
    }

    start_time = time.time()

    for sparsity in sparsity_levels:
        for seed in seeds:
            current_exp += 1

            print(
                f"\n[{current_exp}/{total_experiments}] Sparsity: {sparsity:.3f}, Seed: {seed}"
            )

            # Generate causal system with specified sparsity
            true_adjacency, dynamics_matrix = generate_causal_system_with_sparsity(
                num_vars, sparsity, seed
            )

            # Calculate actual sparsity metrics
            total_possible_edges = num_vars * (num_vars - 1)  # Exclude diagonal
            actual_edges = np.sum(np.abs(true_adjacency) > 0) - np.sum(
                np.diag(np.abs(true_adjacency)) > 0
            )
            actual_sparsity = actual_edges / total_possible_edges

            # Simulate time series data
            time_series_data = simulate_time_series(dynamics_matrix, seed=seed)

            # Test methods
            methods = [
                CorrelationBasedMethod("pearson"),
                DirectSF2MMethod(sf2m_config, silent=True),
            ]

            # Store basic system info
            result = {
                "sparsity_target": sparsity,
                "sparsity_actual": actual_sparsity,
                "num_vars": num_vars,
                "seed": seed,
                "true_edges": actual_edges,
                "experiment_id": current_exp,
                "total_experiments": total_experiments,
            }

            # Test each method
            for method in methods:
                try:
                    start_method_time = time.time()

                    # Fit method
                    if (
                        hasattr(method, "needs_true_adjacency")
                        and method.needs_true_adjacency
                    ):
                        predicted_adjacency = method.fit(
                            time_series_data, true_adjacency
                        )
                    else:
                        predicted_adjacency = method.fit(time_series_data)

                    # Evaluate performance
                    metrics = evaluate_causal_discovery(
                        true_adjacency, predicted_adjacency
                    )
                    training_time = method.get_training_time()

                    # Store results
                    result[f"{method.name}_AUROC"] = metrics["AUROC"]
                    result[f"{method.name}_AUPRC"] = metrics["AUPRC"]
                    result[f"{method.name}_training_time"] = training_time
                    result[f"{method.name}_num_true_edges"] = metrics["num_true_edges"]

                    print(
                        f"    {method.name}: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, Time={training_time:.2f}s"
                    )

                except Exception as e:
                    print(f"    ERROR with {method.name}: {e}")
                    # Store NaN values for failed methods
                    result[f"{method.name}_AUROC"] = float("nan")
                    result[f"{method.name}_AUPRC"] = float("nan")
                    result[f"{method.name}_training_time"] = float("nan")
                    result[f"{method.name}_num_true_edges"] = float("nan")

            all_results.append(result)

    total_time = time.time() - start_time
    print(f"\nSparsity ablation completed in {total_time:.2f}s")

    results_df = pd.DataFrame(all_results)
    return results_df


def analyze_sparsity_results(
    results_df: pd.DataFrame,
    output_file: str = "sparsity_ablation_results.csv",
    create_plots: bool = True,
):
    """Analyze and visualize sparsity ablation results."""

    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Get method names
    method_names = [
        col.replace("_AUROC", "")
        for col in results_df.columns
        if col.endswith("_AUROC")
    ]

    print(f"\n=== SPARSITY ABLATION SUMMARY ===")
    print(f"Methods tested: {method_names}")
    print(f"Sparsity levels: {sorted(results_df['sparsity_target'].unique())}")
    print(f"Total experiments: {len(results_df)}")

    # Summary statistics by sparsity level
    print(f"\n=== PERFORMANCE BY SPARSITY LEVEL ===")

    summary_stats = []
    for sparsity in sorted(results_df["sparsity_target"].unique()):
        sparsity_data = results_df[results_df["sparsity_target"] == sparsity]

        summary_row = {
            "sparsity_target": sparsity,
            "actual_sparsity_mean": sparsity_data["sparsity_actual"].mean(),
            "actual_edges_mean": sparsity_data["true_edges"].mean(),
        }

        for method_name in method_names:
            auroc_col = f"{method_name}_AUROC"
            auprc_col = f"{method_name}_AUPRC"
            time_col = f"{method_name}_training_time"

            if auroc_col in sparsity_data.columns:
                summary_row[f"{method_name}_AUROC_mean"] = sparsity_data[
                    auroc_col
                ].mean()
                summary_row[f"{method_name}_AUROC_std"] = sparsity_data[auroc_col].std()
                summary_row[f"{method_name}_AUPRC_mean"] = sparsity_data[
                    auprc_col
                ].mean()
                summary_row[f"{method_name}_AUPRC_std"] = sparsity_data[auprc_col].std()
                summary_row[f"{method_name}_time_mean"] = sparsity_data[time_col].mean()

        summary_stats.append(summary_row)

    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.round(4))

    # Save summary
    summary_file = output_file.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary statistics saved to {summary_file}")

    if create_plots:
        create_sparsity_plots(results_df, method_names)


def create_sparsity_plots(results_df: pd.DataFrame, method_names: List[str]):
    """Create visualization plots for sparsity ablation results."""

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: AUROC vs Sparsity
    ax1 = axes[0, 0]
    for method_name in method_names:
        auroc_col = f"{method_name}_AUROC"
        if auroc_col in results_df.columns:
            # Group by sparsity and calculate mean and std
            grouped = (
                results_df.groupby("sparsity_target")[auroc_col]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax1.errorbar(
                grouped["sparsity_target"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                label=method_name,
                capsize=5,
            )

    ax1.set_xlabel("Edge Probability (Sparsity)")
    ax1.set_ylabel("AUROC")
    ax1.set_title("AUROC vs Graph Sparsity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: AUPRC vs Sparsity
    ax2 = axes[0, 1]
    for method_name in method_names:
        auprc_col = f"{method_name}_AUPRC"
        if auprc_col in results_df.columns:
            grouped = (
                results_df.groupby("sparsity_target")[auprc_col]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax2.errorbar(
                grouped["sparsity_target"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="s",
                label=method_name,
                capsize=5,
            )

    ax2.set_xlabel("Edge Probability (Sparsity)")
    ax2.set_ylabel("AUPRC")
    ax2.set_title("AUPRC vs Graph Sparsity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training Time vs Sparsity
    ax3 = axes[1, 0]
    for method_name in method_names:
        time_col = f"{method_name}_training_time"
        if time_col in results_df.columns:
            grouped = (
                results_df.groupby("sparsity_target")[time_col]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax3.errorbar(
                grouped["sparsity_target"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="^",
                label=method_name,
                capsize=5,
            )

    ax3.set_xlabel("Edge Probability (Sparsity)")
    ax3.set_ylabel("Training Time (s)")
    ax3.set_title("Training Time vs Graph Sparsity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Actual vs Target Sparsity
    ax4 = axes[1, 1]
    ax4.scatter(
        results_df["sparsity_target"],
        results_df["sparsity_actual"],
        alpha=0.6,
        color="blue",
    )
    ax4.plot(
        [0, results_df["sparsity_target"].max()],
        [0, results_df["sparsity_target"].max()],
        "r--",
        label="Perfect Match",
    )
    ax4.set_xlabel("Target Edge Probability")
    ax4.set_ylabel("Actual Edge Density")
    ax4.set_title("Target vs Actual Graph Sparsity")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sparsity_ablation_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Plots saved as 'sparsity_ablation_plots.png'")


def main():
    """Main sparsity ablation experiment runner."""

    print("=" * 60)
    print("SF2M SPARSITY ABLATION EXPERIMENT")
    print("=" * 60)

    # Define sparsity levels to test
    sparsity_levels = [0.02, 0.05, 0.2, 0.4, 0.8]

    # Generate random seeds
    random_seeds = [random.randint(0, 10000) for _ in range(5)]

    print(f"Experiment setup:")
    print(f"  Fixed dimensionality: N=20")
    print(f"  Sparsity levels: {sparsity_levels}")
    print(f"  Seeds: {random_seeds}")
    print(f"  Total experiments: {len(sparsity_levels) * len(random_seeds)}")

    # Run experiment
    results_df = run_sparsity_experiment(
        sparsity_levels=sparsity_levels,
        num_vars=20,
        seeds=random_seeds,
        num_cores=4,
    )

    # Analyze results
    analyze_sparsity_results(
        results_df, output_file="sparsity_ablation_results.csv", create_plots=True
    )


if __name__ == "__main__":
    main()
