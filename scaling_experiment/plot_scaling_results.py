import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_scaling_results(csv_file="scaling_experiment_results.csv"):
    """
    Plot scaling experiment results showing performance and timing vs system size.

    Args:
        csv_file: Path to the CSV file containing scaling experiment results
    """

    # Read the results
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} experiments from {csv_file}")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    # Get method names (all columns ending with _AUROC)
    method_names = [
        col.replace("_AUROC", "") for col in df.columns if col.endswith("_AUROC")
    ]
    print(f"Found methods: {method_names}")

    # Group by num_vars and compute means and standard errors
    grouped_stats = []

    for method in method_names:
        method_stats = (
            df.groupby("num_vars")
            .agg(
                {
                    f"{method}_AUROC": ["mean", "std", "count"],
                    f"{method}_AUPRC": ["mean", "std", "count"],
                    f"{method}_training_time": ["mean", "std", "count"],
                }
            )
            .round(4)
        )

        # Flatten column names
        method_stats.columns = [
            "_".join(col).strip() for col in method_stats.columns.values
        ]
        method_stats["method"] = method
        method_stats["num_vars"] = method_stats.index

        grouped_stats.append(method_stats)

    # Combine all method stats
    stats_df = pd.concat(grouped_stats, ignore_index=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: AUROC vs System Size
    for method in method_names:
        method_data = stats_df[stats_df["method"] == method]

        x = method_data["num_vars"]
        y_mean = method_data[f"{method}_AUROC_mean"]
        y_std = method_data[f"{method}_AUROC_std"]
        n_samples = method_data[f"{method}_AUROC_count"]

        # Calculate standard error
        y_stderr = y_std / np.sqrt(n_samples)

        # Plot mean with error bars
        ax1.errorbar(
            x,
            y_mean,
            yerr=y_stderr,
            marker="o",
            linewidth=2,
            markersize=8,
            label=method,
            capsize=5,
            capthick=2,
        )

    ax1.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title(
        "Causal Discovery Performance vs System Size", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)  # AUROC is between 0 and 1

    # Plot 2: Training Time vs System Size
    for method in method_names:
        method_data = stats_df[stats_df["method"] == method]

        x = method_data["num_vars"]
        y_mean = method_data[f"{method}_training_time_mean"]
        y_std = method_data[f"{method}_training_time_std"]
        n_samples = method_data[f"{method}_training_time_count"]

        # Calculate standard error
        y_stderr = y_std / np.sqrt(n_samples)

        # Plot mean with error bars
        ax2.errorbar(
            x,
            y_mean,
            yerr=y_stderr,
            marker="s",
            linewidth=2,
            markersize=8,
            label=method,
            capsize=5,
            capthick=2,
        )

    ax2.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax2.set_ylabel("Training Time (seconds)", fontsize=12)
    ax2.set_title("Training Time vs System Size", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")  # Log scale for time since it may vary dramatically

    # Adjust layout
    plt.tight_layout()

    # Save the plots
    output_file = csv_file.replace(".csv", "_plots.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {output_file}")

    # Show summary statistics
    print("\n" + "=" * 60)
    print("SCALING EXPERIMENT SUMMARY")
    print("=" * 60)

    for method in method_names:
        print(f"\n{method}:")
        method_data = stats_df[stats_df["method"] == method]

        for _, row in method_data.iterrows():
            n_vars = int(row["num_vars"])
            auroc_mean = row[f"{method}_AUROC_mean"]
            auroc_std = row[f"{method}_AUROC_std"]
            time_mean = row[f"{method}_training_time_mean"]
            time_std = row[f"{method}_training_time_std"]
            count = int(row[f"{method}_AUROC_count"])

            print(
                f"  N={n_vars:2d}: AUROC={auroc_mean:.4f}±{auroc_std:.4f}, "
                f"Time={time_mean:6.1f}±{time_std:5.1f}s (n={count})"
            )

    # Show the plots
    plt.show()

    return fig, stats_df


def plot_auprc_comparison(csv_file="scaling_experiment_results.csv"):
    """
    Create an additional plot comparing AUPRC vs system size.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    method_names = [
        col.replace("_AUROC", "") for col in df.columns if col.endswith("_AUROC")
    ]

    # Group by num_vars for AUPRC
    plt.figure(figsize=(10, 6))

    for method in method_names:
        method_data = df.groupby("num_vars").agg(
            {f"{method}_AUPRC": ["mean", "std", "count"]}
        )

        x = method_data.index
        y_mean = method_data[f"{method}_AUPRC"]["mean"]
        y_std = method_data[f"{method}_AUPRC"]["std"]
        n_samples = method_data[f"{method}_AUPRC"]["count"]
        y_stderr = y_std / np.sqrt(n_samples)

        plt.errorbar(
            x,
            y_mean,
            yerr=y_stderr,
            marker="D",
            linewidth=2,
            markersize=8,
            label=method,
            capsize=5,
            capthick=2,
        )

    plt.xlabel("System Size (Number of Variables)", fontsize=12)
    plt.ylabel("AUPRC", fontsize=12)
    plt.title(
        "Area Under Precision-Recall Curve vs System Size",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)

    # Save AUPRC plot
    output_file = csv_file.replace(".csv", "_auprc_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"AUPRC plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    # Create the main plots
    fig, stats_df = plot_scaling_results("scaling_experiment_results.csv")

    # Create additional AUPRC plot
    plot_auprc_comparison("scaling_experiment_results.csv")
