import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_scaling_results():
    """
    Plot scaling experiment results showing performance and timing vs system size.
    Uses hardcoded data for StructureFlow and NGM-NODE methods.
    """

    # Hardcoded data from the experiment results
    data = {
        "num_vars": [10, 25, 50, 75, 10, 25, 50, 75],
        "method": [
            "StructureFlow",
            "StructureFlow",
            "StructureFlow",
            "StructureFlow",
            "NGM-NODE",
            "NGM-NODE",
            "NGM-NODE",
            "NGM-NODE",
        ],
        "AUROC_mean": [0.8798, 0.8241, 0.7757, 0.5583, 0.7881, 0.6517, 0.5586, 0.5298],
        "AUROC_std": [0.0546, 0.0262, 0.0340, 0.0366, 0.0740, 0.0337, 0.0077, 0.0123],
        "AUPRC_mean": [0.6435, 0.5690, 0.5026, 0.2448, 0.5694, 0.3386, 0.2348, 0.2159],
        "AUPRC_std": [0.0871, 0.0553, 0.0490, 0.0289, 0.1135, 0.0642, 0.0105, 0.0048],
        "training_time_mean": [
            58.1914,
            82.5819,
            146.6012,
            183.6806,
            333.1977,
            1009.2917,
            2496.8094,
            3651.4100,
        ],
        "training_time_std": [
            0.6159,
            6.8351,
            12.1040,
            2.4694,
            5.7312,
            14.1373,
            98.9156,
            449.3659,
        ],
    }

    df = pd.DataFrame(data)

    # Create mapping for display names
    method_display_names = {
        "StructureFlow": "StructureFlow",
        "NGM-NODE": "NGM NeuralODE",
    }

    method_names = ["StructureFlow", "NGM-NODE"]
    print(f"Found methods: {method_names}")

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: AUROC vs System Size
    for method in method_names:
        method_data = df[df["method"] == method]
        display_name = method_display_names.get(method, method)

        x = method_data["num_vars"]
        y_mean = method_data["AUROC_mean"]
        y_std = method_data["AUROC_std"]

        # Assume n=4 samples for standard error calculation
        n_samples = 4
        y_stderr = y_std / np.sqrt(n_samples)

        # Plot mean with error bars
        ax1.errorbar(
            x,
            y_mean,
            yerr=y_stderr,
            marker="o",
            linewidth=2,
            markersize=8,
            label=display_name,
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
    ax1.set_ylim(0.5, 1.0)

    # Plot 2: Training Time vs System Size
    for method in method_names:
        method_data = df[df["method"] == method]
        display_name = method_display_names.get(method, method)

        x = method_data["num_vars"]
        y_mean = method_data["training_time_mean"]
        y_std = method_data["training_time_std"]

        # Assume n=4 samples for standard error calculation
        n_samples = 4
        y_stderr = y_std / np.sqrt(n_samples)

        # Plot mean with error bars
        ax2.errorbar(
            x,
            y_mean,
            yerr=y_stderr,
            marker="s",
            linewidth=2,
            markersize=8,
            label=display_name,
            capsize=5,
            capthick=2,
        )

    ax2.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax2.set_ylabel("Training Time (seconds)", fontsize=12)
    ax2.set_title("Training Time vs System Size", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Adjust layout
    plt.tight_layout()

    # Save the plots
    plt.savefig("scaling_experiment_plots.png", dpi=300, bbox_inches="tight")
    print("Plots saved to scaling_experiment_plots.png")

    # Show summary statistics
    print("\n" + "=" * 60)
    print("SCALING EXPERIMENT SUMMARY")
    print("=" * 60)

    for method in method_names:
        display_name = method_display_names.get(method, method)
        print(f"\n{display_name}:")
        method_data = df[df["method"] == method]

        for _, row in method_data.iterrows():
            n_vars = int(row["num_vars"])
            auroc_mean = row["AUROC_mean"]
            auroc_std = row["AUROC_std"]
            time_mean = row["training_time_mean"]
            time_std = row["training_time_std"]

            print(
                f"  N={n_vars:2d}: AUROC={auroc_mean:.4f}±{auroc_std:.4f}, "
                f"Time={time_mean:6.1f}±{time_std:5.1f}s"
            )

    # Show the plots
    plt.show()

    return fig, df


def plot_auprc_comparison():
    """
    Create an additional plot comparing AUPRC vs system size.
    """
    # Same hardcoded data
    data = {
        "num_vars": [10, 25, 50, 75, 10, 25, 50, 75],
        "method": [
            "StructureFlow",
            "StructureFlow",
            "StructureFlow",
            "StructureFlow",
            "NGM-NODE",
            "NGM-NODE",
            "NGM-NODE",
            "NGM-NODE",
        ],
        "AUPRC_mean": [0.6435, 0.5690, 0.5026, 0.2448, 0.5694, 0.3386, 0.2348, 0.2159],
        "AUPRC_std": [0.0871, 0.0553, 0.0490, 0.0289, 0.1135, 0.0642, 0.0105, 0.0048],
    }

    df = pd.DataFrame(data)

    method_names = ["StructureFlow", "NGM-NODE"]

    # Create mapping for display names
    method_display_names = {
        "StructureFlow": "StructureFlow",
        "NGM-NODE": "NGM NeuralODE",
    }

    # Group by num_vars for AUPRC
    plt.figure(figsize=(10, 6))

    for method in method_names:
        method_data = df[df["method"] == method]
        display_name = method_display_names.get(method, method)

        x = method_data["num_vars"]
        y_mean = method_data["AUPRC_mean"]
        y_std = method_data["AUPRC_std"]

        # Assume n=4 samples for standard error calculation
        n_samples = 4
        y_stderr = y_std / np.sqrt(n_samples)

        plt.errorbar(
            x,
            y_mean,
            yerr=y_stderr,
            marker="D",
            linewidth=2,
            markersize=8,
            label=display_name,
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
    plt.savefig("scaling_experiment_auprc_plot.png", dpi=300, bbox_inches="tight")
    print("AUPRC plot saved to scaling_experiment_auprc_plot.png")
    plt.show()


if __name__ == "__main__":
    # Create the main plots
    fig, df = plot_scaling_results()

    # Create additional AUPRC plot
    plot_auprc_comparison()
