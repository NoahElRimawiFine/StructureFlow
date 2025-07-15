import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io


def plot_scaling_results():
    """
    Plot scaling experiment results showing performance and timing vs system size.
    Uses CSV data with different sparsity levels for StructureFlow and NGM-NODE methods.
    """

    # CSV data from the experiment results
    csv_data = """seed,system_size,sparsity,structureflow_auroc,structureflow_auprc,structureflow_time,ngmnode_auroc,ngmnode_auprc,ngmnode_time
3728,10,0.05,0.9863,0.5889,58.5515,0.9536,0.7143,316.0287
3728,10,0.20,0.9290,0.5838,58.6428,0.7500,0.3324,339.1738
3728,10,0.40,0.8109,0.6799,58.4385,0.7761,0.6428,342.7522
3728,25,0.05,0.9758,0.5849,94.6083,0.8202,0.2659,1061.2054
3728,25,0.20,0.8890,0.5729,97.4592,0.6722,0.3174,1054.7041
3728,25,0.40,0.7393,0.6392,79.8138,0.6249,0.4913,934.0664
3728,50,0.05,0.9483,0.5154,145.0283,0.6417,0.1137,3170.8114
3728,50,0.20,0.7869,0.4816,122.5391,0.5677,0.2288,2773.2565
3728,50,0.40,0.6382,0.5677,121.8200,0.5204,0.4118,1252.6854
3728,100,0.05,0.9543,0.5147,248.4641,0.5311,0.0535,10463.3073
3728,100,0.20,0.7491,0.4620,235.3687,0.5203,0.2061,2967.9328
3728,100,0.40,0.6158,0.5219,232.6693,0.5080,0.3973,2409.0556
3728,200,0.05,0.6071,0.1962,577.1567,,,
3728,200,0.20,0.6872,0.3768,569.9031,,,
3728,200,0.40,0.5705,0.4719,570.0570,,,
3728,500,0.05,0.6351,0.1589,4965.4467,,,
3728,500,0.20,0.5277,0.2181,5328.5185,,,
3728,500,0.40,0.5106,0.4087,5315.9313,,,"""

    df = pd.read_csv(io.StringIO(csv_data))

    # Set up the plotting style
    plt.style.use("default")

    # Define colors and line styles
    colors = {"StructureFlow": "#1f77b4", "NGM-NODE": "#ff7f0e"}
    line_styles = {0.05: "-", 0.20: "--", 0.40: ":"}
    sparsity_labels = {0.05: "5% sparse", 0.20: "20% sparse", 0.40: "40% sparse"}

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: AUROC vs System Size
    for sparsity in [0.05, 0.20, 0.40]:
        sparsity_data = df[df["sparsity"] == sparsity]

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["structureflow_auroc"]
        ax1.plot(
            x_sf,
            y_sf,
            color=colors["StructureFlow"],
            linestyle=line_styles[sparsity],
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["ngmnode_auroc"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["ngmnode_auroc"]
        ax1.plot(
            x_ngm,
            y_ngm,
            color=colors["NGM-NODE"],
            linestyle=line_styles[sparsity],
            marker="s",
            linewidth=2,
            markersize=6,
            label=f"NGM NeuralODE ({sparsity_labels[sparsity]})",
        )

    ax1.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title("AUROC vs System Size", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xscale("log")
    ax1.set_xticks([10, 25, 50, 100, 200, 500])
    ax1.set_xticklabels([10, 25, 50, 100, 200, 500])

    # Plot 2: AUPRC vs System Size
    for sparsity in [0.05, 0.20, 0.40]:
        sparsity_data = df[df["sparsity"] == sparsity]

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["structureflow_auprc"]
        ax2.plot(
            x_sf,
            y_sf,
            color=colors["StructureFlow"],
            linestyle=line_styles[sparsity],
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["ngmnode_auprc"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["ngmnode_auprc"]
        ax2.plot(
            x_ngm,
            y_ngm,
            color=colors["NGM-NODE"],
            linestyle=line_styles[sparsity],
            marker="s",
            linewidth=2,
            markersize=6,
            label=f"NGM NeuralODE ({sparsity_labels[sparsity]})",
        )

    ax2.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax2.set_ylabel("AUPRC", fontsize=12)
    ax2.set_title("AUPRC vs System Size", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xscale("log")
    ax2.set_xticks([10, 25, 50, 100, 200, 500])
    ax2.set_xticklabels([10, 25, 50, 100, 200, 500])

    # Plot 3: Training Time vs System Size (averaged over sparsity levels)
    # Calculate averages for StructureFlow
    sf_avg_data = df.groupby("system_size")["structureflow_time"].mean().reset_index()
    ax3.plot(
        sf_avg_data["system_size"],
        sf_avg_data["structureflow_time"],
        color=colors["StructureFlow"],
        linestyle="-",
        marker="o",
        linewidth=2,
        markersize=8,
        label="StructureFlow (avg across sparsities)",
    )

    # Calculate averages for NGM-NODE (only where data exists)
    ngm_avg_data = (
        df.dropna(subset=["ngmnode_time"])
        .groupby("system_size")["ngmnode_time"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        ngm_avg_data["system_size"],
        ngm_avg_data["ngmnode_time"],
        color=colors["NGM-NODE"],
        linestyle="-",
        marker="s",
        linewidth=2,
        markersize=8,
        label="NGM NeuralODE (avg across sparsities)",
    )

    ax3.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax3.set_ylabel("Training Time (seconds)", fontsize=12)
    ax3.set_title("Training Time vs System Size", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=9, loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")
    ax3.set_xscale("log")
    ax3.set_xticks([10, 25, 50, 100, 200, 500])
    ax3.set_xticklabels([10, 25, 50, 100, 200, 500])

    # Adjust layout
    plt.tight_layout()

    # Save the plots
    plt.savefig("scaling_experiment_plots.png", dpi=300, bbox_inches="tight")
    print("Plots saved to scaling_experiment_plots.png")

    # Show summary statistics
    print("\n" + "=" * 80)
    print("SCALING EXPERIMENT SUMMARY")
    print("=" * 80)

    for sparsity in [0.05, 0.20, 0.40]:
        print(f"\nSparsity: {sparsity_labels[sparsity]}")
        print("-" * 40)
        sparsity_data = df[df["sparsity"] == sparsity]

        for _, row in sparsity_data.iterrows():
            system_size = int(row["system_size"])
            sf_auroc = row["structureflow_auroc"]
            sf_auprc = row["structureflow_auprc"]
            sf_time = row["structureflow_time"]
            ngm_auroc = row["ngmnode_auroc"] if pd.notna(row["ngmnode_auroc"]) else None
            ngm_auprc = row["ngmnode_auprc"] if pd.notna(row["ngmnode_auprc"]) else None
            ngm_time = row["ngmnode_time"] if pd.notna(row["ngmnode_time"]) else None

            print(
                f"  N={system_size:3d}: StructureFlow AUROC={sf_auroc:.4f}, AUPRC={sf_auprc:.4f}, Time={sf_time:6.1f}s",
                end="",
            )
            if ngm_auroc is not None:
                print(
                    f" | NGM-NODE AUROC={ngm_auroc:.4f}, AUPRC={ngm_auprc:.4f}, Time={ngm_time:6.1f}s"
                )
            else:
                print(" | NGM-NODE: N/A")

    # Show the plots
    plt.show()

    return fig, df


def plot_auprc_comparison():
    """
    Note: AUPRC data not available in the new dataset.
    This function is kept for compatibility but will show a message.
    """
    print("AUPRC data not available in the new dataset.")
    return None


if __name__ == "__main__":
    # Create the main plots
    fig, df = plot_scaling_results()

    # AUPRC comparison not available with new data
    plot_auprc_comparison()
