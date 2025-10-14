# leave_one_out.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import os
import copy
import anndata as ad
import argparse
import umap

# --- Project Specific Imports ---
from src.datamodules.grn_datamodule import TrajectoryStructureDataModule
from src.models.components.plotting import (
    compute_global_jacobian,
    log_causal_graph_matrices,
    plot_auprs,
)
from src.models.rf_module import ReferenceFittingModule
from src.models.StructureFlow_module import StructureFlowModule
from src.models.components.solver import simulate_trajectory, wasserstein, mmd_squared

# Default configuration values (will be overridden by command line arguments)
DEFAULT_DATA_PATH = "data/"
DEFAULT_DATASET_TYPE = "Synthetic"
DEFAULT_DATASET = "dyn-TF"
DEFAULT_MODEL_TYPE = "sf2m"
DEFAULT_N_STEPS_PER_FOLD = 15000
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-3
DEFAULT_ALPHA = 0.1
DEFAULT_REG = 5e-6
DEFAULT_CORRECTION_REG = 1e-3
DEFAULT_GL_REG = 0.04
DEFAULT_KNOCKOUT_HIDDEN = 100
DEFAULT_SCORE_HIDDEN = [100, 100]
DEFAULT_CORRECTION_HIDDEN = [64, 64]
DEFAULT_SIGMA = 1.0
DEFAULT_N_TIMES_SIM = 100
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 2
DEFAULT_RESULTS_DIR = "loo_results"
DEFAULT_USE_CORRECTION_MLP = True

# --- Renge Specific Parameters ---
# DEFAULT_N_STEPS_PER_FOLD = 10000
# DEFAULT_LR = 0.0002
# DEFAULT_REG = 5e-8
# DEFAULT_ALPHA = 0.1
# DEFAULT_GL_REG = 0.02


def compute_causal_graph_metrics(estimated_graph, true_graph, mask_diagonal=True):
    """
    Compute AP and AUROC metrics for causal graph evaluation.

    Args:
        estimated_graph: Estimated adjacency matrix (numpy array)
        true_graph: True adjacency matrix (numpy array or pandas DataFrame)
        mask_diagonal: Whether to mask diagonal elements

    Returns:
        dict: Dictionary containing AP and AUROC scores
    """
    # Convert DataFrame to numpy if needed
    if hasattr(true_graph, "values"):
        true_graph = true_graph.values

    # Mask diagonal if requested
    if mask_diagonal:
        n = min(true_graph.shape[0], estimated_graph.shape[0])
        true_graph_masked = true_graph.copy()
        estimated_graph_masked = estimated_graph.copy()
        for i in range(n):
            if i < true_graph_masked.shape[0] and i < true_graph_masked.shape[1]:
                true_graph_masked[i, i] = 0
            if (
                i < estimated_graph_masked.shape[0]
                and i < estimated_graph_masked.shape[1]
            ):
                estimated_graph_masked[i, i] = 0
    else:
        true_graph_masked = true_graph
        estimated_graph_masked = estimated_graph

    # Create binary ground truth (presence/absence of edges)
    y_true = np.abs(np.sign(true_graph_masked)).astype(int).flatten()

    # Use absolute values of estimated graph as prediction scores
    y_pred = np.abs(estimated_graph_masked).flatten()

    # Compute metrics
    try:
        if (
            len(np.unique(y_true)) > 1
        ):  # Check if we have both positive and negative samples
            ap_score = average_precision_score(y_true, y_pred)
            auroc_score = roc_auc_score(y_true, y_pred)
        else:
            # If all true labels are the same, metrics are undefined
            ap_score = np.nan
            auroc_score = np.nan
    except Exception as e:
        print(f"Warning: Could not compute AP/AUROC metrics: {e}")
        ap_score = np.nan
        auroc_score = np.nan

    return {
        "ap_score": ap_score,
        "auroc_score": auroc_score,
        "n_true_edges": np.sum(y_true),
        "n_total_edges": len(y_true),
    }


def create_trajectory_pca_plot(
    adata, predictions, ko_name, held_out_time, folder_path, model_type
):
    """
    Create and save a PCA plot showing the entire trajectory with predictions overlay.

    Args:
        adata: AnnData object containing the full trajectory data
        predictions: Model's predicted final state (numpy array)
        ko_name: Name of the knockout for labeling
        held_out_time: The held-out timepoint
        folder_path: Path to save the plot
        model_type: Type of model used ("rf", "sf2m", etc.)
    """
    # Create component folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Get all data and times from the anndata object
    full_data = adata.X
    times = adata.obs["t"].values

    # Fit PCA on all data
    pca = PCA(n_components=2)
    pca.fit(full_data)

    # Transform the data
    full_data_pca = pca.transform(full_data)

    # Transform the predictions
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    pred_pca = pca.transform(predictions)

    # Set larger font sizes
    plt.rcParams.update(
        {
            "font.size": 18,  # Base font size
            "axes.titlesize": 22,  # Title font size
            "axes.labelsize": 20,  # Axis label font size
            "xtick.labelsize": 18,  # X-tick label font size
            "ytick.labelsize": 18,  # Y-tick label font size
            "legend.fontsize": 18,  # Legend font size
        }
    )

    # Create the plot
    plt.figure(figsize=(12, 10))  # Slightly larger figure

    # Plot the true trajectory points, colored by time
    scatter = plt.scatter(
        full_data_pca[:, 0],
        full_data_pca[:, 1],
        c=times,
        cmap="viridis",
        label="True trajectory",
        s=70,  # Larger point size
    )

    # Plot the model predictions
    plt.scatter(
        pred_pca[:, 0],
        pred_pca[:, 1],
        c="salmon",
        s=120,  # Make predictions larger and more visible
        marker="x",
        linewidth=2,  # Thicker lines for the X markers
        label=f"{model_type} predictions",
    )

    # Add colorbar with larger font
    cbar = plt.colorbar(scatter)
    cbar.set_label("Time", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Add title and labels
    ko_label = "Wild Type" if ko_name is None else f"Knockout: {ko_name}"
    plt.title(f"{ko_label} - {model_type} Prediction", fontweight="bold")
    plt.xlabel("PC1", fontweight="bold")
    plt.ylabel("PC2", fontweight="bold")

    # Larger and better positioned legend
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=14)

    # Clearer grid
    plt.grid(True, alpha=0.4, linestyle="--")

    # Add a border around the plot
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)

    # Save the figure
    filename = f"traj_{'wildtype' if ko_name is None else f'ko_{ko_name}'}.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=200, bbox_inches="tight")
    plt.close()

    # Reset rcParams to default to avoid affecting other plots
    plt.rcParams.update(plt.rcParamsDefault)


def create_multi_ko_pca_plot(
    full_adatas, predictions_dict, ko_names, held_out_time, folder_path, model_type
):
    """
    Create and save a PCA plot showing multiple KO trajectories with predictions in subplots.

    Args:
        full_adatas: List of AnnData objects containing the full trajectory data
        predictions_dict: Dictionary mapping dataset_idx to predicted final states
        ko_names: List of knockout names
        held_out_time: The held-out timepoint
        folder_path: Path to save the plot
        model_type: Type of model used ("rf", "sf2m", etc.)
    """
    os.makedirs(folder_path, exist_ok=True)

    if len(ko_names) < 3:
        print(
            "Not enough knockout conditions to create multi-KO plot (need at least 3)"
        )
        return

    ko_indices_to_plot = list(range(min(3, len(ko_names))))

    # Create figure with plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    all_data = np.vstack([adata.X for adata in full_adatas])
    pca = PCA(n_components=2)
    pca.fit(all_data)

    # Increase font sizes
    plt.rcParams.update(
        {
            "font.size": 16,  # Base font size
            "axes.titlesize": 20,  # Subplot title font size
            "axes.labelsize": 20,  # Axis labels font size
            "xtick.labelsize": 16,  # X-tick labels font size
            "ytick.labelsize": 16,  # Y-tick labels font size
            "legend.fontsize": 18,  # Increased legend font size
        }
    )

    # Create a custom colormap that starts darker
    from matplotlib.colors import LinearSegmentedColormap

    # Make a custom colormap that starts with a darker red
    red_cmap_colors = plt.cm.Reds(
        np.linspace(0.3, 1, 256)
    )  # Starting from 30% instead of 0%
    dark_reds = LinearSegmentedColormap.from_list("DarkerReds", red_cmap_colors)

    # Define a pastel blue color for predictions
    pastel_blue = "#86BFEF"  # Light/pastel blue color

    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for i, (ax, ko_idx) in enumerate(zip(axes, ko_indices_to_plot)):
        adata = full_adatas[ko_idx]
        ko_name = ko_names[ko_idx]

        is_knockout = ko_name and "_ko_" in ko_name
        if is_knockout:
            ko_display_name = ko_name.split("_ko_")[-1]
        else:
            ko_display_name = ko_name

        times = adata.obs["t"].values
        ko_data_pca = pca.transform(adata.X)

        if ko_idx in predictions_dict:
            predictions = predictions_dict[ko_idx]
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            pred_pca = pca.transform(predictions)
        else:
            print(f"No predictions available for KO: {ko_name}")
            continue

        # Using custom darker Reds colormap for true data
        scatter = ax.scatter(
            ko_data_pca[:, 0],
            ko_data_pca[:, 1],
            c=times,
            cmap=dark_reds,  # Using custom darker Reds colormap
            label="True samples" if i == 0 else None,
            s=60,
        )

        # Using pastel blue for predictions
        ax.scatter(
            pred_pca[:, 0],
            pred_pca[:, 1],
            c=pastel_blue,  # Pastel blue color
            s=100,
            marker="x",
            linewidth=2,
            label="Predictions" if i == 0 else None,
        )

        x_min = min(x_min, ko_data_pca[:, 0].min(), pred_pca[:, 0].min())
        x_max = max(x_max, ko_data_pca[:, 0].max(), pred_pca[:, 0].max())
        y_min = min(y_min, ko_data_pca[:, 1].min(), pred_pca[:, 1].min())
        y_max = max(y_max, ko_data_pca[:, 1].max(), pred_pca[:, 1].max())

        ax.set_xlabel("PC1", fontsize=20)
        if i == 0:
            ax.set_ylabel("PC2", fontsize=20)

        ko_label = f"Knockout {ko_display_name}" if is_knockout else "Observational"
        ax.set_title(ko_label, pad=10)

        ax.grid(True, alpha=0.3, linestyle="--")

        for spine in ax.spines.values():
            spine.set_visible(True)

    for ax in axes:
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # Apply tight layout
    plt.tight_layout()

    # Get the position of the rightmost plot and first plot
    pos_right = axes[-1].get_position()
    right_edge = pos_right.x1

    # Create space for colorbar to the right of the plots
    cbar_ax = fig.add_axes([right_edge + 0.02, pos_right.y0, 0.02, pos_right.height])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Time", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    # Add legend positioned at the bottom, about 1/3 across the figure width
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.26, 0.12),
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
    )

    # Save the figure
    filename_base = f"multi_ko_comparison_{model_type}_holdout_{held_out_time}"
    plt.savefig(os.path.join(folder_path, f"{filename_base}.pdf"), bbox_inches="tight")
    plt.savefig(
        os.path.join(folder_path, f"{filename_base}.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()

    plt.rcParams.update(plt.rcParamsDefault)


def create_multi_ko_pca_plot_wgrey(
    full_adatas,
    predictions_dict,
    ko_names,
    held_out_time,
    folder_path,
    model_type,
    dataset_type="Synthetic",
):
    """
    Create and save a dimensionality reduction plot showing multiple KO trajectories with predictions in subplots.
    Uses UMAP for Renge data and PCA for other datasets.

    Args:
        full_adatas: List of AnnData objects containing the full trajectory data
        predictions_dict: Dictionary mapping dataset_idx to predicted final states
        ko_names: List of knockout names
        held_out_time: The held-out timepoint
        folder_path: Path to save the plot
        model_type: Type of model used ("rf", "sf2m", etc.)
        dataset_type: Type of dataset ("Renge", "Synthetic", "Curated")
    """
    os.makedirs(folder_path, exist_ok=True)

    if len(ko_names) < 3:
        print(
            "Not enough knockout conditions to create multi-KO plot (need at least 3)"
        )
        return

    ko_indices_to_plot = list(range(min(3, len(ko_names))))

    # Create figure with plots - even more extra space for legend
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))

    all_data = np.vstack([adata.X for adata in full_adatas])

    # Choose dimensionality reduction method based on dataset type
    if dataset_type == "Renge":
        # For Renge, compute UMAP fresh so predictions and data are in same space
        # Use PCA first then UMAP (matching scanpy's approach)
        print("Computing UMAP for Renge data (need to project predictions)")

        # First reduce to PCA space (using 50 components like scanpy default)
        pca_reducer = PCA(n_components=min(50, all_data.shape[0], all_data.shape[1]))
        all_data_pca = pca_reducer.fit_transform(all_data)

        # Then run UMAP on PCA space
        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
        )
        reducer.fit(all_data_pca)
        reduction_method = "UMAP"

        # Store PCA reducer for transforming new data
        reducer.pca_reducer = pca_reducer
    else:
        reducer = PCA(n_components=2)
        reducer.fit(all_data)
        reduction_method = "PCA"

    # Increase font sizes by approximately 20%
    plt.rcParams.update(
        {
            "font.size": 38,  # ~20% increase from 32
            "axes.titlesize": 44,  # ~20% increase from 36
            "axes.labelsize": 44,  # ~20% increase from 36
            "xtick.labelsize": 38,  # ~20% increase from 32
            "ytick.labelsize": 38,  # ~20% increase from 32
            "legend.fontsize": 38,  # ~20% increase from 32
        }
    )

    # Define colors
    highlight_color = "#E41A1C"  # Bright red for the held_out_time
    prediction_color = "#377EB8"  # Blue for predictions

    # Transform all data for consistent plotting bounds
    if dataset_type == "Renge":
        all_data_reduced = reducer.transform(reducer.pca_reducer.transform(all_data))
    else:
        all_data_reduced = reducer.transform(all_data)
    x_min, x_max = all_data_reduced[:, 0].min(), all_data_reduced[:, 0].max()
    y_min, y_max = all_data_reduced[:, 1].min(), all_data_reduced[:, 1].max()

    for i, (ax, ko_idx) in enumerate(zip(axes, ko_indices_to_plot)):
        adata = full_adatas[ko_idx]
        ko_name = ko_names[ko_idx]

        is_knockout = ko_name and "_ko_" in ko_name
        if is_knockout:
            ko_display_name = ko_name.split("_ko_")[-1]
        else:
            ko_display_name = ko_name

        times = adata.obs["t"].values
        if dataset_type == "Renge":
            ko_data_reduced = reducer.transform(reducer.pca_reducer.transform(adata.X))
        else:
            ko_data_reduced = reducer.transform(adata.X)

        if ko_idx in predictions_dict:
            predictions = predictions_dict[ko_idx]
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()

            if dataset_type == "Renge":
                pred_reduced = reducer.transform(
                    reducer.pca_reducer.transform(predictions)
                )
            else:
                pred_reduced = reducer.transform(predictions)
        else:
            print(f"No predictions available for KO: {ko_name}")
            continue

        if dataset_type == "Renge":
            # For Renge: Plot ALL data in grayscale background, then highlight specific condition

            # First plot all data in grayscale with time-based shading
            data_start_idx = 0
            max_time_global = max(
                max(adata_temp.obs["t"].values) for adata_temp in full_adatas
            )

            for adata_idx, adata_temp in enumerate(full_adatas):
                temp_times = adata_temp.obs["t"].values
                temp_data_size = adata_temp.X.shape[0]
                temp_data_reduced = all_data_reduced[
                    data_start_idx : data_start_idx + temp_data_size
                ]

                # Plot all conditions in grayscale
                for t in sorted(set(temp_times)):
                    t_mask = temp_times == t
                    shade = 0.3 + 0.4 * (
                        t / max_time_global
                    )  # Map to darkness between 0.3-0.7
                    gray_color = str(1 - shade)  # Grayscale as string

                    ax.scatter(
                        temp_data_reduced[t_mask, 0],
                        temp_data_reduced[t_mask, 1],
                        c=gray_color,
                        s=60,
                        alpha=0.6,
                        label=(
                            f"t={t}"
                            if i == 0 and adata_idx == 0 and t == min(temp_times)
                            else None
                        ),
                    )

                data_start_idx += temp_data_size

            # Now highlight the specific knockout condition at held_out_time
            is_held_out = times == held_out_time
            if any(is_held_out):
                ax.scatter(
                    ko_data_reduced[is_held_out, 0],
                    ko_data_reduced[is_held_out, 1],
                    c=highlight_color,
                    s=80,
                    label=f"t={held_out_time} (held out)" if i == 0 else None,
                )

            # Plot predictions
            ax.scatter(
                pred_reduced[:, 0],
                pred_reduced[:, 1],
                c=prediction_color,
                s=100,
                marker="x",
                linewidth=2,
                label="Predictions" if i == 0 else None,
            )

        else:
            # Original logic for non-Renge datasets
            # Create mask for different time points
            is_held_out = times == held_out_time

            # Plot non-held-out times in grayscale with darker shades for later times
            for t in sorted(set(times)):
                if t == held_out_time:
                    continue

                t_mask = times == t
                shade = 0.3 + 0.5 * (t / max(times))  # Map to darkness between 0.3-0.8
                gray_color = str(1 - shade)  # Grayscale as string: '0.2' to '0.7'

                ax.scatter(
                    ko_data_reduced[t_mask, 0],
                    ko_data_reduced[t_mask, 1],
                    c=gray_color,
                    s=60,
                    label=f"t={t}" if i == 0 and t == min(times) else None,
                )

            # Plot held-out time in highlight color
            if any(is_held_out):
                ax.scatter(
                    ko_data_reduced[is_held_out, 0],
                    ko_data_reduced[is_held_out, 1],
                    c=highlight_color,
                    s=80,
                    label=f"t={held_out_time} (held out)" if i == 0 else None,
                )

            # Plot predictions
            ax.scatter(
                pred_reduced[:, 0],
                pred_reduced[:, 1],
                c=prediction_color,
                s=100,
                marker="x",
                linewidth=2,
                label="Predictions" if i == 0 else None,
            )

        # Use appropriate axis labels based on reduction method
        component1_label = "UMAP1" if reduction_method == "UMAP" else "PC1"
        component2_label = "UMAP2" if reduction_method == "UMAP" else "PC2"

        ax.set_xlabel(component1_label, fontsize=44)
        if i == 0:
            ax.set_ylabel(component2_label, fontsize=44)

        ko_label = f"Knockout {ko_display_name}" if is_knockout else "Observational"
        ax.set_title(ko_label, pad=15, fontsize=44)

        # Keep original grid
        ax.grid(True, alpha=0.3, linestyle="--")

        for spine in ax.spines.values():
            spine.set_visible(True)

    for ax in axes:
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # First, apply tight layout to get good spacing for the plots
    plt.tight_layout()

    # Move the plots up significantly to make room for the legend at the bottom
    plt.subplots_adjust(bottom=0.35)

    # Add legend positioned well below the plots with increased font size
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=min(len(handles), 3),
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=38,  # Increased by ~20% from 32
    )

    # Save the figure
    filename_base = f"multi_ko_comparison_{model_type}_holdout_{held_out_time}"
    plt.savefig(os.path.join(folder_path, f"{filename_base}.pdf"), bbox_inches="tight")
    plt.savefig(
        os.path.join(folder_path, f"{filename_base}.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()

    plt.rcParams.update(plt.rcParamsDefault)


def main(args):
    # Extract configuration from arguments
    DATA_PATH = args.data_path
    DATASET_TYPE = args.dataset_type
    DATASET = args.dataset
    N_STEPS_PER_FOLD = args.n_steps_per_fold
    BATCH_SIZE = args.batch_size
    LR = args.lr
    ALPHA = args.alpha
    REG = args.reg
    CORRECTION_REG = args.correction_reg
    GL_REG = args.gl_reg
    KNOCKOUT_HIDDEN = args.knockout_hidden
    SCORE_HIDDEN = [int(x) for x in args.score_hidden.split(",")]
    CORRECTION_HIDDEN = [int(x) for x in args.correction_hidden.split(",")]
    SIGMA = args.sigma
    N_TIMES_SIM = args.n_times_sim
    DEVICE = args.device
    SEED = args.seed
    RESULTS_DIR = args.results_dir
    MODEL_TYPE = args.model_type
    USE_CORRECTION_MLP = args.use_correction_mlp

    # Create results directory with model type and seed info
    RESULTS_DIR = os.path.join(
        RESULTS_DIR,
        f"{DATASET_TYPE}_{MODEL_TYPE}_{'_' + DATASET if DATASET_TYPE == 'Synthetic' else ''}_seed{SEED}",
    )

    seed_everything(SEED, workers=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Load Full Data Once ---
    print("Loading full dataset...")
    datamodule = TrajectoryStructureDataModule(
        data_path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        dataset=DATASET,
        batch_size=BATCH_SIZE,
        use_dummy_train_loader=True,
        train_val_test_split=(1, 0, 0),
        dummy_loader_steps=N_STEPS_PER_FOLD,
        num_workers=11,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    full_adatas = datamodule.get_subset_adatas()
    if not full_adatas:
        raise ValueError("No datasets loaded.")
    T_max = int(max(adata.obs["t"].max() for adata in full_adatas))
    T_times = T_max + 1
    DT_data = 1.0 / T_times

    print(f"Data loaded. Found {len(full_adatas)} datasets with T_max={T_max}.")

    # Print data statistics for verification (especially useful for Renge)
    if DATASET_TYPE == "Renge":
        total_cells = sum(adata.shape[0] for adata in full_adatas)
        n_genes = full_adatas[0].shape[1] if full_adatas else 0
        print(f"Renge data: {total_cells} total cells, {n_genes} genes")
        print(f"Knockout conditions: {datamodule.kos}")
        for i, (adata, ko) in enumerate(zip(full_adatas, datamodule.kos)):
            print(f"  KO {i} ({ko}): {adata.shape[0]} cells")
    print(f"Kos: {datamodule.kos}")
    print(f"Using model type: {MODEL_TYPE}")

    # --- 2. Leave-One-Out Cross-Validation Loop ---
    results = []
    all_fold_metrics = []
    timepoints_to_hold_out = range(1, T_times - 1)

    for held_out_time in timepoints_to_hold_out:
        fold_name = f"{MODEL_TYPE}_holdout_{held_out_time}"
        print(f"\n===== Training Fold: {fold_name} =====")

        # --- 2.1 Create Filtered Data for this Fold ---
        print(f"Creating filtered dataset excluding t={held_out_time}...")
        fold_adatas = []
        for adata_orig in full_adatas:
            adata_filt = adata_orig[adata_orig.obs["t"] != held_out_time].copy()
            fold_adatas.append(adata_filt)

        # --- 2.3 Instantiate and Train Model ---
        model = None

        if MODEL_TYPE == "rf":
            print("Using Reference Fitting model...")
            # Initialize RF model
            model = ReferenceFittingModule(
                use_cuda=(DEVICE == "cuda"),
                iter=(5000 if DATASET_TYPE == "Renge" else 1000),
            )

            # Fit the model with holdout time (RF handles data filtering internally)
            print(f"Fitting RF model with holdout time {held_out_time}...")
            model.fit_model_with_holdout(fold_adatas, datamodule.kos, held_out_time)

            # No Lightning Trainer used for RF
            print("RF model fitting complete.")

        else:  # "sf2m" or "mlp_baseline"
            print(f"Using {'SF2M' if MODEL_TYPE=='sf2m' else 'MLP Baseline'} model...")

            # Configure correct flags based on MODEL_TYPE
            use_mlp = MODEL_TYPE == "mlp_baseline"
            use_correction = USE_CORRECTION_MLP and not use_mlp

            model = StructureFlowModule(
                datamodule=datamodule,
                T=T_times,
                sigma=SIGMA,
                dt=DT_data,
                batch_size=BATCH_SIZE,
                alpha=ALPHA,
                reg=REG,
                correction_reg_strength=CORRECTION_REG,
                n_steps=N_STEPS_PER_FOLD,
                lr=LR,
                device=DEVICE,
                GL_reg=GL_REG,
                knockout_hidden=KNOCKOUT_HIDDEN,
                score_hidden=SCORE_HIDDEN,
                correction_hidden=CORRECTION_HIDDEN,
                enable_epoch_end_hook=False,
                use_mlp_baseline=use_mlp,
                use_correction_mlp=use_correction,
                held_out_time=held_out_time,
            )

            # Train the model with Lightning
            print("Setting up Trainer...")
            # fold_logger = TensorBoardLogger(RESULTS_DIR, name=fold_name)
            trainer = Trainer(
                max_epochs=-1,
                max_steps=N_STEPS_PER_FOLD,
                accelerator="cpu" if DEVICE == "cpu" else "gpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
                log_every_n_steps=N_STEPS_PER_FOLD + 1,
                num_sanity_val_steps=0,
                limit_val_batches=0.0,
            )

            print(f"Training model for {N_STEPS_PER_FOLD} steps...")
            trainer.fit(model, datamodule=datamodule)
            print("Training complete.")

        # --- 2.6 Evaluate on the Held-Out Timepoint ---
        print(
            f"Evaluating model on predicting t={held_out_time} from t={held_out_time-1}..."
        )
        model.eval()
        if MODEL_TYPE != "rf":  # SF2M and MLP need explicit device placement
            model.to(DEVICE)

        # --- 2.6.1 Compute Causal Graph Metrics ---
        causal_metrics = {}
        try:
            if MODEL_TYPE == "rf":
                # For RF model, get the adjacency matrix directly
                interaction_matrix = model.get_interaction_matrix()
                if interaction_matrix is not None:
                    # Convert to numpy if it's a tensor
                    if hasattr(interaction_matrix, "detach"):
                        W_v = interaction_matrix.detach().cpu().numpy()
                    else:
                        W_v = np.array(interaction_matrix)
                    A_estim = W_v.copy()  # For RF, they are the same
                else:
                    print("Warning: RF model has no interaction matrix")
                    W_v = None
                    A_estim = None
            else:  # SF2M or MLP Baseline
                # Compute the global Jacobian for SF2M models
                with torch.no_grad():
                    A_estim = compute_global_jacobian(
                        model.func_v, fold_adatas, dt=DT_data, device=DEVICE
                    )
                # Get the causal graph from the model
                W_v = model.func_v.causal_graph(w_threshold=0.0).T

            # Get the ground truth matrix
            true_matrix = datamodule.true_matrix

            # Compute metrics for both Jacobian and causal graph if available
            if A_estim is not None and W_v is not None:
                # Handle different dataset types for matrix alignment
                if DATASET_TYPE == "Renge":
                    # Get gene names from the dataset
                    gene_names = datamodule.adatas[0].var_names

                    # Get reference network rows and columns
                    ref_rows = true_matrix.index
                    ref_cols = true_matrix.columns

                    # Create DataFrames for the estimated graphs with all genes
                    A_estim_df = pd.DataFrame(
                        A_estim, index=gene_names, columns=gene_names
                    )
                    W_v_df = pd.DataFrame(W_v, index=gene_names, columns=gene_names)

                    # Extract the exact subset that corresponds to the reference network dimensions
                    A_estim_subset = A_estim_df.loc[ref_rows, ref_cols]
                    W_v_subset = W_v_df.loc[ref_rows, ref_cols]

                    # Convert to numpy arrays for evaluation
                    A_estim_eval = A_estim_subset.values
                    W_v_eval = W_v_subset.values
                    A_true_eval = true_matrix.values
                    mask_diag = False  # Don't mask diagonal for Renge
                else:
                    # For synthetic data, use matrices directly
                    A_estim_eval = A_estim
                    W_v_eval = W_v
                    A_true_eval = true_matrix
                    mask_diag = True  # Mask diagonal for synthetic data

                # Compute metrics for Jacobian-based estimation
                jacobian_metrics = compute_causal_graph_metrics(
                    A_estim_eval, A_true_eval, mask_diagonal=mask_diag
                )
                causal_metrics["jacobian_ap"] = jacobian_metrics["ap_score"]
                causal_metrics["jacobian_auroc"] = jacobian_metrics["auroc_score"]

                # Compute metrics for causal graph extraction
                causal_graph_metrics = compute_causal_graph_metrics(
                    W_v_eval, A_true_eval, mask_diagonal=mask_diag
                )
                causal_metrics["causal_graph_ap"] = causal_graph_metrics["ap_score"]
                causal_metrics["causal_graph_auroc"] = causal_graph_metrics[
                    "auroc_score"
                ]

                print(
                    f"  Causal Graph Metrics - Jacobian AP: {jacobian_metrics['ap_score']:.4f}, AUROC: {jacobian_metrics['auroc_score']:.4f}"
                )
                print(
                    f"  Causal Graph Metrics - Graph AP: {causal_graph_metrics['ap_score']:.4f}, AUROC: {causal_graph_metrics['auroc_score']:.4f}"
                )
            else:
                print(
                    "  Warning: Could not compute causal graph metrics (missing matrices)"
                )
                causal_metrics["jacobian_ap"] = np.nan
                causal_metrics["jacobian_auroc"] = np.nan
                causal_metrics["causal_graph_ap"] = np.nan
                causal_metrics["causal_graph_auroc"] = np.nan

        except Exception as e:
            print(f"  Warning: Error computing causal graph metrics: {e}")
            causal_metrics["jacobian_ap"] = np.nan
            causal_metrics["jacobian_auroc"] = np.nan
            causal_metrics["causal_graph_ap"] = np.nan
            causal_metrics["causal_graph_auroc"] = np.nan

        # Create folder for trajectory PCA plots
        # Only create plots for the last timepoint (T_max)
        if held_out_time == T_max - 1:
            pca_folder = os.path.join(RESULTS_DIR, "pca_plots")
            fold_pca_folder = os.path.join(pca_folder, f"{MODEL_TYPE}_final_trajectory")
            os.makedirs(fold_pca_folder, exist_ok=True)

        fold_distances_list = []
        predictions_dict = {}  # Store predictions for multi-KO plot

        with torch.no_grad():
            # Iterate through the original full datasets
            for i, adata_full in enumerate(full_adatas):
                ko_name = datamodule.kos[i]

                # Get initial conditions (t-1) and true final state (t)
                x0_np = adata_full.X[adata_full.obs["t"] == held_out_time - 1]
                true_dist_np = adata_full.X[adata_full.obs["t"] == held_out_time]

                # Check if data exists for this transition
                if x0_np.shape[0] == 0 or true_dist_np.shape[0] == 0:
                    print(
                        f"  Skipping dataset {i} (KO: {ko_name}): No data for t={held_out_time-1} -> t={held_out_time}."
                    )
                    continue

                x0 = torch.from_numpy(x0_np).float()
                true_dist_cpu = torch.from_numpy(true_dist_np).float()

                if MODEL_TYPE == "rf":
                    # RF simulates differently than SF2M
                    # Always use non-wildtype (full) model for RF simulation
                    is_wildtype = False  # Always use the full model

                    # Simulate trajectory using RF model
                    sort_indices = torch.argsort(x0[:, 0])
                    x0_sorted = x0[sort_indices]
                    traj_rf = model.simulate_trajectory(
                        x0_sorted,
                        n_times=1,  # Simulate one time step
                        use_wildtype=is_wildtype,  # Always use full model
                        n_points=N_TIMES_SIM,
                    )

                    # RF simulation result
                    # traj_rf = traj_rf[-1].cpu()

                    # Calculate metrics
                    w_dist_rf = wasserstein(traj_rf, true_dist_cpu)
                    mmd2_rf = mmd_squared(traj_rf, true_dist_cpu)

                    # Use same values for both ODE and SDE metrics for consistent formatting
                    w_dist_ode = w_dist_rf
                    w_dist_sde = w_dist_rf
                    mmd2_ode = mmd2_rf
                    mmd2_sde = mmd2_rf

                    # Store predictions for multi-KO plot
                    if held_out_time == T_max - 1:
                        predictions_dict[i] = traj_rf.numpy()

                        # Create individual trajectory PCA plot
                        create_trajectory_pca_plot(
                            adata_full,
                            traj_rf.numpy(),
                            ko_name,
                            held_out_time,
                            fold_pca_folder,
                            "RF",
                        )

                else:  # SF2M or MLP Baseline
                    # Move tensors to the right device
                    x0 = x0.to(DEVICE)
                    true_dist = true_dist_cpu.to(DEVICE)

                    # Get conditional vector
                    cond_vector_template = model.conditionals[i].to(DEVICE)
                    if (
                        cond_vector_template is not None
                        and cond_vector_template.nelement() > 0
                    ):
                        cond_vector = cond_vector_template[0].repeat(x0.shape[0], 1)
                    else:
                        n_genes = model.n_genes
                        if model.score_net.conditional:
                            cond_dim = model.score_net.conditional_dim
                            cond_vector = torch.zeros(
                                x0.shape[0], cond_dim, device=DEVICE
                            )
                        else:
                            cond_vector = None

                    # Common simulation arguments
                    common_sim_args = {
                        "flow_model": model.func_v,
                        "corr_model": model.v_correction,
                        "score_model": model.score_net,
                        "x0": x0,
                        "dataset_idx": i,
                        "start_time": held_out_time - 1,
                        "end_time": held_out_time,
                        "n_times": N_TIMES_SIM,
                        "cond_vector": cond_vector,
                        "T": T_times,
                        "sigma": SIGMA,
                        "device": DEVICE,
                    }

                    # ODE Simulation
                    traj_ode = simulate_trajectory(**common_sim_args, use_sde=False)
                    sim_ode_final = traj_ode[-1].cpu()

                    # SDE Simulation
                    traj_sde = simulate_trajectory(**common_sim_args, use_sde=True)
                    sim_sde_final = traj_sde[-1].cpu()

                    # Compute metrics
                    w_dist_ode = wasserstein(sim_ode_final, true_dist_cpu)
                    w_dist_sde = wasserstein(sim_sde_final, true_dist_cpu)
                    mmd2_ode = mmd_squared(sim_ode_final, true_dist_cpu)
                    mmd2_sde = mmd_squared(sim_sde_final, true_dist_cpu)

                    # Store predictions for multi-KO plot (using ODE predictions)
                    if held_out_time == T_max - 1:
                        predictions_dict[i] = sim_ode_final.numpy()

                        # Create individual trajectory PCA plot
                        create_trajectory_pca_plot(
                            adata_full,
                            sim_ode_final.numpy(),
                            ko_name,
                            held_out_time,
                            fold_pca_folder,
                            MODEL_TYPE,
                        )

                # Store metrics
                fold_distances_list.append(
                    {
                        "dataset_idx": i,
                        "ko": ko_name,
                        "w_dist_ode": w_dist_ode,
                        "w_dist_sde": w_dist_sde,
                        "mmd2_ode": mmd2_ode,
                        "mmd2_sde": mmd2_sde,
                    }
                )

                # Log metrics
                print(
                    f"  Dataset {i} (KO: {ko_name}): W_dist(ODE)={w_dist_ode:.4f}, MMD2(ODE)={mmd2_ode:.4f}"
                )
                if MODEL_TYPE != "rf":
                    print(
                        f"  Dataset {i} (KO: {ko_name}): W_dist(SDE)={w_dist_sde:.4f}, MMD2(SDE)={mmd2_sde:.4f}"
                    )

                if held_out_time == T_max - 1:
                    print(
                        f"  Created trajectory PCA plot for dataset {i} (KO: {ko_name})"
                    )

        # Create multi-KO plot if we're at the last timepoint and have predictions
        if held_out_time == T_max - 1 and predictions_dict:
            create_multi_ko_pca_plot_wgrey(
                full_adatas,
                predictions_dict,
                datamodule.kos,
                held_out_time,
                fold_pca_folder,
                MODEL_TYPE,
                DATASET_TYPE,
            )
            print(
                f"  Created multi-KO comparison plot with {len(predictions_dict)} conditions"
            )
        # --- 2.7 Aggregate and Store Results for this Fold ---
        if fold_distances_list:
            fold_df = pd.DataFrame(fold_distances_list)
            avg_ode_dist = fold_df["w_dist_ode"].mean()
            avg_sde_dist = fold_df["w_dist_sde"].mean()
            avg_mmd2_ode = fold_df["mmd2_ode"].mean()
            avg_mmd2_sde = fold_df["mmd2_sde"].mean()

            print(f"Fold {fold_name} Avg W_dist: ODE={avg_ode_dist:.4f}")
            if MODEL_TYPE != "rf":
                print(f"Fold {fold_name} Avg W_dist: SDE={avg_sde_dist:.4f}")

            # Add causal graph metrics to the results
            fold_result = {
                "held_out_time": held_out_time,
                "model_type": MODEL_TYPE,
                "avg_ode_distance": avg_ode_dist,
                "avg_sde_distance": avg_sde_dist,
                "avg_mmd2_ode": avg_mmd2_ode,
                "avg_mmd2_sde": avg_mmd2_sde,
            }

            # Add causal graph metrics
            fold_result.update(causal_metrics)

            results.append(fold_result)

            # Add individual distances to a global list for detailed analysis later
            for record in fold_distances_list:
                record["held_out_time"] = held_out_time
                record["model_type"] = MODEL_TYPE
                record["seed"] = SEED
                record["dataset_type"] = DATASET_TYPE
                all_fold_metrics.append(record)
        else:
            print(f"Fold {fold_name}: No evaluation results.")
            fold_result = {
                "held_out_time": held_out_time,
                "model_type": MODEL_TYPE,
                "avg_ode_distance": np.nan,
                "avg_sde_distance": np.nan,
                "avg_mmd2_ode": np.nan,
                "avg_mmd2_sde": np.nan,
            }
            # Add causal graph metrics (will be NaN if not computed)
            fold_result.update(causal_metrics)
            results.append(fold_result)

        # if MODEL_TYPE == "sf2m" and held_out_time == T_max:  # Only for SF2M
        #     with torch.no_grad():
        #         A_estim = compute_global_jacobian(model.func_v, datamodule.adatas, dt=DT_data, device="cpu")

        #     # Get the causal graph from the model
        #     W_v = model.func_v.causal_graph(w_threshold=0.0).T

        #     # Get the ground truth matrix
        #     A_true = model.true_matrix
        #     # Also display AUPR plot
        #     plot_auprs(W_v, A_estim, A_true)
        #     log_causal_graph_matrices(A_estim, W_v, A_true)

    # --- 3. Final Reporting ---
    print(f"\n===== Leave-One-Out Cross-Validation Summary ({MODEL_TYPE}) =====")
    if results:
        summary_df = pd.DataFrame(results)
        summary_df["seed"] = SEED
        summary_df["dataset_type"] = DATASET_TYPE

        print("Average Metrics per Fold:")
        print(summary_df.to_string(index=False))

        # Calculate overall averages
        final_avg_ode = summary_df["avg_ode_distance"].mean()
        final_avg_sde = summary_df["avg_sde_distance"].mean()
        final_avg_mmd2_ode = summary_df["avg_mmd2_ode"].mean()
        final_avg_mmd2_sde = summary_df["avg_mmd2_sde"].mean()

        final_std_ode = summary_df["avg_ode_distance"].std()
        final_std_sde = summary_df["avg_sde_distance"].std()
        final_std_mmd2_ode = summary_df["avg_mmd2_ode"].std()
        final_std_mmd2_sde = summary_df["avg_mmd2_sde"].std()

        print(
            f"\nOverall Average W-Distance (ODE): {final_avg_ode:.4f} +/- {final_std_ode:.4f}"
        )
        print(
            f"Overall Average W-Distance (SDE): {final_avg_sde:.4f} +/- {final_std_sde:.4f}"
        )
        print(
            f"Overall Average MMD2 (ODE): {final_avg_mmd2_ode:.4f} +/- {final_std_mmd2_ode:.4f}"
        )
        print(
            f"Overall Average MMD2 (SDE): {final_avg_mmd2_sde:.4f} +/- {final_std_mmd2_sde:.4f}"
        )

        # Calculate and report causal graph metrics if available
        if "jacobian_ap" in summary_df.columns:
            final_avg_jac_ap = summary_df["jacobian_ap"].mean()
            final_std_jac_ap = summary_df["jacobian_ap"].std()
            final_avg_jac_auroc = summary_df["jacobian_auroc"].mean()
            final_std_jac_auroc = summary_df["jacobian_auroc"].std()

            print(f"\nCausal Graph Metrics (Jacobian-based):")
            print(
                f"Overall Average AP: {final_avg_jac_ap:.4f} +/- {final_std_jac_ap:.4f}"
            )
            print(
                f"Overall Average AUROC: {final_avg_jac_auroc:.4f} +/- {final_std_jac_auroc:.4f}"
            )

        if "causal_graph_ap" in summary_df.columns:
            final_avg_cg_ap = summary_df["causal_graph_ap"].mean()
            final_std_cg_ap = summary_df["causal_graph_ap"].std()
            final_avg_cg_auroc = summary_df["causal_graph_auroc"].mean()
            final_std_cg_auroc = summary_df["causal_graph_auroc"].std()

            print(f"\nCausal Graph Metrics (Graph extraction-based):")
            print(
                f"Overall Average AP: {final_avg_cg_ap:.4f} +/- {final_std_cg_ap:.4f}"
            )
            print(
                f"Overall Average AUROC: {final_avg_cg_auroc:.4f} +/- {final_std_cg_auroc:.4f}"
            )

        # Save summary results
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, f"loo_summary_{MODEL_TYPE}_seed{SEED}.csv"),
            index=False,
        )

        if all_fold_metrics:
            detailed_df = pd.DataFrame(all_fold_metrics)
            detailed_df.to_csv(
                os.path.join(
                    RESULTS_DIR, f"loo_detailed_metrics_{MODEL_TYPE}_seed{SEED}.csv"
                ),
                index=False,
            )
            print(
                f"\nDetailed metrics saved to loo_detailed_metrics_{MODEL_TYPE}_seed{SEED}.csv"
            )

    else:
        print("No results generated.")

    print(f"\nLogs and results saved in: {RESULTS_DIR}")
    print(f"PCA plots saved in: {os.path.join(RESULTS_DIR, 'pca_plots')}")
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leave-One-Out Cross-Validation for GRN models"
    )

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to data directory",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=DEFAULT_DATASET_TYPE,
        choices=["Synthetic", "Curated", "Renge"],
        help="Type of dataset to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name (only used for Synthetic dataset_type)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["sf2m", "rf", "mlp_baseline"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--use_correction_mlp",
        action="store_true",
        default=DEFAULT_USE_CORRECTION_MLP,
        help="Whether to use correction MLP for SF2M",
    )

    # Training parameters
    parser.add_argument(
        "--n_steps_per_fold",
        type=int,
        default=DEFAULT_N_STEPS_PER_FOLD,
        help="Number of steps per fold",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Alpha weighting for score vs flow loss",
    )
    parser.add_argument(
        "--reg", type=float, default=DEFAULT_REG, help="Regularization for flow model"
    )
    parser.add_argument(
        "--correction_reg",
        type=float,
        default=DEFAULT_CORRECTION_REG,
        help="Regularization for correction network",
    )
    parser.add_argument(
        "--gl_reg",
        type=float,
        default=DEFAULT_GL_REG,
        help="Group Lasso regularization strength",
    )

    # Model architecture parameters
    parser.add_argument(
        "--knockout_hidden",
        type=int,
        default=DEFAULT_KNOCKOUT_HIDDEN,
        help="Knockout hidden dimension",
    )
    parser.add_argument(
        "--score_hidden",
        type=str,
        default=",".join(map(str, DEFAULT_SCORE_HIDDEN)),
        help="Score hidden dimensions (comma-separated)",
    )
    parser.add_argument(
        "--correction_hidden",
        type=str,
        default=",".join(map(str, DEFAULT_CORRECTION_HIDDEN)),
        help="Correction hidden dimensions (comma-separated)",
    )

    # Simulation parameters
    parser.add_argument(
        "--sigma", type=float, default=DEFAULT_SIGMA, help="Noise level for simulation"
    )
    parser.add_argument(
        "--n_times_sim",
        type=int,
        default=DEFAULT_N_TIMES_SIM,
        help="Number of steps for trajectory simulation",
    )

    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save results",
    )

    args = parser.parse_args()
    main(args)
