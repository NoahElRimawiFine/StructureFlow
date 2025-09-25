# leave_ko_out.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lightning.pytorch import Trainer, seed_everything
import os
import argparse

# --- Project Specific Imports ---
from src.datamodules.grn_datamodule import TrajectoryStructureDataModule
from src.models.rf_module import ReferenceFittingModule
from src.models.sf2m_module import SF2MLitModule
from src.models.components.solver import simulate_trajectory, wasserstein, mmd_squared
from src.models.components.plotting import (
    compute_global_jacobian,
    plot_auprs,
    log_causal_graph_matrices,
)


# Default configuration values (will be overridden by command line arguments)
DEFAULT_DATA_PATH = "data/"
DEFAULT_DATASET_TYPE = "Curated"
DEFAULT_DATASET = "dyn-TF"
DEFAULT_MODEL_TYPE = "rf"
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
DEFAULT_SEED = 42
DEFAULT_RESULTS_DIR = "lko_results"
DEFAULT_USE_CORRECTION_MLP = True

# Knockout indices to leave out for testing (will be set in main)
LEAVE_OUT_KO_INDICES = [1, 2, 3]  # Default values, will be adjusted in main

# --- Renge Specific Parameters ---
# DEFAULT_N_STEPS_PER_FOLD = 10000
# DEFAULT_LR = 0.0002
# DEFAULT_REG = 5e-8
# DEFAULT_ALPHA = 0.1
# DEFAULT_GL_REG = 0.02


def create_trajectory_pca_plot(
    adata, predictions, ko_name, time_point, folder_path, model_type
):
    """
    Create and save a PCA plot showing the entire trajectory with predictions overlay.

    Args:
        adata: AnnData object containing the full trajectory data
        predictions: Model's predicted final state (numpy array)
        ko_name: Name of the knockout for labeling
        time_point: The current timepoint
        folder_path: Path to save the plot
        model_type: Type of model used ("rf", "sf2m", etc.)
    """
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
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )

    # Create the plot
    plt.figure(figsize=(12, 10))

    # Plot the true trajectory points, colored by time
    scatter = plt.scatter(
        full_data_pca[:, 0],
        full_data_pca[:, 1],
        c=times,
        cmap="viridis",
        label="True trajectory",
        s=70,
    )

    # Plot the model predictions
    plt.scatter(
        pred_pca[:, 0],
        pred_pca[:, 1],
        c="salmon",
        s=120,
        marker="x",
        linewidth=2,
        label=f"{model_type} predictions",
    )

    # Add colorbar with larger font
    cbar = plt.colorbar(scatter)
    cbar.set_label("Time", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Add title and labels
    ko_label = "Wild Type" if ko_name is None else f"Knockout: {ko_name}"
    plt.title(
        f"{ko_label} - {model_type} Prediction at t={time_point}", fontweight="bold"
    )
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
    filename = (
        f"traj_{'wildtype' if ko_name is None else f'ko_{ko_name}'}_t{time_point}.png"
    )
    plt.savefig(os.path.join(folder_path, filename), dpi=200, bbox_inches="tight")
    plt.close()

    # Reset rcParams to default to avoid affecting other plots
    plt.rcParams.update(plt.rcParamsDefault)


def create_multi_ko_pca_plot(
    full_adatas,
    predictions_dict,
    ko_names,
    left_out_ko_indices,
    folder_path,
    model_type,
):
    """
    Create and save a PCA plot showing multiple KO trajectories with predictions in subplots.
    Shows the left-out knockouts vs wildtype/other knockouts.

    Args:
        full_adatas: List of AnnData objects containing the full trajectory data
        predictions_dict: Dictionary mapping ko_idx to predicted final states
        ko_names: List of knockout names
        left_out_ko_indices: List of indices of left-out knockouts
        folder_path: Path to save the plot
        model_type: Type of model used ("rf", "sf2m", etc.)
    """
    os.makedirs(folder_path, exist_ok=True)

    if len(left_out_ko_indices) < 3:
        print(
            "Not enough left-out knockout conditions to create multi-KO plot (need at least 3)"
        )
        return

    ko_indices_to_plot = left_out_ko_indices[:3]  # Show first 3 left-out KOs

    # Create figure with plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    all_data = np.vstack([adata.X for adata in full_adatas])
    pca = PCA(n_components=2)
    pca.fit(all_data)

    # Increase font sizes
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
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
    pastel_blue = "#86BFEF"

    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for i, (ax, ko_idx) in enumerate(zip(axes, ko_indices_to_plot)):
        adata = full_adatas[ko_idx]
        ko_name = ko_names[ko_idx]

        times = adata.obs["t"].values
        ko_data_pca = pca.transform(adata.X)

        if ko_idx in predictions_dict:
            predictions = predictions_dict[ko_idx]
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            pred_pca = pca.transform(predictions)
        else:
            print(f"No predictions available for left-out KO: {ko_name}")
            continue

        # Using custom darker Reds colormap for true data
        scatter = ax.scatter(
            ko_data_pca[:, 0],
            ko_data_pca[:, 1],
            c=times,
            cmap=dark_reds,
            label="True samples" if i == 0 else None,
            s=60,
        )

        # Using pastel blue for predictions
        ax.scatter(
            pred_pca[:, 0],
            pred_pca[:, 1],
            c=pastel_blue,
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

        ko_label = "Wild Type" if ko_name is None else f"Knockout {ko_name}"
        ax.set_title(ko_label, pad=10)

        ax.grid(True, alpha=0.3, linestyle="--")

        for spine in ax.spines.values():
            spine.set_visible(True)

    for ax in axes:
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    plt.tight_layout()

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(handles),
        frameon=True,
        framealpha=0.9,
    )

    # Save the figure
    filename_base = f"multi_ko_comparison_{model_type}_leftout_kos"
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
    left_out_ko_indices,
    folder_path,
    model_type,
):
    """
    Create and save a PCA plot showing multiple left-out KO trajectories with predictions in subplots.
    Uses grayscale for different time points and highlights predictions.

    Args:
        full_adatas: List of AnnData objects containing the full trajectory data
        predictions_dict: Dictionary mapping ko_idx to predicted final states
        ko_names: List of knockout names
        left_out_ko_indices: List of indices of left-out knockouts
        folder_path: Path to save the plot
        model_type: Type of model used ("rf", "sf2m", etc.)
    """
    os.makedirs(folder_path, exist_ok=True)

    if len(left_out_ko_indices) < 3:
        print(
            "Not enough left-out knockout conditions to create multi-KO plot (need at least 3)"
        )
        return

    ko_indices_to_plot = left_out_ko_indices[:3]  # Show first 3 left-out KOs

    # Create figure with plots - extra space for legend
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))

    all_data = np.vstack([adata.X for adata in full_adatas])
    pca = PCA(n_components=2)
    pca.fit(all_data)

    # Set reasonable font sizes
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 28,
            "axes.labelsize": 28,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 26,
        }
    )

    # Define colors
    highlight_color = "#E41A1C"  # Bright red for the latest time point
    prediction_color = "#377EB8"  # Blue for predictions

    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for i, (ax, ko_idx) in enumerate(zip(axes, ko_indices_to_plot)):
        adata = full_adatas[ko_idx]
        ko_name = ko_names[ko_idx]

        times = adata.obs["t"].values
        ko_data_pca = pca.transform(adata.X)

        if ko_idx in predictions_dict:
            predictions = predictions_dict[ko_idx]
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            pred_pca = pca.transform(predictions)
        else:
            print(f"No predictions available for left-out KO: {ko_name}")
            continue

        # Get the maximum time to determine which is the final time point
        max_time = max(times)

        # Plot non-final times in grayscale with darker shades for later times
        for t in sorted(set(times)):
            if t == max_time:
                continue

            t_mask = times == t
            shade = 0.3 + 0.5 * (t / max_time)  # Map to darkness between 0.3-0.8
            gray_color = str(1 - shade)  # Grayscale as string: '0.2' to '0.7'

            ax.scatter(
                ko_data_pca[t_mask, 0],
                ko_data_pca[t_mask, 1],
                c=gray_color,
                s=60,
                label=f"t={t}" if i == 0 and t == min(times) else None,
            )

        # Plot final time in highlight color
        final_time_mask = times == max_time
        if any(final_time_mask):
            ax.scatter(
                ko_data_pca[final_time_mask, 0],
                ko_data_pca[final_time_mask, 1],
                c=highlight_color,
                s=80,
                label=f"t={max_time} (final)" if i == 0 else None,
            )

        # Plot predictions
        ax.scatter(
            pred_pca[:, 0],
            pred_pca[:, 1],
            c=prediction_color,
            s=100,
            marker="x",
            linewidth=2,
            label="Predictions" if i == 0 else None,
        )

        x_min = min(x_min, ko_data_pca[:, 0].min(), pred_pca[:, 0].min())
        x_max = max(x_max, ko_data_pca[:, 0].max(), pred_pca[:, 0].max())
        y_min = min(y_min, ko_data_pca[:, 1].min(), pred_pca[:, 1].min())
        y_max = max(y_max, ko_data_pca[:, 1].max(), pred_pca[:, 1].max())

        ax.set_xlabel("PC1", fontsize=28)
        if i == 0:
            ax.set_ylabel("PC2", fontsize=28)

        ko_label = "Wild Type" if ko_name is None else f"Knockout {ko_name}"
        ax.set_title(ko_label, pad=15, fontsize=28)

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
        fontsize=26,
    )

    # Save the figure
    filename_base = f"multi_ko_comparison_{model_type}_leftout_kos"
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

    # Determine the KO indices to leave out
    total_kos = len(full_adatas)
    LEAVE_OUT_KO_INDICES = args.leave_out_ko_indices

    # Validate KO indices
    for ko_idx in LEAVE_OUT_KO_INDICES:
        if ko_idx >= total_kos:
            raise ValueError(
                f"Invalid knockout index {ko_idx}. Only {total_kos} knockouts available."
            )

    print(f"Total knockouts available: {total_kos}")
    print(f"Knockouts to leave out: {LEAVE_OUT_KO_INDICES}")
    print(
        f"Knockout names to leave out: {[datamodule.kos[idx] for idx in LEAVE_OUT_KO_INDICES]}"
    )

    T_max = int(max(adata.obs["t"].max() for adata in full_adatas))
    T_times = T_max + 1
    DT_data = 1.0 / T_times

    print(f"Data loaded. Found {len(full_adatas)} datasets with T_max={T_max}.")
    print(f"Kos: {datamodule.kos}")
    print(f"Using model type: {MODEL_TYPE}")

    # --- 2. Leave-KO-Out Cross-Validation Loop ---
    results = []
    all_ko_metrics = []

    # Dictionary to store predictions for multi-KO plots
    predictions_dict = {}

    for leave_out_idx in LEAVE_OUT_KO_INDICES:
        ko_name = datamodule.kos[leave_out_idx]
        fold_name = f"{MODEL_TYPE}_leave_out_ko_{ko_name}"
        print(f"\n===== Training Fold: {fold_name} =====")

        # --- 2.1 Create Filtered Data for this Fold (excluding the knockout) ---
        print(
            f"Creating filtered dataset excluding knockout: {ko_name} (index {leave_out_idx})..."
        )
        fold_adatas = [
            adata for i, adata in enumerate(full_adatas) if i != leave_out_idx
        ]
        fold_kos = [ko for i, ko in enumerate(datamodule.kos) if i != leave_out_idx]
        fold_ko_indices = [
            idx for i, idx in enumerate(datamodule.ko_indices) if i != leave_out_idx
        ]

        # --- 2.2 Train Model ---
        model = None

        if MODEL_TYPE == "rf":
            print("Using Reference Fitting model...")
            # Initialize RF model
            model = ReferenceFittingModule(
                use_cuda=(DEVICE == "cuda"),
                iter=(5000 if DATASET_TYPE == "Renge" else 1000),
            )

            # Fit the model with the filtered data (excluding the left-out knockout)
            print(f"Fitting RF model without knockout {ko_name}...")
            print(f"  Training KOs: {fold_kos}")
            print(f"  Total training datasets: {len(fold_adatas)}")
            model.fit_model(fold_adatas, fold_kos)

            print("RF model fitting complete.")

        else:  # "sf2m" or "mlp_baseline"
            print(f"Using {'SF2M' if MODEL_TYPE=='sf2m' else 'MLP Baseline'} model...")

            # Configure correct flags based on MODEL_TYPE
            use_mlp = MODEL_TYPE == "mlp_baseline"
            use_correction = USE_CORRECTION_MLP and not use_mlp

            # Create a modified SF2M model that leaves out the specified knockout during training
            model = SF2MLitModule(
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
                held_out_time=None,  # Not using time holdout in this script
                leave_ko_out_idx=leave_out_idx,
            )

            # Train the model with Lightning
            print("Setting up Trainer...")
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

        # --- 2.3 Evaluate on the Held-Out Knockout ---
        model.eval()
        if MODEL_TYPE != "rf":  # SF2M and MLP need explicit device placement
            model.to(DEVICE)

        print(
            f"Evaluating model on predicting trajectories for left-out knockout: {ko_name}..."
        )

        # Create folder for trajectory PCA plots
        pca_folder = os.path.join(RESULTS_DIR, "pca_plots")
        fold_pca_folder = os.path.join(pca_folder, f"{MODEL_TYPE}_ko_{ko_name}")
        os.makedirs(fold_pca_folder, exist_ok=True)

        # Get the left-out knockout data
        left_out_adata = full_adatas[leave_out_idx]

        # Simulate trajectory for each time step and compute metrics
        ko_distances = []

        with torch.no_grad():
            # For each time step, predict t to t+1
            for t in range(T_max - 1):
                # Get initial conditions (time t) and true final state (time t+1)
                x0_np = left_out_adata.X[left_out_adata.obs["t"] == t]
                # x0_np = left_out_adata.X[left_out_adata.obs["t"] == 0]
                true_dist_np = left_out_adata.X[left_out_adata.obs["t"] == t + 1]

                # Check if data exists for this transition
                if x0_np.shape[0] == 0 or true_dist_np.shape[0] == 0:
                    print(f"  Skipping time {t} to {t+1}: No data for this transition.")
                    continue

                x0 = torch.from_numpy(x0_np).float()
                true_dist_cpu = torch.from_numpy(true_dist_np).float()

                if MODEL_TYPE == "rf":
                    # Simulate trajectory using RF model
                    traj_rf = model.simulate_trajectory(
                        x0,
                        n_times=t + 1,  # Simulate for duration of 1 (from 0 to 1)
                        use_wildtype=False,  # Use the full model
                        n_points=N_TIMES_SIM,
                        ko_condition=ko_name,  # Specify which knockout to simulate
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

                    # Store predictions for multi-KO plot (using final time step predictions)
                    if t == T_max - 2:  # Last transition (t to t+1 where t+1 is final)
                        predictions_dict[leave_out_idx] = traj_rf.numpy()

                    # Create trajectory PCA plot
                    create_trajectory_pca_plot(
                        left_out_adata,
                        traj_rf.numpy(),
                        ko_name,
                        t + 1,
                        fold_pca_folder,
                        "RF",
                    )

                else:  # SF2M or MLP Baseline
                    # Move tensors to the right device
                    x0 = x0.to(DEVICE)
                    true_dist = true_dist_cpu.to(DEVICE)

                    # Get conditional vector for the left-out knockout
                    cond_vector_template = model.conditionals[leave_out_idx].to(DEVICE)
                    if (
                        cond_vector_template is not None
                        and cond_vector_template.nelement() > 0
                    ):
                        cond_vector = cond_vector_template[0].repeat(x0.shape[0], 1)
                    else:
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
                        "dataset_idx": leave_out_idx,
                        "start_time": t,
                        "end_time": t + 1,
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

                    # Store predictions for multi-KO plot (using final time step predictions)
                    if t == T_max - 2:  # Last transition (t to t+1 where t+1 is final)
                        predictions_dict[leave_out_idx] = sim_ode_final.numpy()

                    # Create trajectory PCA plot using ODE simulation
                    create_trajectory_pca_plot(
                        left_out_adata,
                        sim_ode_final.numpy(),
                        ko_name,
                        t + 1,
                        fold_pca_folder,
                        MODEL_TYPE,
                    )

                # Store metrics for this time point
                ko_distances.append(
                    {
                        "time_from": t,
                        "time_to": t + 1,
                        "w_dist_ode": w_dist_ode,
                        "w_dist_sde": w_dist_sde,
                        "mmd2_ode": mmd2_ode,
                        "mmd2_sde": mmd2_sde,
                    }
                )

                # Log metrics
                print(
                    f"  Time {t} to {t+1}: W_dist(ODE)={w_dist_ode:.4f}, MMD2(ODE)={mmd2_ode:.4f}"
                )
                if MODEL_TYPE != "rf":
                    print(
                        f"  Time {t} to {t+1}: W_dist(SDE)={w_dist_sde:.4f}, MMD2(SDE)={mmd2_sde:.4f}"
                    )

        # --- 2.4 Aggregate Results for this Knockout ---
        if ko_distances:
            ko_df = pd.DataFrame(ko_distances)
            avg_ode_dist = ko_df["w_dist_ode"].mean()
            avg_sde_dist = ko_df["w_dist_sde"].mean()
            avg_mmd2_ode = ko_df["mmd2_ode"].mean()
            avg_mmd2_sde = ko_df["mmd2_sde"].mean()

            std_ode_dist = ko_df["w_dist_ode"].std()
            std_sde_dist = ko_df["w_dist_sde"].std()
            std_mmd2_ode = ko_df["mmd2_ode"].std()
            std_mmd2_sde = ko_df["mmd2_sde"].std()

            print(
                f"Knockout {ko_name} Avg W_dist: ODE={avg_ode_dist:.4f} ± {std_ode_dist:.4f}"
            )
            if MODEL_TYPE != "rf":
                print(
                    f"Knockout {ko_name} Avg W_dist: SDE={avg_sde_dist:.4f} ± {std_sde_dist:.4f}"
                )
                print(
                    f"Knockout {ko_name} Avg MMD2: ODE={avg_mmd2_ode:.4f} ± {std_mmd2_ode:.4f}"
                )
                print(
                    f"Knockout {ko_name} Avg MMD2: SDE={avg_mmd2_sde:.4f} ± {std_mmd2_sde:.4f}"
                )

            results.append(
                {
                    "ko_index": leave_out_idx,
                    "ko_name": ko_name,
                    "model_type": MODEL_TYPE,
                    "avg_ode_distance": avg_ode_dist,
                    "avg_sde_distance": avg_sde_dist,
                    "avg_mmd2_ode": avg_mmd2_ode,
                    "avg_mmd2_sde": avg_mmd2_sde,
                    "std_ode_distance": std_ode_dist,
                    "std_sde_distance": std_sde_dist,
                    "std_mmd2_ode": std_mmd2_ode,
                    "std_mmd2_sde": std_mmd2_sde,
                }
            )

            # Add detailed metrics to global list
            for record in ko_distances:
                record["ko_index"] = leave_out_idx
                record["ko_name"] = ko_name
                record["model_type"] = MODEL_TYPE
                record["seed"] = SEED
                record["dataset_type"] = DATASET_TYPE
                all_ko_metrics.append(record)

            # Save detailed metrics for this knockout
            ko_detail_df = pd.DataFrame(ko_distances)
            ko_detail_df.to_csv(
                os.path.join(RESULTS_DIR, f"ko_{ko_name}_detailed_metrics.csv"),
                index=False,
            )
            print(
                f"Detailed metrics for KO {ko_name} saved to ko_{ko_name}_detailed_metrics.csv"
            )

        else:
            print(f"Knockout {ko_name}: No evaluation results.")
            results.append(
                {
                    "ko_index": leave_out_idx,
                    "ko_name": ko_name,
                    "model_type": MODEL_TYPE,
                    "avg_ode_distance": np.nan,
                    "avg_sde_distance": np.nan,
                    "avg_mmd2_ode": np.nan,
                    "avg_mmd2_sde": np.nan,
                    "std_ode_distance": np.nan,
                    "std_sde_distance": np.nan,
                    "std_mmd2_ode": np.nan,
                    "std_mmd2_sde": np.nan,
                }
            )

    # --- 2.5 Create Multi-KO Comparison Plots ---
    if predictions_dict and len(LEAVE_OUT_KO_INDICES) >= 3:
        print(
            f"\nCreating multi-KO comparison plots with {len(predictions_dict)} left-out knockouts..."
        )

        # Create folder for multi-KO plots
        multi_ko_pca_folder = os.path.join(
            RESULTS_DIR, "pca_plots", "multi_ko_comparison"
        )
        os.makedirs(multi_ko_pca_folder, exist_ok=True)

        # Create the multi-KO plots
        create_multi_ko_pca_plot_wgrey(
            full_adatas,
            predictions_dict,
            datamodule.kos,
            LEAVE_OUT_KO_INDICES,
            multi_ko_pca_folder,
            MODEL_TYPE,
        )
        print(f"Multi-KO comparison plots saved to: {multi_ko_pca_folder}")
    elif len(LEAVE_OUT_KO_INDICES) < 3:
        print(
            "Not enough left-out knockouts to create multi-KO comparison plots (need at least 3)"
        )
    else:
        print("No predictions available for multi-KO comparison plots")

    # --- 3. Final Reporting ---
    print(f"\n===== Leave-KO-Out Cross-Validation Summary ({MODEL_TYPE}) =====")
    if results:
        summary_df = pd.DataFrame(results)
        summary_df["seed"] = SEED
        summary_df["dataset_type"] = DATASET_TYPE

        print("Average Metrics per Knockout:")
        print(summary_df.to_string(index=False))

        # Calculate overall averages
        final_avg_ode = summary_df["avg_ode_distance"].mean()
        final_avg_sde = summary_df["avg_sde_distance"].mean()
        final_avg_mmd2_ode = summary_df["avg_mmd2_ode"].mean()
        final_avg_mmd2_sde = summary_df["avg_mmd2_sde"].mean()

        final_std_ode = np.sqrt(np.mean(summary_df["std_ode_distance"] ** 2))
        final_std_sde = np.sqrt(np.mean(summary_df["std_sde_distance"] ** 2))
        final_std_mmd2_ode = np.sqrt(np.mean(summary_df["std_mmd2_ode"] ** 2))
        final_std_mmd2_sde = np.sqrt(np.mean(summary_df["std_mmd2_sde"] ** 2))

        print(
            f"\nOverall Average W-Distance (ODE): {final_avg_ode:.4f} ± {final_std_ode:.4f}"
        )
        print(
            f"Overall Average W-Distance (SDE): {final_avg_sde:.4f} ± {final_std_sde:.4f}"
        )
        print(
            f"Overall Average MMD2 (ODE): {final_avg_mmd2_ode:.4f} ± {final_std_mmd2_ode:.4f}"
        )
        print(
            f"Overall Average MMD2 (SDE): {final_avg_mmd2_sde:.4f} ± {final_std_mmd2_sde:.4f}"
        )

        # Save summary results
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, f"lko_summary_{MODEL_TYPE}_seed{SEED}.csv"),
            index=False,
        )

        if all_ko_metrics:
            detailed_df = pd.DataFrame(all_ko_metrics)
            detailed_df.to_csv(
                os.path.join(
                    RESULTS_DIR, f"lko_detailed_metrics_{MODEL_TYPE}_seed{SEED}.csv"
                ),
                index=False,
            )
            print(
                f"\nDetailed metrics saved to lko_detailed_metrics_{MODEL_TYPE}_seed{SEED}.csv"
            )

    else:
        print("No results generated.")

    print(f"\nLogs and results saved in: {RESULTS_DIR}")
    print(f"Individual PCA plots saved in: {os.path.join(RESULTS_DIR, 'pca_plots')}")
    if predictions_dict and len(LEAVE_OUT_KO_INDICES) >= 3:
        print(
            f"Multi-KO comparison plots saved in: {os.path.join(RESULTS_DIR, 'pca_plots', 'multi_ko_comparison')}"
        )
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leave-KO-Out Cross-Validation for GRN models"
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

    # Leave KO parameters
    parser.add_argument(
        "--leave_out_ko_indices",
        type=int,
        nargs="+",
        default=LEAVE_OUT_KO_INDICES,
        help="Indices of knockouts to leave out (space-separated list)",
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
