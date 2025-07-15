import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import matplotlib.pyplot as plt

from src.datamodules.grn_datamodule import TrajectoryStructureDataModule
from src.models.sf2m_module import SF2MLitModule
from src.models.rf_module import ReferenceFittingModule
from src.models.components.plotting import (
    compute_global_jacobian,
    plot_auprs,
    log_causal_graph_matrices,
)

# Default parameters
DEFAULT_DATA_PATH = "data/"
DEFAULT_DATASET_TYPE = "Synthetic"
DEFAULT_MODEL_TYPE = "sf2m"
DEFAULT_N_STEPS = 15000
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
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 1
DEFAULT_RESULTS_DIR = "results"
DEFAULT_USE_CORRECTION_MLP = True

# --- Renge Specific Parameters ---
DEFAULT_N_STEPS = 10000
DEFAULT_LR = 0.0002
DEFAULT_REG = 5e-8
DEFAULT_ALPHA = 0.1
DEFAULT_GL_REG = 0.02


def main(args):
    # Extract configuration from arguments
    DATA_PATH = args.data_path
    DATASET_TYPE = args.dataset_type
    N_STEPS = args.n_steps
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
    DEVICE = args.device
    SEED = args.seed
    RESULTS_DIR = args.results_dir
    MODEL_TYPE = args.model_type
    USE_CORRECTION_MLP = args.use_correction_mlp

    # Create results directory with model type and seed info
    RESULTS_DIR = os.path.join(RESULTS_DIR, f"{DATASET_TYPE}_{MODEL_TYPE}_seed{SEED}")

    seed_everything(SEED, workers=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Load Data ---
    print("Loading dataset...")
    datamodule = TrajectoryStructureDataModule(
        data_path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        batch_size=BATCH_SIZE,
        use_dummy_train_loader=True,
        dummy_loader_steps=N_STEPS,
        num_workers=20,
        train_val_test_split=(1, 0, 0),
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
    print(f"Kos: {datamodule.kos}")
    print(f"Using model type: {MODEL_TYPE}")

    # --- 2. Create and Train Model ---
    print(f"Creating model...")

    model = None

    if MODEL_TYPE == "rf":
        print("Using Reference Fitting model...")
        # Initialize RF model
        model = ReferenceFittingModule(use_cuda=(DEVICE == "cuda"))

        # Fit the model
        print("Fitting RF model...")
        model.fit_model(full_adatas, datamodule.kos)

        # No Lightning Trainer used for RF
        print("RF model fitting complete.")

    else:  # "sf2m" or "mlp_baseline"
        # Configure correct flags based on MODEL_TYPE
        use_mlp = MODEL_TYPE == "mlp_baseline"
        use_correction = USE_CORRECTION_MLP and not use_mlp

        model = SF2MLitModule(
            datamodule=datamodule,
            T=T_times,
            sigma=SIGMA,
            dt=DT_data,
            batch_size=BATCH_SIZE,
            alpha=ALPHA,
            reg=REG,
            correction_reg_strength=CORRECTION_REG,
            n_steps=N_STEPS,
            lr=LR,
            device=DEVICE,
            GL_reg=GL_REG,
            knockout_hidden=KNOCKOUT_HIDDEN,
            score_hidden=SCORE_HIDDEN,
            correction_hidden=CORRECTION_HIDDEN,
            enable_epoch_end_hook=False,
            use_mlp_baseline=use_mlp,
            use_correction_mlp=use_correction,
        )

        # Train the model with Lightning
        print("Setting up Trainer...")
        # logger = TensorBoardLogger(RESULTS_DIR, name="grn_training", default_hp_metric=False)
        trainer = Trainer(
            max_epochs=-1,
            max_steps=N_STEPS,
            accelerator="cpu" if DEVICE == "cpu" else "gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=100,
        )

        print(f"Training model for {N_STEPS} steps...")
        trainer.fit(model, datamodule=datamodule)
        print("Training complete.")

    # --- 3. Evaluate and Plot GRN ---
    model.eval()

    if MODEL_TYPE != "rf":
        # Compute the global Jacobian for SF2M models
        with torch.no_grad():
            A_estim = compute_global_jacobian(
                model.func_v, model.adatas, dt=DT_data, device=DEVICE
            )

        # Get the causal graph from the model
        W_v = model.func_v.causal_graph(w_threshold=0.0).T
    else:
        # For RF model, get the adjacency matrix directly
        W_v = model.get_interaction_matrix().detach().cpu().numpy()
        A_estim = W_v.copy()  # For RF, they are the same

    # Get the ground truth matrix
    true_matrix = datamodule.true_matrix

    # Special handling for Renge dataset to align gene sets
    if DATASET_TYPE == "Renge":
        # A_estim = A_estim.T
        # W_v = W_v.T
        # Get gene names from the dataset
        gene_names = datamodule.adatas[0].var_names

        # Get reference network rows and columns
        ref_rows = true_matrix.index
        ref_cols = true_matrix.columns

        # Create DataFrames for the estimated graphs with all genes
        A_estim_df = pd.DataFrame(A_estim, index=gene_names, columns=gene_names)
        W_v_df = pd.DataFrame(W_v, index=gene_names, columns=gene_names)

        # Extract the exact subset that corresponds to the reference network dimensions
        A_estim_subset = A_estim_df.loc[ref_rows, ref_cols]
        W_v_subset = W_v_df.loc[ref_rows, ref_cols]

        # Convert to numpy arrays for evaluation
        A_estim = A_estim_subset.values
        W_v = W_v_subset.values
        A_true = true_matrix.values
    else:
        # For synthetic data, use matrices directly
        A_true = true_matrix.values

    plot_auprs(
        W_v, A_estim, A_true, mask_diagonal=True if DATASET_TYPE != "Renge" else False
    )
    log_causal_graph_matrices(
        A_estim, W_v, A_true, mask_diagonal=True if DATASET_TYPE != "Renge" else False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRN inference from single-cell data")

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

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["sf2m", "mlp_baseline", "rf"],
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
        "--n_steps", type=int, default=DEFAULT_N_STEPS, help="Number of training steps"
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
