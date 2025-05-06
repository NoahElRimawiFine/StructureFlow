# leave_one_ko_out.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lightning.pytorch import Trainer, seed_everything
import os
import argparse

# --- Project Specific Imports ---
from src.datamodules.grn_datamodule import AnnDataDataset, TrajectoryStructureDataModule
from src.models.rf_module import ReferenceFittingModule
from src.models.sf2m_module import SF2MLitModule
from src.models.components.solver import mmd_squared, simulate_trajectory, wasserstein

# Default configuration values
DEFAULT_DATA_PATH = "data/"
DEFAULT_DATASET_TYPE = "Synthetic"
DEFAULT_MODEL_TYPE = "rf"
DEFAULT_N_STEPS_PER_KO = 1000
DEFAULT_BATCH_SIZE = 128
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
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42
DEFAULT_RESULTS_DIR = "loko_results"
DEFAULT_USE_CORRECTION_MLP = True


def create_trajectory_pca_plot(adata, predictions, ko_name, timepoint, folder_path, model_type):
    """
    Create and save a PCA plot showing the entire trajectory with predictions overlay.
    
    Args:
        adata: AnnData object containing the full trajectory data
        predictions: Model's predicted final state (numpy array)
        ko_name: Name of the knockout for labeling
        timepoint: The timepoint being predicted
        folder_path: Path to save the plot
        model_type: Type of model used ("rf", "sf2m", etc.)
    """
    # Create component folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Get all data and times from the anndata object
    full_data = adata.X
    times = adata.obs['t'].values
    
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
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot the true trajectory points, colored by time
    scatter = plt.scatter(
        full_data_pca[:, 0], 
        full_data_pca[:, 1], 
        c=times, 
        cmap='viridis', 
        label="True trajectory",
        s=70
    )
    
    # Plot the model predictions
    plt.scatter(
        pred_pca[:, 0], 
        pred_pca[:, 1], 
        c='salmon', 
        s=120,
        marker='x',
        linewidth=2,
        label=f"{model_type} predictions"
    )
    
    # Add colorbar with larger font
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Add title and labels
    plt.title(f"Knockout: {ko_name} (t={timepoint}) - {model_type}", fontweight='bold')
    plt.xlabel("PC1", fontweight='bold')
    plt.ylabel("PC2", fontweight='bold')
    
    # Larger and better positioned legend
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=14)
    
    # Clearer grid
    plt.grid(True, alpha=0.4, linestyle='--')
    
    # Add a border around the plot
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
    # Save the figure
    filename = f"ko_{ko_name}_t{timepoint}.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Reset rcParams to default to avoid affecting other plots
    plt.rcParams.update(plt.rcParamsDefault)


def main(args):
    # Extract configuration from arguments
    DATA_PATH = args.data_path
    DATASET_TYPE = args.dataset_type
    N_STEPS_PER_KO = args.n_steps_per_ko
    BATCH_SIZE = args.batch_size
    LR = args.lr
    ALPHA = args.alpha
    REG = args.reg
    CORRECTION_REG = args.correction_reg
    GL_REG = args.gl_reg
    KNOCKOUT_HIDDEN = args.knockout_hidden
    SCORE_HIDDEN = [int(x) for x in args.score_hidden.split(',')]
    CORRECTION_HIDDEN = [int(x) for x in args.correction_hidden.split(',')]
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
        f"{DATASET_TYPE}_{MODEL_TYPE}_seed{SEED}"
    )
    
    seed_everything(SEED, workers=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Load Full Data Once ---
    print("Loading full dataset...")
    datamodule = TrajectoryStructureDataModule(
        data_path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        batch_size=BATCH_SIZE,
        use_dummy_train_loader=True,
        dummy_loader_steps=N_STEPS_PER_KO,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    full_adatas = datamodule.get_subset_adatas()
    if not full_adatas:
        raise ValueError("No datasets loaded.")
    T_max = int(max(adata.obs['t'].max() for adata in full_adatas))
    T_times = T_max + 1
    DT_data = 1.0 / T_times

    print(f"Data loaded. Found {len(full_adatas)} datasets with T_max={T_max}.")
    print(f"Kos: {datamodule.kos}")
    print(f"Using model type: {MODEL_TYPE}")

    # --- 2. Leave-One-KO-Out Cross-Validation Loop ---
    results = []
    ko_metrics = []
    
    # Create PCA plots directory
    pca_folder = os.path.join(RESULTS_DIR, "pca_plots")
    os.makedirs(pca_folder, exist_ok=True)

    for held_out_ko_idx, held_out_ko_name in enumerate(datamodule.kos):
        fold_name = f"{MODEL_TYPE}_holdout_ko_{held_out_ko_name}"
        print(f"\n===== Training Fold: {fold_name} =====")

        # --- 2.1 Create Filtered Dataset for this Fold ---
        print(f"Creating filtered dataset excluding knockout: {held_out_ko_name}...")
        fold_adatas = []
        fold_kos = []
        fold_ko_indices = []
        
        for i, (adata, ko, ko_idx) in enumerate(zip(full_adatas, datamodule.kos, datamodule.ko_indices)):
            if i != held_out_ko_idx:
                fold_adatas.append(adata)
                fold_kos.append(ko)
                fold_ko_indices.append(ko_idx)

        # --- 2.2 Create a Temporary DataModule ---
        temp_datamodule = TrajectoryStructureDataModule(
            data_path=DATA_PATH,
            dataset_type=DATASET_TYPE,
            batch_size=BATCH_SIZE,
            use_dummy_train_loader=True,
            dummy_loader_steps=N_STEPS_PER_KO,
            num_workers=0,
        )
        
        # Override with our filtered data
        temp_datamodule.adatas = fold_adatas
        temp_datamodule.kos = fold_kos
        temp_datamodule.ko_indices = fold_ko_indices
        temp_datamodule.true_matrix = datamodule.true_matrix
        temp_datamodule.dim = datamodule.dim
        temp_datamodule.gene_to_index = datamodule.gene_to_index
        
        # Create datasets for training
        wrapped_datasets = [
            AnnDataDataset(adata, source_id=i) for i, adata in enumerate(fold_adatas)
        ]
        temp_datamodule._dataset_lengths = [len(ds) for ds in wrapped_datasets]
        temp_datamodule._full_dataset = torch.utils.data.ConcatDataset(wrapped_datasets)
        
        # Set up train/val/test splits
        full_len = len(temp_datamodule._full_dataset)
        train_len = int(full_len * 0.8)  # Use same splits as the original datamodule
        val_len = int(full_len * 0.1)
        test_len = full_len - train_len - val_len
        temp_datamodule.dataset_train, temp_datamodule.dataset_val, temp_datamodule.dataset_test = \
            torch.utils.data.random_split(temp_datamodule._full_dataset, [train_len, val_len, test_len])
        
        # --- 2.3 Train Model ---
        model = None
        
        if MODEL_TYPE == "rf":
            print("Using Reference Fitting model...")
            # Initialize RF model
            model = ReferenceFittingModule(use_cuda=(DEVICE == "cuda"), iter=N_STEPS_PER_KO)
            
            # Fit the model without the held-out knockout
            print(f"Fitting RF model without knockout {held_out_ko_name}...")
            model.fit_model(fold_adatas, fold_kos)
            
            print("RF model fitting complete.")

        else:  # "sf2m" or "mlp_baseline"
            print(f"Using {'SF2M' if MODEL_TYPE=='sf2m' else 'MLP Baseline'} model...")
            
            # Configure correct flags based on MODEL_TYPE
            use_mlp = MODEL_TYPE == "mlp_baseline"
            use_correction = USE_CORRECTION_MLP and not use_mlp
            
            model = SF2MLitModule(
                datamodule=temp_datamodule,
                T=T_times,
                sigma=SIGMA,
                dt=DT_data,
                batch_size=BATCH_SIZE,
                alpha=ALPHA,
                reg=REG,
                correction_reg_strength=CORRECTION_REG,
                n_steps=N_STEPS_PER_KO,
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
            trainer = Trainer(
                max_epochs=-1,
                max_steps=N_STEPS_PER_KO,
                accelerator="cpu" if DEVICE == "cpu" else "gpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
                log_every_n_steps=N_STEPS_PER_KO + 1,
                num_sanity_val_steps=0,
                limit_val_batches=0.0,
            )
            
            print(f"Training model for {N_STEPS_PER_KO} steps...")
            trainer.fit(model, datamodule=temp_datamodule)
            print("Training complete.")

        # --- 2.4 Evaluate Model on Held-Out Knockout ---
        print(f"Evaluating model on held-out knockout: {held_out_ko_name}...")
        model.eval()
        if MODEL_TYPE != "rf":  # SF2M and MLP need explicit device placement
            model.to(DEVICE)
        
        # Get the held-out knockout's data
        held_out_adata = full_adatas[held_out_ko_idx]
        held_out_ko_idx_value = datamodule.ko_indices[held_out_ko_idx]
        
        # For each timepoint (except t=0), predict from t-1 to t
        timepoint_results = []
        fold_pca_folder = os.path.join(pca_folder, f"{MODEL_TYPE}_ko_{held_out_ko_name}")
        os.makedirs(fold_pca_folder, exist_ok=True)
        
        with torch.no_grad():
            for t in range(1, T_times):
                print(f"  Evaluating on timepoint t={t}...")
                
                # Get initial conditions (t-1) and true final state (t)
                x0_np = held_out_adata.X[held_out_adata.obs["t"] == t - 1]
                true_dist_np = held_out_adata.X[held_out_adata.obs["t"] == t]

                # Check if data exists for this transition
                if x0_np.shape[0] == 0 or true_dist_np.shape[0] == 0:
                    print(f"  Skipping timepoint t={t}: No data for t={t-1} -> t={t}.")
                    continue

                x0 = torch.from_numpy(x0_np).float()
                true_dist_cpu = torch.from_numpy(true_dist_np).float()
                
                if MODEL_TYPE == "rf":
                    # For RF, we use the model directly
                    # RF doesn't explicitly encode knockout information in the model structure,
                    # it learns the dynamics from the data
                    traj_rf = model.simulate_trajectory(
                        x0, 
                        n_times=1,  # Simulate one time step
                        use_wildtype=False,  # Use full model
                        n_points=N_TIMES_SIM
                    )
                    
                    # RF simulation result
                    sim_rf_final = traj_rf[-1].cpu()
                    
                    # Calculate metrics
                    w_dist_rf = wasserstein(sim_rf_final, true_dist_cpu)
                    mmd2_rf = mmd_squared(sim_rf_final, true_dist_cpu)
                    
                    # Use same values for both ODE and SDE metrics for consistent formatting
                    w_dist_ode = w_dist_rf
                    w_dist_sde = w_dist_rf
                    mmd2_ode = mmd2_rf
                    mmd2_sde = mmd2_rf
                    
                    # Create trajectory PCA plot
                    create_trajectory_pca_plot(
                        held_out_adata,
                        sim_rf_final.numpy(),
                        held_out_ko_name,
                        t,
                        fold_pca_folder,
                        "RF"
                    )
                    
                else:  # SF2M or MLP Baseline
                    # Move tensors to the right device
                    x0 = x0.to(DEVICE)
                    true_dist = true_dist_cpu.to(DEVICE)
                    
                    # For SF2M, we need to create a new conditional vector for the held-out knockout
                    # SF2M model expects a one-hot encoding for the knockout genes
                    ko_dim = model.n_genes  # Number of genes
                    
                    # Create a one-hot vector for the knockout
                    if held_out_ko_idx_value is not None:
                        ko_vector = torch.zeros(1, ko_dim, device=DEVICE)
                        ko_vector[0, held_out_ko_idx_value] = 1.0
                    else:
                        # For wildtype, all zeros
                        ko_vector = torch.zeros(1, ko_dim, device=DEVICE)
                    
                    # Repeat for batch size
                    ko_vector = ko_vector.repeat(x0.shape[0], 1)

                    # Common simulation arguments
                    common_sim_args = {
                        "flow_model": model.func_v,
                        "corr_model": model.v_correction,
                        "score_model": model.score_net,
                        "x0": x0,
                        "dataset_idx": 0,  # This doesn't matter for simulation
                        "start_time": t - 1,
                        "end_time": t,
                        "n_times": N_TIMES_SIM,
                        "cond_vector": ko_vector,  # Use our custom knockout vector
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

                    # Create trajectory PCA plot
                    create_trajectory_pca_plot(
                        held_out_adata,
                        sim_ode_final.numpy(),
                        held_out_ko_name,
                        t,
                        fold_pca_folder,
                        MODEL_TYPE
                    )

                # Store metrics for this timepoint
                timepoint_metrics = {
                    "timepoint": t,
                    "w_dist_ode": w_dist_ode,
                    "w_dist_sde": w_dist_sde,
                    "mmd2_ode": mmd2_ode,
                    "mmd2_sde": mmd2_sde,
                }
                
                timepoint_results.append(timepoint_metrics)
                
                # Log metrics for this timepoint
                print(f"  Timepoint t={t}: W_dist(ODE)={w_dist_ode:.4f}, MMD2(ODE)={mmd2_ode:.4f}")
                if MODEL_TYPE != "rf":
                    print(f"  Timepoint t={t}: W_dist(SDE)={w_dist_sde:.4f}, MMD2(SDE)={mmd2_sde:.4f}")
        
        # Calculate average metrics across all timepoints for this knockout
        if timepoint_results:
            ko_timepoints_df = pd.DataFrame(timepoint_results)
            avg_ode_dist = ko_timepoints_df["w_dist_ode"].mean()
            avg_sde_dist = ko_timepoints_df["w_dist_sde"].mean()
            avg_mmd2_ode = ko_timepoints_df["mmd2_ode"].mean()
            avg_mmd2_sde = ko_timepoints_df["mmd2_sde"].mean()
            
            print(f"KO {held_out_ko_name} Avg W_dist: ODE={avg_ode_dist:.4f}")
            if MODEL_TYPE != "rf":
                print(f"KO {held_out_ko_name} Avg W_dist: SDE={avg_sde_dist:.4f}")
                
            # Store results for this knockout
            ko_result = {
                "held_out_ko": held_out_ko_name,
                "held_out_ko_idx": held_out_ko_idx,
                "model_type": MODEL_TYPE,
                "avg_ode_distance": avg_ode_dist,
                "avg_sde_distance": avg_sde_dist,
                "avg_mmd2_ode": avg_mmd2_ode,
                "avg_mmd2_sde": avg_mmd2_sde,
            }
            results.append(ko_result)
            
            # Store detailed metrics for each timepoint
            for metric in timepoint_results:
                metric_with_ko = metric.copy()
                metric_with_ko["held_out_ko"] = held_out_ko_name
                metric_with_ko["held_out_ko_idx"] = held_out_ko_idx
                metric_with_ko["model_type"] = MODEL_TYPE
                metric_with_ko["seed"] = SEED
                metric_with_ko["dataset_type"] = DATASET_TYPE
                ko_metrics.append(metric_with_ko)
                
            # Save the timepoint results for this knockout
            ko_timepoints_df.to_csv(os.path.join(RESULTS_DIR, f"ko_{held_out_ko_name}_timepoints_{MODEL_TYPE}.csv"), index=False)
        else:
            print(f"No evaluation results for knockout: {held_out_ko_name}")
            results.append({
                "held_out_ko": held_out_ko_name,
                "held_out_ko_idx": held_out_ko_idx,
                "model_type": MODEL_TYPE,
                "avg_ode_distance": np.nan,
                "avg_sde_distance": np.nan,
                "avg_mmd2_ode": np.nan,
                "avg_mmd2_sde": np.nan,
            })

    # --- 3. Final Reporting and Result Storage ---
    print(f"\n===== Leave-One-KO-Out Cross-Validation Summary ({MODEL_TYPE}) =====")
    if results:
        # Create summary dataframe of knockout-level results
        summary_df = pd.DataFrame(results)
        summary_df['seed'] = SEED
        summary_df['dataset_type'] = DATASET_TYPE
        
        print("Average Metrics per Knockout:")
        print(summary_df.to_string(index=False))

        # Calculate overall averages across all knockouts
        final_avg_ode = summary_df["avg_ode_distance"].mean()
        final_avg_sde = summary_df["avg_sde_distance"].mean()
        final_avg_mmd2_ode = summary_df["avg_mmd2_ode"].mean()
        final_avg_mmd2_sde = summary_df["avg_mmd2_sde"].mean()
        
        final_std_ode = summary_df["avg_ode_distance"].std()
        final_std_sde = summary_df["avg_sde_distance"].std()
        final_std_mmd2_ode = summary_df["avg_mmd2_ode"].std()
        final_std_mmd2_sde = summary_df["avg_mmd2_sde"].std()

        print(f"\nOverall Average W-Distance (ODE): {final_avg_ode:.4f} +/- {final_std_ode:.4f}")
        print(f"Overall Average W-Distance (SDE): {final_avg_sde:.4f} +/- {final_std_sde:.4f}")
        print(f"Overall Average MMD2 (ODE): {final_avg_mmd2_ode:.4f} +/- {final_std_mmd2_ode:.4f}")
        print(f"Overall Average MMD2 (SDE): {final_avg_mmd2_sde:.4f} +/- {final_std_mmd2_sde:.4f}")

        # Save summary results
        summary_df.to_csv(os.path.join(RESULTS_DIR, f"loko_summary_{MODEL_TYPE}_seed{SEED}.csv"), index=False)

        # Save detailed metrics if we have them
        if ko_metrics:
            detailed_df = pd.DataFrame(ko_metrics)
            detailed_df.to_csv(os.path.join(RESULTS_DIR, f"loko_detailed_metrics_{MODEL_TYPE}_seed{SEED}.csv"), index=False)
            print(f"\nDetailed metrics saved to loko_detailed_metrics_{MODEL_TYPE}_seed{SEED}.csv")

    else:
        print("No results generated.")

    print(f"\nLogs and results saved in: {RESULTS_DIR}")
    print(f"PCA plots saved in: {os.path.join(RESULTS_DIR, 'pca_plots')}")
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-One-KO-Out Cross-Validation for GRN models")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to data directory")
    parser.add_argument("--dataset_type", type=str, default=DEFAULT_DATASET_TYPE, choices=["Synthetic", "Curated", "Renge"], help="Type of dataset to use")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["sf2m", "rf", "mlp_baseline"], help="Type of model to use")
    parser.add_argument("--use_correction_mlp", action="store_true", default=DEFAULT_USE_CORRECTION_MLP, help="Whether to use correction MLP for SF2M")
    
    # Training parameters
    parser.add_argument("--n_steps_per_ko", type=int, default=DEFAULT_N_STEPS_PER_KO, help="Number of steps per knockout fold")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Alpha weighting for score vs flow loss")
    parser.add_argument("--reg", type=float, default=DEFAULT_REG, help="Regularization for flow model")
    parser.add_argument("--correction_reg", type=float, default=DEFAULT_CORRECTION_REG, help="Regularization for correction network")
    parser.add_argument("--gl_reg", type=float, default=DEFAULT_GL_REG, help="Group Lasso regularization strength")
    
    # Model architecture parameters
    parser.add_argument("--knockout_hidden", type=int, default=DEFAULT_KNOCKOUT_HIDDEN, help="Knockout hidden dimension")
    parser.add_argument("--score_hidden", type=str, default=",".join(map(str, DEFAULT_SCORE_HIDDEN)), help="Score hidden dimensions (comma-separated)")
    parser.add_argument("--correction_hidden", type=str, default=",".join(map(str, DEFAULT_CORRECTION_HIDDEN)), help="Correction hidden dimensions (comma-separated)")
    
    # Simulation parameters
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Noise level for simulation")
    parser.add_argument("--n_times_sim", type=int, default=DEFAULT_N_TIMES_SIM, help="Number of steps for trajectory simulation")
    
    # Other parameters
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to save results")
    
    args = parser.parse_args()
    main(args)