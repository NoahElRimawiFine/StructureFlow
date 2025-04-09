# leave_one_out.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import os
import copy
import anndata as ad # To handle AnnData objects

# --- Project Specific Imports ---
# Adjust these paths based on your project structure
from src.datamodules.grn_datamodule import TrajectoryStructureDataModule # Or your DataLoader equivalent
from src.models.rf_module import ReferenceFittingModule
from src.models.sf2m_module import SF2MLitModule
from src.models.components.solver import mmd_squared, simulate_trajectory, wasserstein
# from src.models.components.plotting import ... # Import if needed for plotting

# --- Configuration ---
# Match these with the relevant hyperparameters from SF2MLitModule and your training setup
DATA_PATH = "data/" # Or the path your DataModule expects
DATASET_TYPE = "Synthetic"   
N_EPOCHS = 50 # Or the type your DataModule expects
N_STEPS_PER_FOLD = 1000      # Training steps for each fold
BATCH_SIZE = 64              # Training batch size
LR = 3e-3
ALPHA = 0.1                   # Weighting for score vs flow loss
REG = 5e-6                    # Regularization for flow model (L2 + Group Lasso)
CORRECTION_REG = 1e-3         # Regularization for correction network
GL_REG = 0.04                 # Group Lasso specific strength
KNOCKOUT_HIDDEN = 100
SCORE_HIDDEN = [100, 100]
CORRECTION_HIDDEN = [64, 64]
SIGMA = 1.0                   # Noise level for OTFM and SDE simulation
N_TIMES_SIM = 100             # Number of steps for trajectory simulation
DEVICE = "cpu"
SEED = 42
RESULTS_DIR = "loo_results"   # Directory to save logs and potentially plots
MODEL_TYPE = "sf2m" # "sf2m" or "rf" or "mlp_baseline"
USE_CORRECTION_MLP = True


def create_trajectory_pca_plot(adata, predictions, ko_name, held_out_time, folder_path, model_type):
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
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the true trajectory points, colored by time
    scatter = plt.scatter(
        full_data_pca[:, 0], 
        full_data_pca[:, 1], 
        c=times, 
        cmap='viridis', 
        label="True trajectory"
    )
    
    # Plot the model predictions
    plt.scatter(
        pred_pca[:, 0], 
        pred_pca[:, 1], 
        c='salmon', 
        s=80,  # Make slightly larger
        marker='x',
        label=f"{model_type} predictions"
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time')
    
    # Add title and labels
    ko_label = "Wild Type" if ko_name is None else f"Knockout: {ko_name}"
    plt.title(f"{ko_label} - {model_type} Prediction")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    filename = f"traj_{'wildtype' if ko_name is None else f'ko_{ko_name}'}.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    seed_everything(SEED, workers=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Load Full Data Once ---
    print("Loading full dataset...")
    datamodule = TrajectoryStructureDataModule(
        data_path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        batch_size=BATCH_SIZE,
        use_dummy_train_loader=True,
        dummy_loader_steps=N_STEPS_PER_FOLD,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    full_adatas = datamodule.get_subset_adatas()
    if not full_adatas:
        raise ValueError("No datasets loaded.")
    T_max = int(max(adata.obs['t'].max() for adata in full_adatas))
    T_times = T_max + 1
    DT_data = 1.0 / T_max if T_max > 0 else 1.0

    print(f"Data loaded. Found {len(full_adatas)} datasets with T_max={T_max}.")
    print(f"Kos: {datamodule.kos}")
    print(f"Using model type: {MODEL_TYPE}")

    # --- 2. Leave-One-Out Cross-Validation Loop ---
    results = []
    all_fold_metrics = []
    timepoints_to_hold_out = range(1, T_times)

    for held_out_time in timepoints_to_hold_out:
        fold_name = f"{MODEL_TYPE}_holdout_{held_out_time}"
        print(f"\n===== Training Fold: {fold_name} =====")

        # --- 2.1 Create Filtered Data for this Fold ---
        print(f"Creating filtered dataset excluding t={held_out_time}...")
        fold_adatas = []
        for adata_orig in full_adatas:
            adata_filt = adata_orig[adata_orig.obs['t'] != held_out_time].copy()
            fold_adatas.append(adata_filt)

        # --- 2.2 Create a Temporary DataModule View for this Fold ---
        original_adatas_backup = datamodule.adatas
        datamodule.adatas = fold_adatas

        # --- 2.3 Instantiate and Train Model ---
        model = None
        
        if MODEL_TYPE == "rf":
            print("Using Reference Fitting model...")
            # Initialize RF model
            model = ReferenceFittingModule(use_cuda=(DEVICE == "cuda"))
            
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
            
            model = SF2MLitModule(
                datamodule=datamodule,
                T=T_max,
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
            )
            
            # Train the model with Lightning
            print("Setting up Trainer...")
            fold_logger = TensorBoardLogger(RESULTS_DIR, name=fold_name)
            trainer = Trainer(
                max_epochs=-1,
                max_steps=N_STEPS_PER_FOLD,
                accelerator="cpu",
                devices=1,
                logger=fold_logger,
                enable_checkpointing=False,
                enable_progress_bar=True,
                log_every_n_steps=N_STEPS_PER_FOLD + 1,
                num_sanity_val_steps=0,
                limit_val_batches=0.0,
            )
            
            print(f"Training model for {N_STEPS_PER_FOLD} steps...")
            trainer.fit(model, datamodule=datamodule)
            print("Training complete.")

        # --- Restore Original Data in DataModule ---
        datamodule.adatas = original_adatas_backup

        # --- 2.6 Evaluate on the Held-Out Timepoint ---
        print(f"Evaluating model on predicting t={held_out_time} from t={held_out_time-1}...")
        model.eval()
        if MODEL_TYPE != "rf":  # SF2M and MLP need explicit device placement
            model.to(DEVICE)
        
        # Create folder for trajectory PCA plots
        # Only create plots for the last timepoint (T_max)
        if held_out_time == T_max:
            pca_folder = os.path.join(RESULTS_DIR, "pca_plots")
            fold_pca_folder = os.path.join(pca_folder, f"{MODEL_TYPE}_final_trajectory")
            os.makedirs(fold_pca_folder, exist_ok=True)

        fold_distances_list = []
        with torch.no_grad():
            # Iterate through the original full datasets
            for i, adata_full in enumerate(full_adatas):
                ko_name = datamodule.kos[i]

                # Get initial conditions (t-1) and true final state (t)
                x0_np = adata_full.X[adata_full.obs["t"] == held_out_time - 1]
                true_dist_np = adata_full.X[adata_full.obs["t"] == held_out_time]

                # Check if data exists for this transition
                if x0_np.shape[0] == 0 or true_dist_np.shape[0] == 0:
                    print(f"  Skipping dataset {i} (KO: {ko_name}): No data for t={held_out_time-1} -> t={held_out_time}.")
                    continue

                x0 = torch.from_numpy(x0_np).float()
                true_dist_cpu = torch.from_numpy(true_dist_np).float()
                
                if MODEL_TYPE == "rf":
                    # RF simulates differently than SF2M
                    # Always use non-wildtype (full) model for RF simulation
                    is_wildtype = False  # Always use the full model
                    
                    # Simulate trajectory using RF model
                    traj_rf = model.simulate_trajectory(
                        x0, 
                        n_times=1,  # Simulate one time step
                        use_wildtype=is_wildtype,  # Always use full model
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
                    
                    # Create trajectory PCA plot only for the last timepoint
                    if held_out_time == T_max:
                        create_trajectory_pca_plot(
                            adata_full,
                            sim_rf_final.numpy(),
                            ko_name,
                            held_out_time,
                            fold_pca_folder,
                            "RF"
                        )
                    
                else:  # SF2M or MLP Baseline
                    # Move tensors to the right device
                    x0 = x0.to(DEVICE)
                    true_dist = true_dist_cpu.to(DEVICE)
                    
                    # Get conditional vector
                    cond_vector_template = model.conditionals[i].to(DEVICE)
                    if cond_vector_template is not None and cond_vector_template.nelement() > 0:
                        cond_vector = cond_vector_template[0].repeat(x0.shape[0], 1)
                    else:
                        n_genes = model.n_genes
                        if model.score_net.conditional:
                            cond_dim = model.score_net.conditional_dim
                            cond_vector = torch.zeros(x0.shape[0], cond_dim, device=DEVICE)
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

                    # Create trajectory PCA plot only for the last timepoint
                    if held_out_time == T_max:
                        # For SF2M, use the SDE simulation for the plot
                        create_trajectory_pca_plot(
                            adata_full,
                            sim_sde_final.numpy(),
                            ko_name,
                            held_out_time,
                            fold_pca_folder,
                            MODEL_TYPE
                        )

                # Store metrics
                fold_distances_list.append({
                    "dataset_idx": i, 
                    "ko": ko_name,
                    "w_dist_ode": w_dist_ode,
                    "w_dist_sde": w_dist_sde,
                    "mmd2_ode": mmd2_ode,
                    "mmd2_sde": mmd2_sde,
                })
                
                # Log metrics
                print(f"  Dataset {i} (KO: {ko_name}): W_dist(ODE)={w_dist_ode:.4f}, MMD2(ODE)={mmd2_ode:.4f}")
                if MODEL_TYPE != "rf":
                    print(f"  Dataset {i} (KO: {ko_name}): W_dist(SDE)={w_dist_sde:.4f}, MMD2(SDE)={mmd2_sde:.4f}")
                
                if held_out_time == T_max:
                    print(f"  Created trajectory PCA plot for dataset {i} (KO: {ko_name})")

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
                
            results.append({
                "held_out_time": held_out_time,
                "model_type": MODEL_TYPE,
                "avg_ode_distance": avg_ode_dist,
                "avg_sde_distance": avg_sde_dist,
                "avg_mmd2_ode": avg_mmd2_ode,
                "avg_mmd2_sde": avg_mmd2_sde,
            })
            
            # Add individual distances to a global list for detailed analysis later
            for record in fold_distances_list:
                record['held_out_time'] = held_out_time
                record['model_type'] = MODEL_TYPE
                all_fold_metrics.append(record)
        else:
            print(f"Fold {fold_name}: No evaluation results.")
            results.append({
                "held_out_time": held_out_time,
                "model_type": MODEL_TYPE,
                "avg_ode_distance": np.nan,
                "avg_sde_distance": np.nan,
                "avg_mmd2_ode": np.nan,
                "avg_mmd2_sde": np.nan,
            })

    # --- 3. Final Reporting ---
    print(f"\n===== Leave-One-Out Cross-Validation Summary ({MODEL_TYPE}) =====")
    if results:
        summary_df = pd.DataFrame(results)
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

        print(f"\nOverall Average W-Distance (ODE): {final_avg_ode:.4f} +/- {final_std_ode:.4f}")
        print(f"Overall Average W-Distance (SDE): {final_avg_sde:.4f} +/- {final_std_sde:.4f}")
        print(f"Overall Average MMD2 (ODE): {final_avg_mmd2_ode:.4f} +/- {final_std_mmd2_ode:.4f}")
        print(f"Overall Average MMD2 (SDE): {final_avg_mmd2_sde:.4f} +/- {final_std_mmd2_sde:.4f}")

        # Save summary results
        summary_df.to_csv(os.path.join(RESULTS_DIR, f"loo_summary_{MODEL_TYPE}.csv"), index=False)

        if all_fold_metrics:
            detailed_df = pd.DataFrame(all_fold_metrics)
            detailed_df.to_csv(os.path.join(RESULTS_DIR, f"loo_detailed_metrics_{MODEL_TYPE}.csv"), index=False)
            print(f"\nDetailed metrics saved to loo_detailed_metrics_{MODEL_TYPE}.csv")

    else:
        print("No results generated.")

    print(f"\nLogs and results saved in: {RESULTS_DIR}")
    print(f"PCA plots saved in: {os.path.join(RESULTS_DIR, 'pca_plots')}")
    print("Script finished.")


if __name__ == "__main__":
    main()