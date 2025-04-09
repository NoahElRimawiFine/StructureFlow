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
USE_MLP_BASELINE = False
USE_CORRECTION_MLP = True

def create_pca_plot(x0, true_dist, predicted_dist, ko_name, held_out_time, folder_path, sim_type="SDE"):
    """
    Create and save a PCA plot comparing true and predicted distributions.
    
    Args:
        x0: Initial state (t-1) as numpy array
        true_dist: True final state (t) as numpy array
        predicted_dist: Predicted final state (t) as numpy array
        ko_name: Name of the knockout for labeling
        held_out_time: The held-out timepoint
        folder_path: Path to save the plot
        sim_type: "SDE" or "ODE" for labeling
    """
    # Create component folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Fit PCA on the combined true data points
    pca = PCA(n_components=2)
    combined_true_data = np.vstack((x0, true_dist))
    pca.fit(combined_true_data)
    
    # Transform data
    x0_pca = pca.transform(x0)
    true_dist_pca = pca.transform(true_dist)
    predicted_dist_pca = pca.transform(predicted_dist)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x0_pca[:, 0], x0_pca[:, 1], alpha=0.7, label=f"t={held_out_time-1}", c='blue')
    plt.scatter(true_dist_pca[:, 0], true_dist_pca[:, 1], alpha=0.7, label=f"True t={held_out_time}", c='green')
    plt.scatter(predicted_dist_pca[:, 0], predicted_dist_pca[:, 1], alpha=0.7, 
               label=f"{sim_type} Predicted t={held_out_time}", c='red', marker='x')
    
    # Add title and labels
    ko_label = "Wild Type" if ko_name is None else f"Knockout: {ko_name}"
    plt.title(f"PCA: {ko_label}, {sim_type} Prediction, t={held_out_time-1}→{held_out_time}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    filename = f"{sim_type.lower()}_{'wildtype' if ko_name is None else f'ko_{ko_name}'}_t{held_out_time}.png"
    plt.savefig(os.path.join(folder_path, filename), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    seed_everything(SEED, workers=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Load Full Data Once ---
    print("Loading full dataset...")
    # Use the same DataModule class as used for training SF2MLitModule
    datamodule = TrajectoryStructureDataModule(
        data_path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        batch_size=BATCH_SIZE,
        use_dummy_train_loader=True,
        dummy_loader_steps=N_STEPS_PER_FOLD,
        num_workers=0,
        # Add any other necessary arguments for your specific DataModule
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit") # Load full data

    # Get a reference to the original list of AnnData objects
    # We'll create filtered copies later, but need the originals for evaluation
    full_adatas = datamodule.get_subset_adatas()
    # Determine the time range from the data
    if not full_adatas:
        raise ValueError("No datasets loaded.")
    T_max = int(max(adata.obs['t'].max() for adata in full_adatas))
    T_times = T_max + 1 # Number of distinct time points (e.g., 0, 1, ..., T_max)
    DT_data = 1.0 / T_max if T_max > 0 else 1.0 # Time step duration in the data

    print(f"Data loaded. Found {len(full_adatas)} datasets with T_max={T_max}.")
    print(f"Kos: {datamodule.kos}")

    # --- 2. Leave-One-Out Cross-Validation Loop ---
    results = []
    all_fold_metrics = []

    # Define the timepoints to hold out during training
    # We can only evaluate prediction from t-1 to t, so we hold out t=1, 2, ..., T_max
    timepoints_to_hold_out = range(1, T_times)

    for held_out_time in timepoints_to_hold_out:
        fold_name = f"holdout_{held_out_time}"
        print(f"\n===== Training Fold: {fold_name} =====")

        # --- 2.1 Create Filtered Data for this Fold ---
        print(f"Creating filtered dataset excluding t={held_out_time}...")
        fold_adatas = []
        for adata_orig in full_adatas:
            # Create a copy to avoid modifying the original list/objects
            adata_filt = adata_orig[adata_orig.obs['t'] != held_out_time].copy()
            fold_adatas.append(adata_filt)

        # --- 2.2 Create a Temporary DataModule View for this Fold ---
        # We'll temporarily replace the 'adatas' list in the datamodule instance.
        # SF2MLitModule reads from this list in its __init__ to build OTFMs etc.
        original_adatas_backup = datamodule.adatas
        datamodule.adatas = fold_adatas # Replace with filtered data

        # --- 2.3 Instantiate Model for this Fold ---
        # A fresh model instance is created for each fold.
        # It will initialize using the filtered `datamodule.adatas`.
        print("Instantiating SF2MLitModule...")
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
            use_mlp_baseline=USE_MLP_BASELINE,
            use_correction_mlp=USE_CORRECTION_MLP and not USE_MLP_BASELINE,
        )
        print("Setting up Trainer...")
        fold_logger = TensorBoardLogger(RESULTS_DIR, name=fold_name)
        trainer = Trainer(
            max_epochs=-1,
            max_steps=N_STEPS_PER_FOLD,
            accelerator="cpu", # Let Lightning detect GPU/CPU
            devices=1,
            logger=fold_logger,
            enable_checkpointing=False, # Simplifies the process for now
            enable_progress_bar=True,
            log_every_n_steps=N_STEPS_PER_FOLD + 1,
            num_sanity_val_steps=0,
            limit_val_batches=0.0,
            # Add deterministic=True if needed, potentially requires setting CUBLAS workspace config
            # deterministic=True,
        )

        # --- 2.5 Train the Model ---
        print(f"Training model for {N_STEPS_PER_FOLD} steps...")
        # The trainer will use the dataloaders provided by the datamodule,
        # which implicitly use the filtered 'fold_adatas' list via the OTFMs
        # created inside SF2MLitModule's __init__.
        trainer.fit(model, datamodule=datamodule) # Pass datamodule for completeness

        # --- Restore Original Data in Datamodule ---
        # Crucial before evaluation or the next fold!
        datamodule.adatas = original_adatas_backup

        print(f"Evaluating model on predicting t={held_out_time} from t={held_out_time-1}...")
        model.eval()
        model.to(DEVICE)
        
        # Create folders for this fold's PCA plots
        pca_folder = os.path.join(RESULTS_DIR, "pca_plots")
        fold_pca_folder = os.path.join(pca_folder, f"holdout_{held_out_time}")
        os.makedirs(fold_pca_folder, exist_ok=True)

        fold_distances_list = []
        with torch.no_grad():
            # Iterate through the *original* full datasets
            for i, adata_full in enumerate(full_adatas):
                ko_name = datamodule.kos[i]

                # Get initial conditions (t-1) and true final state (t) from the *full* data
                x0_np = adata_full.X[adata_full.obs["t"] == held_out_time - 1]
                true_dist_np = adata_full.X[adata_full.obs["t"] == held_out_time]

                # Check if data exists for this transition
                if x0_np.shape[0] == 0 or true_dist_np.shape[0] == 0:
                    print(f"  Skipping dataset {i} (KO: {ko_name}): No data for t={held_out_time-1} -> t={held_out_time}.")
                    continue

                x0 = torch.from_numpy(x0_np).float().to(DEVICE)
                true_dist = torch.from_numpy(true_dist_np).float().to(DEVICE)
                true_dist_cpu = true_dist.cpu()

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

                # Create PCA plots for both ODE and SDE simulations
                create_pca_plot(
                    x0_np, 
                    true_dist_np, 
                    sim_ode_final.numpy(), 
                    ko_name, 
                    held_out_time, 
                    fold_pca_folder, 
                    sim_type="ODE"
                )
                
                create_pca_plot(
                    x0_np, 
                    true_dist_np, 
                    sim_sde_final.numpy(), 
                    ko_name, 
                    held_out_time, 
                    fold_pca_folder, 
                    sim_type="SDE"
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
                print(f"  Dataset {i} (KO: {ko_name}): W_dist(ODE)={w_dist_ode:.4f}, MMD2(ODE)={mmd2_ode:.4f}, W_dist(SDE)={w_dist_sde:.4f}, MMD2(SDE)={mmd2_sde:.4f}")
                print(f"  Created PCA plots for dataset {i} (KO: {ko_name})")

        # --- 2.7 Aggregate and Store Results for this Fold ---
        if fold_distances_list:
            fold_df = pd.DataFrame(fold_distances_list)
            avg_ode_dist = fold_df["w_dist_ode"].mean()
            avg_sde_dist = fold_df["w_dist_sde"].mean()
            avg_mmd2_ode = fold_df["mmd2_ode"].mean()
            avg_mmd2_sde = fold_df["mmd2_sde"].mean()
            print(f"Fold {fold_name} Avg W_dist: ODE={avg_ode_dist:.4f}, SDE={avg_sde_dist:.4f}")
            results.append({
                "held_out_time": held_out_time,
                "avg_ode_distance": avg_ode_dist,
                "avg_sde_distance": avg_sde_dist,
                "avg_mmd2_ode": avg_mmd2_ode,
                "avg_mmd2_sde": avg_mmd2_sde,
            })
            # Add individual distances to a global list for detailed analysis later
            for record in fold_distances_list:
                record['held_out_time'] = held_out_time # Add fold info
                all_fold_metrics.append(record)
        else:
            print(f"Fold {fold_name}: No evaluation results.")
            results.append({
                "held_out_time": held_out_time,
                "avg_ode_distance": np.nan,
                "avg_sde_distance": np.nan,
                "avg_mmd2_ode": np.nan,
                "avg_mmd2_sde": np.nan,
            })

    # --- 3. Final Reporting ---
    print("\n===== Leave-One-Out Cross-Validation Summary =====")
    if results:
        summary_df = pd.DataFrame(results)
        print("Average Wasserstein Distances per Fold:")
        print(summary_df.to_string(index=False))

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
        summary_df.to_csv(os.path.join(RESULTS_DIR, "loo_summary.csv"), index=False)

        if all_fold_metrics:
            detailed_df = pd.DataFrame(all_fold_metrics)
            detailed_df.to_csv(os.path.join(RESULTS_DIR, "loo_detailed_metrics.csv"), index=False)
            print("\nDetailed metrics saved to loo_detailed_metrics.csv")

    else:
        print("No results generated.")

    print(f"\nLogs and results saved in: {RESULTS_DIR}")
    print("Script finished.")

if __name__ == "__main__":
    main()