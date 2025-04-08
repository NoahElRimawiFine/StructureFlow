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
from src.models.components.solver import simulate_trajectory, wasserstein
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
            datamodule=datamodule, # Pass the datamodule instance (currently holding filtered data)
            T=T_max,
            sigma=SIGMA,
            dt=DT_data,
            batch_size=BATCH_SIZE,
            alpha=ALPHA,
            reg=REG,
            correction_reg_strength=CORRECTION_REG,
            n_steps=N_STEPS_PER_FOLD, # Total steps for this fold's training
            lr=LR,
            device=DEVICE, # Pass device hint, although Trainer manages final placement
            GL_reg=GL_REG,
            knockout_hidden=KNOCKOUT_HIDDEN,
            score_hidden=SCORE_HIDDEN,
            correction_hidden=CORRECTION_HIDDEN,
            enable_epoch_end_hook=False
            # Let SF2MLitModule handle its own optimizer creation via configure_optimizers
            # optimizer=optimizer_partial, # Removed this line
        )
        # Note: device placement is handled by Trainer, but model needs internal device hint if used early

        # --- 2.4 Setup Trainer for this Fold ---
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

        # --- 2.6 Evaluate on the Held-Out Timepoint ---
        print(f"Evaluating model on predicting t={held_out_time} from t={held_out_time-1}...")
        model.eval() # Set model to evaluation mode
        model.to(DEVICE) # Ensure model is on the correct device for inference

        fold_distances_list = [] # Store distances for each dataset within this fold
        with torch.no_grad():
            # Iterate through the *original* full datasets
            for i, adata_full in enumerate(full_adatas):
                ko_name = datamodule.kos[i] # Get knockout name for logging

                # Get initial conditions (t-1) and true final state (t) from the *full* data
                x0_np = adata_full.X[adata_full.obs["t"] == held_out_time - 1]
                true_dist_np = adata_full.X[adata_full.obs["t"] == held_out_time]

                # Check if data exists for this transition
                if x0_np.shape[0] == 0 or true_dist_np.shape[0] == 0:
                    print(f"  Skipping dataset {i} (KO: {ko_name}): No data for t={held_out_time-1} -> t={held_out_time}.")
                    continue

                x0 = torch.from_numpy(x0_np).float().to(DEVICE)
                true_dist = torch.from_numpy(true_dist_np).float().to(DEVICE)

                # Get the conditional vector corresponding to this dataset index (i)
                # SF2MLitModule stores these based on the initial datamodule order
                cond_vector_template = model.conditionals[i].to(DEVICE) # Shape [train_batch_size, n_genes]
                if cond_vector_template is not None and cond_vector_template.nelement() > 0:
                    # Use the template's pattern, replicated for the actual batch size of x0
                    cond_vector = cond_vector_template[0].repeat(x0.shape[0], 1)
                else: # Handle case where there might be no conditioning (e.g., WT default)
                    # Check if model expects None or zeros
                    # Assuming zeros if template is empty/None but conditioning is conceptually needed
                    n_genes = model.n_genes
                    # Check if score_net expects conditional input
                    if model.score_net.conditional:
                        cond_dim = model.score_net.conditional_dim
                        cond_vector = torch.zeros(x0.shape[0], cond_dim, device=DEVICE)
                    else: # if score_net is not conditional
                        cond_vector = None

                # --- Simulate Trajectories ---
                # Use the standalone simulate_trajectory function
                common_sim_args = {
                    "flow_model": model.func_v,
                    "correction_model": model.v_correction,
                    "score_model": model.score_net,
                    "x0": x0,
                    "dataset_idx": i, # Pass dataset index for potential use in models (e.g., masks)
                    "start_time": held_out_time - 1,
                    "end_time": held_out_time,
                    "n_times": N_TIMES_SIM,
                    "cond_vector": cond_vector,
                    "T_max": T_max, # Pass max time for correct time scaling
                    "sigma": SIGMA,
                    "device": DEVICE,
                }

                # ODE Simulation
                traj_ode = simulate_trajectory(**common_sim_args, use_sde=False)
                sim_ode_final = traj_ode[-1].cpu() # Get final state, move to CPU for metric calc

                # SDE Simulation
                traj_sde = simulate_trajectory(**common_sim_args, use_sde=True)
                sim_sde_final = traj_sde[-1].cpu()

                # Compute Wasserstein distance
                # Ensure wasserstein function handles tensors
                w_dist_ode = wasserstein(sim_ode_final, true_dist.cpu())
                w_dist_sde = wasserstein(sim_sde_final, true_dist.cpu())

                fold_distances_list.append({
                    "dataset_idx": i,
                    "ko": ko_name,
                    "w_dist_ode": w_dist_ode.item() if isinstance(w_dist_ode, torch.Tensor) else w_dist_ode,
                    "w_dist_sde": w_dist_sde.item() if isinstance(w_dist_sde, torch.Tensor) else w_dist_sde,
                })
                print(f"  Dataset {i} (KO: {ko_name}): W_dist(ODE)={w_dist_ode:.4f}, W_dist(SDE)={w_dist_sde:.4f}")

                # --- Optional: Plotting for this specific dataset/fold ---
                # You could add PCA plotting logic here, comparing true_dist_np and sim_sde_final.numpy()
                # Example for the last fold and first dataset (if desired)
                # if held_out_time == T_max and i == 0:
                #     print("Generating PCA plot...")
                #     # PCA fitting (fit on combined data or just this dataset/timepoint?)
                #     pca = PCA(n_components=2)
                #     # Fit PCA on the true data points involved in the transition
                #     combined_true_data = np.vstack((x0_np, true_dist_np))
                #     pca.fit(combined_true_data)

                #     # Transform data
                #     x0_pca = pca.transform(x0_np)
                #     true_dist_pca = pca.transform(true_dist_np)
                #     sim_sde_pca = pca.transform(sim_sde_final.numpy()) # Use SDE results for plot

                #     plt.figure(figsize=(8, 6))
                #     plt.scatter(x0_pca[:, 0], x0_pca[:, 1], alpha=0.5, label=f"True t={held_out_time-1}", c='blue')
                #     plt.scatter(true_dist_pca[:, 0], true_dist_pca[:, 1], alpha=0.5, label=f"True t={held_out_time}", c='green')
                #     plt.scatter(sim_sde_pca[:, 0], sim_sde_pca[:, 1], alpha=0.5, label=f"Simulated t={held_out_time} (SDE)", c='red', marker='x')
                #     plt.title(f"PCA: Holdout t={held_out_time}, Dataset {i} (KO: {ko_name})")
                #     plt.xlabel("PC1")
                #     plt.ylabel("PC2")
                #     plt.legend()
                #     plt.grid(True)
                #     plot_filename = os.path.join(RESULTS_DIR, fold_name, f"pca_ds{i}_t{held_out_time}.png")
                #     os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
                #     plt.savefig(plot_filename)
                #     plt.close()


        # --- 2.7 Aggregate and Store Results for this Fold ---
        if fold_distances_list:
            fold_df = pd.DataFrame(fold_distances_list)
            avg_ode_dist = fold_df["w_dist_ode"].mean()
            avg_sde_dist = fold_df["w_dist_sde"].mean()
            print(f"Fold {fold_name} Avg W_dist: ODE={avg_ode_dist:.4f}, SDE={avg_sde_dist:.4f}")
            results.append({
                "held_out_time": held_out_time,
                "avg_ode_distance": avg_ode_dist,
                "avg_sde_distance": avg_sde_dist,
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
            })

    # --- 3. Final Reporting ---
    print("\n===== Leave-One-Out Cross-Validation Summary =====")
    if results:
        summary_df = pd.DataFrame(results)
        print("Average Wasserstein Distances per Fold:")
        print(summary_df.to_string(index=False))

        final_avg_ode = summary_df["avg_ode_distance"].mean()
        final_avg_sde = summary_df["avg_sde_distance"].mean()
        final_std_ode = summary_df["avg_ode_distance"].std()
        final_std_sde = summary_df["avg_sde_distance"].std()


        print(f"\nOverall Average W-Distance (ODE): {final_avg_ode:.4f} +/- {final_std_ode:.4f}")
        print(f"Overall Average W-Distance (SDE): {final_avg_sde:.4f} +/- {final_std_sde:.4f}")

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