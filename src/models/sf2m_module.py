import copy
import math
import os
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchsde
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchvision import transforms
from tqdm import tqdm

from src.datamodules.grn_datamodule import TrajectoryStructureDataModule
from src.models.components.base import MLPODEFKO
from src.models.components.cond_mlp import MLP as CONDMLP, MLPFlow
from src.models.components.simple_mlp import MLP

from .components.distribution_distances import compute_distribution_distances
from .components.optimal_transport import EntropicOTFM
from .components.plotting import (
    compute_global_jacobian,
    log_causal_graph_matrices,
    plot_auprs,
    plot_paths,
    plot_samples,
    plot_trajectory,
    maskdiag,
)
from .components.schedule import ConstantNoiseScheduler, NoiseScheduler
from .components.solver import TrajectorySolver, simulate_trajectory, wasserstein


class SF2MLitModule(LightningModule):
    def __init__(
        self,
        datamodule,
        T=10,
        sigma=1.0,
        dt=0.1,
        batch_size=64,
        alpha=0.5,
        reg=1e-5,
        correction_reg_strength=1e-3,
        n_steps=2000,
        lr=1e-3,
        device=None,
        GL_reg=0.04,
        knockout_hidden=100,
        score_hidden=[100, 100],
        correction_hidden=[64, 64],
        optimizer=Any,
        enable_epoch_end_hook: bool = True,
        use_mlp_baseline: bool = False,
        use_correction_mlp: bool = True,
    ):
        """Initializes the sf2m_ngm model and loads data.

        Args:
            data_path (str): Path to your data.
            dataset_type (str): Type of dataset (e.g., "Synthetic").
            T (float): Maximum time in your data (for OTFM).
            sigma (float): Sigma value used in the OTFMs.
            dt (float): Step size for time in OTFMs.
            batch_size (int): Batch size for training.
            alpha (float): Mixing parameter for score vs. flow matching losses.
            reg (float): Regularization weight for flow model.
            correction_reg_strength (float): Regularization weight for correction model.
            n_steps (int): Number of training steps.
            lr (float): Learning rate.
            device (str or None): If None, auto-detect GPU if available; else use CPU.
            GL_reg (float): Group Lasso regularization parameter (used in MLPODEFKO).
            knockout_hidden (int): Hidden dimension for MLPODEFKO.
            score_hidden (list): Hidden sizes for the score network.
            correction_hidden (list): Hidden sizes for the correction MLP.
        """
        super().__init__()

        # Store hyperparameters
        self.T = T
        self.sigma = sigma
        self.dt = dt
        self.batch_size = batch_size
        self.alpha = alpha
        self.reg = reg
        self.correction_reg_strength = correction_reg_strength
        self.n_steps = n_steps
        self.lr = lr
        self.use_mlp_baseline = use_mlp_baseline
        self.use_correction_mlp = use_correction_mlp

        self.save_hyperparameters()

        self.enable_epoch_end_hook = enable_epoch_end_hook

        # -----------------------
        # 1. Load the data
        # -----------------------
        self.data_loader = datamodule
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        self.adatas = self.data_loader.get_subset_adatas()
        self.kos = self.data_loader.kos
        self.ko_indices = self.data_loader.ko_indices
        self.true_matrix = self.data_loader.true_matrix.values

        # Example shape from the first dataset
        self.n_genes = self.adatas[0].X.shape[1]

        # -----------------------
        # 2. Build conditionals
        # -----------------------
        # For each dataset, build a one-hot vector that indicates which knockout it belongs to
        self.conditionals = []
        for i, ko_name in enumerate(self.kos):
            cond_matrix = torch.zeros(self.batch_size, self.n_genes)
            if ko_name is not None:
                cond_matrix[:, self.ko_indices[i]] = 1
            self.conditionals.append(cond_matrix)

        # -----------------------
        # 3. Build knockout masks
        # -----------------------
        self.knockout_masks = []
        for i, adata in enumerate(self.adatas):
            d = adata.X.shape[1]
            mask_i = self.build_knockout_mask(d, self.ko_indices[i])
            self.knockout_masks.append(mask_i)

        self.automatic_optimization = False

        # -----------------------
        # 4. Create the models
        # -----------------------
        # Dimensions for MLPODEFKO
        self.dims = [self.n_genes, knockout_hidden, 1]

        if self.use_mlp_baseline:
            self.func_v = MLPFlow(
                dims=self.dims, GL_reg=GL_reg, bias=True, knockout_masks=self.knockout_masks, device=device
            )
        else:
            self.func_v = MLPODEFKO(
                dims=self.dims, GL_reg=GL_reg, bias=True, knockout_masks=self.knockout_masks, device=device
            )

        self.score_net = CONDMLP(
            d=self.n_genes,
            hidden_sizes=score_hidden,
            time_varying=True,
            conditional=True,
            conditional_dim=self.n_genes,
            device=device
        )

        # Create correction network only if enabled
        if self.use_correction_mlp:
            self.v_correction = MLP(d=self.n_genes, hidden_sizes=correction_hidden, time_varying=True)
        else:
            # Create a dummy module that returns zeros
            class ZeroModule(torch.nn.Module):
                def __init__(self): 
                    super().__init__()
                def forward(self, t, x): 
                    return torch.zeros_like(x)
            self.v_correction = ZeroModule()

        # -----------------------
        # 5. Build OTFMs
        # -----------------------
        self.otfms = self.build_entropic_otfms(self.adatas, T=self.T, sigma=self.sigma, dt=self.dt)

        # -----------------------
        # 6. Setup optimizer
        # -----------------------
        self.optimizer = optimizer

        # For tracking losses
        self.val_results = []
        self.train_results = []

    # -------------------------------------------------------------------------
    # Supporting methods
    # -------------------------------------------------------------------------
    def build_knockout_mask(self, d, ko_idx):
        """Build a [d, d] adjacency mask for a knockout of gene ko_idx.

        If ko_idx is None, return a mask of all ones (wild-type).
        """
        if ko_idx is None:
            # No knockout => no edges removed
            return np.ones((d, d), dtype=np.float32)
        else:
            mask = np.ones((d, d), dtype=np.float32)
            g = ko_idx
            mask[:, g] = 0.0
            mask[g, g] = 1.0
            return mask

    def build_entropic_otfms(self, adatas, T, sigma, dt):
        """Returns a list of EntropicOTFM objects, one per dataset."""
        otfms = []
        for i, adata in enumerate(adatas):
            x_tensor = torch.tensor(adata.X, dtype=torch.float32)
            t_idx = torch.tensor(adata.obs["t"], dtype=torch.long)
            model = EntropicOTFM(
                x=x_tensor,
                t_idx=t_idx,
                dt=dt,
                sigma=sigma,
                T=T,
                dim=x_tensor.shape[1],
                device=self.device,
            )
            otfms.append(model)
        return otfms

    def proximal(self, w, dims, lam=0.1, eta=0.1):
        """Proximal operator used for group-lasso style regularization in the hidden weights."""
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    def mlp_l2_reg(self, mlp):
        """Compute L2 sum of parameters for a generic MLP."""
        l2_sum = 0.0
        for param in mlp.parameters():
            l2_sum += torch.sum(param**2)
        return l2_sum

    # -------------------------------------------------------------------------
    # Core training loop
    # -------------------------------------------------------------------------
    def training_step(self, *args, **kwargs):
        """Combine flow matching + score matching with multiple datasets."""
        optimizer = self.optimizers()

        # Randomly pick which dataset to train on
        ds_idx = np.random.randint(0, len(self.adatas))
        model = self.otfms[ds_idx]
        cond_vector = self.conditionals[ds_idx].to(self.device)

        # Sample bridging flows
        _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
            batch_size=self.batch_size, skip_time=None
        )

        optimizer.zero_grad()

        # Prepare inputs
        s_input = _x.unsqueeze(1)
        v_input = _x.unsqueeze(1)
        t_input = _t.unsqueeze(1)
        B = _x.shape[0]

        # Expand conditional vectors to match batch size
        cond_expanded = cond_vector.repeat(B // cond_vector.shape[0] + 1, 1)[:B]

        # Score net output
        s_fit = self.score_net(_t, _x, cond_expanded).squeeze(1)

        # Flow net output, with or without correction
        if self.global_step <= 500 or not self.use_correction_mlp:
            # Warmup phase or no correction
            v_fit = self.func_v(t_input, v_input).squeeze(1) - (
                model.sigma**2 / 2
            ) * self.score_net(_t, _x, cond_expanded)
        else:
            # Full training phase with correction
            v_fit = self.func_v(t_input, v_input).squeeze(1) + self.v_correction(_t, _x)
            v_fit = v_fit - (model.sigma**2 / 2) * self.score_net(_t, _x, cond_expanded)

        # Losses
        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean(
            (_t_orig * (1 - _t_orig)) * (v_fit * model.dt - _u) ** 2
        )  # weighting by (_t_orig * (1 - _t_orig))
        L_reg = self.func_v.l2_reg() + self.func_v.fc1_reg()
        
        # Only apply correction regularization if we're using the correction network
        if self.use_correction_mlp:
            L_reg_correction = self.mlp_l2_reg(self.v_correction)
        else:
            L_reg_correction = torch.tensor(0.0, device=self.device)

        # Loss combination logic
        if self.global_step < 100:
            # Train only score initially
            L = self.alpha * L_score
        elif self.global_step <= 500 or not self.use_correction_mlp:
            # Mix score + flow + small reg (or no correction case)
            L = self.alpha * L_score + (1 - self.alpha) * L_flow + self.reg * L_reg
        else:
            # Full combined loss with correction reg
            L = (
                self.alpha * L_score
                + (1 - self.alpha) * L_flow
                + self.reg * L_reg
                + self.correction_reg_strength * L_reg_correction
            )

        if self.global_step % 100 == 0:
            print(
                f"Step={self.global_step}, ds={ds_idx}, "
                f"L_score={L_score.item():.4f}, L_flow={L_flow.item():.4f}, "
                f"Reg(Flow)={L_reg.item():.4f}, Reg(Corr)={L_reg_correction.item():.4f}"
            )

        self.log("train/loss", L.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/flow_loss", L_flow.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/score_loss",
            L_score.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/reg_loss", L_reg.item(), on_step=True, on_epoch=True, prog_bar=True)

        # Backprop and update
        self.manual_backward(L)
        optimizer.step()

        # Proximal step (group-lasso style)
        if not self.use_mlp_baseline:
            self.proximal(self.func_v.fc1.weight, self.func_v.dims, lam=self.func_v.GL_reg, eta=0.01)

    def on_train_epoch_end(self):
        if not self.enable_epoch_end_hook:
            return
        
        # Import pandas at the beginning of the method
        import pandas as pd
        
        # Get all gene names from the model
        model_gene_names = self.data_loader.adatas[0].var_names
        
        # Compute the full Jacobian (estimated causal graph)
        with torch.no_grad():
            A_estim = compute_global_jacobian(self.func_v, self.adatas, dt=1/self.T, device=torch.device("cpu"))
        
        # Get the directly extracted causal graph from the model
        W_v = self.func_v.causal_graph(w_threshold=0.0).T
        
        # Different handling for DataFrame (Renge) vs. numpy array (synthetic)
        if isinstance(self.data_loader.true_matrix, pd.DataFrame):
            # For Renge dataset: extract the exact subset that matches the reference network
            ref_network = self.data_loader.true_matrix
            ref_rows = ref_network.index
            ref_cols = ref_network.columns
            
            # Create DataFrames for the estimated graphs with all genes
            A_estim_df = pd.DataFrame(A_estim, index=model_gene_names, columns=model_gene_names)
            W_v_df = pd.DataFrame(W_v, index=model_gene_names, columns=model_gene_names)
            
            # Extract the exact subset that corresponds to the reference network dimensions
            A_estim_subset = A_estim_df.loc[ref_rows, ref_cols]
            W_v_subset = W_v_df.loc[ref_rows, ref_cols]
            
            # Convert to numpy arrays for evaluation
            A_estim_np = A_estim_subset.values
            W_v_np = W_v_subset.values
            A_true_np = ref_network.values
            
            # Plot with the subset matrices
            plot_auprs(W_v_np, A_estim_np, A_true_np, self.logger, self.global_step)
            log_causal_graph_matrices(A_estim_np, W_v_np, A_true_np, self.logger, self.global_step)
        else:
            # Standard handling for synthetic data
            A_true = self.true_matrix
            plot_auprs(W_v, A_estim, A_true, self.logger, self.global_step)
            log_causal_graph_matrices(A_estim, W_v, A_true, self.logger, self.global_step)

        table_rows = []

        if self.current_epoch % 10 == 0:
            results = []
            for time in range(1, self.T):
                time_distances = []
                for i, adata in enumerate(self.adatas):
                    x0 = torch.from_numpy(adata.X[adata.obs["t"] == time - 1]).float()
                    true_dist = torch.from_numpy(adata.X[adata.obs["t"] == time]).float()
                    cond_vector = self.conditionals[i]
                    if cond_vector is not None:
                        cond_vector = cond_vector[0].repeat(len(x0), 1)

                    if len(x0) == 0 or len(true_dist) == 0:
                        continue

                    traj_ode = simulate_trajectory(
                        self.func_v,
                        self.v_correction,
                        self.score_net,
                        x0,
                        dataset_idx=i,
                        start_time=time - 1,
                        end_time=time,
                        n_times=min(len(x0), len(true_dist)),
                        cond_vector=cond_vector,
                    )

                    traj_sde = simulate_trajectory(
                        self.func_v,
                        self.v_correction,
                        self.score_net,
                        x0,
                        dataset_idx=i,
                        start_time=time - 1,
                        end_time=time,
                        n_times=min(len(x0), len(true_dist)),
                        cond_vector=cond_vector,
                        use_sde=True,
                    )

                    w_dist_ode = wasserstein(traj_ode[-1], true_dist)
                    w_dist_sde = wasserstein(traj_sde[-1], true_dist)
                    time_distances.append({"ode": w_dist_ode, "sde": w_dist_sde})

                if time_distances:
                    avg_ode = np.mean([d["ode"] for d in time_distances])
                    avg_sde = np.mean([d["sde"] for d in time_distances])
                else:
                    avg_ode, avg_sde = None, None

                results.append({"Time": time, "Avg ODE": avg_ode, "Avg SDE": avg_sde})

            self.train_results.extend(results)

            # Flatten results if multiple batches
            all_results = self.train_results

            # Sort results by time and log the table
            all_results = sorted(all_results, key=lambda r: r["Time"])
            df = pd.DataFrame(all_results)
            table_str = df.to_markdown(index=False)
            if self.logger is not None:
                self.logger.experiment.add_text("Validation Wasserstein Distances", table_str, global_step=self.global_step)
            # Optionally also log individual metrics:
            for row in all_results:
                self.log(f"train/w_dist_ode_time_{row['Time']}", row["Avg ODE"], prog_bar=True)
                self.log(f"train/w_dist_sde_time_{row['Time']}", row["Avg SDE"], prog_bar=True)

            self.train_results.clear()
        else:
            pass

    def validation_step(self, batch, batch_idx):
        if not self.enable_epoch_end_hook:
            return
        
        # Get validation adatas
        val_adatas = self.data_loader.get_subset_adatas("val")
        
        # Get all gene names from the model
        model_gene_names = self.data_loader.adatas[0].var_names
        
        # Compute the full Jacobian (estimated causal graph)
        with torch.no_grad():
            A_estim = compute_global_jacobian(self.func_v, val_adatas, dt=1/self.T, device=torch.device("cpu"))
        
        # Get the directly extracted causal graph from the model
        W_v = self.func_v.causal_graph(w_threshold=0.0).T
        
        # Different handling for DataFrame (Renge) vs. numpy array (synthetic)
        if isinstance(self.data_loader.true_matrix, pd.DataFrame):
            # For Renge dataset: extract the exact subset that matches the reference network
            ref_network = self.data_loader.true_matrix
            ref_rows = ref_network.index
            ref_cols = ref_network.columns
            
            # Create DataFrames for the estimated graphs with all genes
            A_estim_df = pd.DataFrame(A_estim, index=model_gene_names, columns=model_gene_names)
            W_v_df = pd.DataFrame(W_v, index=model_gene_names, columns=model_gene_names)
            
            # Extract the exact subset that corresponds to the reference network dimensions
            A_estim_subset = A_estim_df.loc[ref_rows, ref_cols]
            W_v_subset = W_v_df.loc[ref_rows, ref_cols]
            
            # Convert to numpy arrays for evaluation
            A_estim_np = A_estim_subset.values
            W_v_np = W_v_subset.values
            A_true_np = ref_network.values
            
            # Plot with the subset matrices
            plot_auprs(W_v_np, A_estim_np, A_true_np, self.logger, self.global_step)
            log_causal_graph_matrices(A_estim_np, W_v_np, A_true_np, self.logger, self.global_step)
        else:
            # Standard handling for synthetic data
            A_true = self.true_matrix
            plot_auprs(W_v, A_estim, A_true, self.logger, self.global_step)
            log_causal_graph_matrices(A_estim, W_v, A_true, self.logger, self.global_step)
        
        # Continue with the standard Wasserstein distance evaluation
        results = []

        for time in range(1, self.T):
            time_distances = []
            for i, adata in enumerate(val_adatas):
                x0 = torch.from_numpy(adata.X[adata.obs["t"] == time - 1]).float()
                true_dist = torch.from_numpy(adata.X[adata.obs["t"] == time]).float()
                cond_vector = self.conditionals[i]
                if cond_vector is not None:
                    cond_vector = cond_vector[0].repeat(len(x0), 1)

                if len(x0) == 0 or len(true_dist) == 0:
                    continue

                traj_ode = simulate_trajectory(
                    self.func_v,
                    self.v_correction,
                    self.score_net,
                    x0,
                    dataset_idx=i,
                    start_time=time - 1,
                    end_time=time,
                    n_times=min(len(x0), len(true_dist)),
                    cond_vector=cond_vector,
                )

                # You might also compute the SDE metric similarly
                traj_sde = simulate_trajectory(
                    self.func_v,
                    self.v_correction,
                    self.score_net,
                    x0,
                    dataset_idx=i,
                    start_time=time - 1,
                    end_time=time,
                    n_times=min(len(x0), len(true_dist)),
                    cond_vector=cond_vector,
                    use_sde=True,
                )

                w_dist_ode = wasserstein(traj_ode[-1], true_dist)
                w_dist_sde = wasserstein(traj_sde[-1], true_dist)
                time_distances.append({"ode": w_dist_ode, "sde": w_dist_sde})

            if time_distances:
                avg_ode = np.mean([d["ode"] for d in time_distances])
                avg_sde = np.mean([d["sde"] for d in time_distances])
            else:
                avg_ode, avg_sde = None, None

            results.append({"Time": time, "Avg ODE": avg_ode, "Avg SDE": avg_sde})

        self.val_results.extend(results)

    def on_validation_epoch_end(self):        
        # Flatten results if multiple batches
        all_results = self.val_results

        # Sort results by time and log the table
        all_results = sorted(all_results, key=lambda r: r["Time"])
        df = pd.DataFrame(all_results)
        table_str = df.to_markdown(index=False)
        if self.logger is not None:
            self.logger.experiment.add_text("Validation Wasserstein Distances", table_str, global_step=self.global_step)
        # Optionally also log individual metrics:
        for row in all_results:
            self.log(f"val/w_dist_ode_time_{row['Time']}", row["Avg ODE"], prog_bar=True)
            self.log(f"val/w_dist_sde_time_{row['Time']}", row["Avg SDE"], prog_bar=True)

        self.val_results.clear()


    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        params_to_optimize = list(self.func_v.parameters()) + list(self.score_net.parameters())
        
        # Only include correction network parameters if we're using it
        if self.use_correction_mlp:
            params_to_optimize += list(self.v_correction.parameters())
            
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.hparams.lr,
            eps=1e-7,
        )
        return optimizer
