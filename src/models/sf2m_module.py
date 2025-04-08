import copy
import math
import os
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchsde
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchvision import transforms
from tqdm import tqdm

from src.datamodules.grn_datamodule import TrajectoryStructureDataModule
from src.models.components.base import MLPODEFKO
from src.models.components.cond_mlp import MLP as CONDMLP
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
        self.func_v = MLPODEFKO(
            dims=self.dims, GL_reg=GL_reg, bias=True, knockout_masks=self.knockout_masks
        )

        self.score_net = CONDMLP(
            d=self.n_genes,
            hidden_sizes=score_hidden,
            time_varying=True,
            conditional=True,
            conditional_dim=self.n_genes,
        )

        self.v_correction = MLP(d=self.n_genes, hidden_sizes=correction_hidden, time_varying=True)

        # -----------------------
        # 5. Build OTFMs
        # -----------------------
        self.otfms = self.build_entropic_otfms(self.adatas, T=self.T, sigma=self.sigma, dt=self.dt)

        # -----------------------
        # 6. Setup optimizer
        # -----------------------
        self.optimizer = optimizer

        # For tracking losses
        self.loss_history = []
        self.score_loss_history = []
        self.flow_loss_history = []
        self.reg_loss_history = []
        self.reg_corr_loss_history = []

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
        for adata in adatas:
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
        if self.global_step <= 500:
            # Warmup phase
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
        L_reg_correction = self.mlp_l2_reg(self.v_correction)

        if self.global_step < 100:
            # Train only score initially
            L = self.alpha * L_score
        elif self.global_step <= 500:
            # Mix score + flow + small reg
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
        self.proximal(self.func_v.fc1.weight, self.func_v.dims, lam=self.func_v.GL_reg, eta=0.01)

    def on_train_epoch_end(self):
        if not self.enable_epoch_end_hook:
            return
        with torch.no_grad():
            A_estim = compute_global_jacobian(self.func_v, self.adatas, dt=1 / 5, device="cpu")
        W_v = self.func_v.causal_graph(w_threshold=0.0).T
        A_true = self.true_matrix

        plot_auprs(W_v, A_estim, A_true, self.logger, self.global_step)
        log_causal_graph_matrices(A_estim, W_v, A_true, self.logger, self.global_step)

        table_rows = []

        if self.current_epoch % 10 == 0:
            for time in range(self.T):
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
                        n_times=400,
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
                        n_times=400,
                        cond_vector=cond_vector,
                        use_sde=True,
                    )

                    # Compute Wasserstein distances for ODE and SDE simulated trajectories.
                    w_dist_ode = wasserstein(traj_ode[-1], true_dist)
                    w_dist_sde = wasserstein(traj_sde[-1], true_dist)

                    time_distances.append({"ode": w_dist_ode, "sde": w_dist_sde})

                # Compute averages for this time step if we have any results.
                if time_distances:
                    avg_ode = np.mean([d["ode"] for d in time_distances])
                    avg_sde = np.mean([d["sde"] for d in time_distances])
                else:
                    avg_ode, avg_sde = None, None

                # Save the results for this time step.
                table_rows.append({"Time": time, "Avg ODE": avg_ode, "Avg SDE": avg_sde})

            import pandas as pd

            df = pd.DataFrame(table_rows)
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.add_text(
                    "Validation Wasserstein Distances",
                    df.to_markdown(),
                    global_step=self.global_step,
                )
                self.print("Validation Wasserstein Distances:\n", df.to_string(index=False))
            else:
                self.print("Validation Wasserstein Distances:\n", df.to_string(index=False))

    def validation_step(self, batch, batch_idx):
        if not self.enable_epoch_end_hook:
            return
        x_val = batch["X"]
        t_val = batch["t"]

        unique_times = torch.unique(t_val).tolist()
        results = []

        # TODO: get adatas batched by ko index
        for time in sorted(unique_times):
            idx_x0 = t_val == time - 1
            idx_true = t_val == time
            if idx_x0.sum() == 0 or idx_true.sum() == 0:
                continue

            # Extract x0 and the true final state; move to device.
            x0 = x_val[idx_x0]
            true_dist = x_val[idx_true]

            if self.cond_matrix is not None and len(self.cond_matrix) > 0:
                # For example, take the first one (or select by some identifier)
                cond_vector = self.cond_matrix[0].to(self.device)
                # Repeat it for the number of samples in x0:
                cond_vector = cond_vector[0].repeat(x0.shape[0], 1)
            else:
                cond_vector = None

            # Simulate trajectory with ODE dynamics (using your simulate_trajectory function)
            traj_ode = simulate_trajectory(
                self.func_v,
                self.v_correction,
                self.score_net,
                x0,
                dataset_idx=0,  # adjust if you have multiple datasets
                start_time=time - 1,
                end_time=time,
                n_times=400,
                cond_vector=cond_vector,
                use_sde=False,
            )

            # Simulate trajectory with SDE dynamics.
            traj_sde = simulate_trajectory(
                self.func_v,
                self.v_correction,
                self.score_net,
                x0,
                dataset_idx=0,
                start_time=time - 1,
                end_time=time,
                n_times=400,
                cond_vector=cond_vector,
                use_sde=True,
            )

            # Compute Wasserstein distances for ODE and SDE trajectories.
            w_dist_ode = wasserstein(traj_ode[-1], true_dist)
            w_dist_sde = wasserstein(traj_sde[-1], true_dist)

            # Save metrics for this time slice.
            results.append({"Time": time, "Avg ODE": w_dist_ode, "Avg SDE": w_dist_sde})

        # Return results as the output of validation_step.
        return results

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        optimizer = torch.optim.AdamW(
            list(self.func_v.parameters())
            + list(self.score_net.parameters())
            + list(self.v_correction.parameters()),
            lr=self.hparams.lr,
            eps=1e-7,
        )
        return optimizer


# class SF2MLitModule(LightningModule):
#     """SF2M Module for training generative models and learning structure."""

#     def __init__(
#         self,
#         net: Any,
#         score_net: Any,
#         corr_net: Any,
#         optimizer: Any,
#         datamodule: LightningDataModule,
#         partial_solver: FlowSolver,
#         scheduler: Optional[Any] = None,
#         neural_ode: Optional[Any] = None,
#         ot_sampler: Optional[Union[str, Any]] = EntropicOTFM,
#         sigma: Optional[NoiseScheduler] = None,
#         sigma_min: float = 0.1,
#         alpha: float = 0.1,
#         corr_strength: float = 1e-3,
#         reg: float = 1e-4,
#         lr: float = 1e-3,
#         batch_size: int = 64,
#         skip_time=None,
#         avg_size: int = -1,
#         leaveout_timepoint: int = -1,
#         test_nfe: int = 100,
#         plot: bool = False,
#         nice_name: Optional[str] = "SF2M",
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters(
#             ignore=[
#                 "net",
#                 "corr_net",
#                 "score_net",
#                 "scheduler",
#                 "datamodule",
#                 "partial_solver",
#             ],
#             logger=False,
#         )

#         self.is_trajectory = False
#         if hasattr(datamodule, "IS_TRAJECTORY"):
#             self.is_trajectory = datamodule.IS_TRAJECTORY
#         if hasattr(datamodule, "dim"):
#             self.dim = datamodule.dim
#         elif hasattr(datamodule, "dims"):
#             self.dim = datamodule.dims
#         else:
#             raise NotImplementedError("Datamodule must have either dim or dims")
#         self.net = net
#         self.score_net = score_net
#         self.corr_net = corr_net

#         self.partial_solver = partial_solver
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.ot_sampler = ot_sampler
#         self.alpha = alpha
#         self.corr_strength = corr_strength
#         self.reg = reg
#         self.batch_size = batch_size
#         self.skip_time = skip_time
#         self.sigma = sigma
#         if sigma is None:
#             self.sigma = ConstantNoiseScheduler(sigma_min)
#         self.criterion = torch.nn.MSELoss()
#         self.otfms = []
#         self.cond_matrix = []
