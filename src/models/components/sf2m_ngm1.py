import glob
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.datamodules.components import sc_dataset as util
from src.datamodules.grn_datamodule import TrajectoryStructureDataModule
from src.models.components.base import MLPODEFKO
from src.models.components.bayesian_drift import BayesianDrift
from src.models.components.cond_mlp import MLP as CONDMLP
from src.models.components.optimal_transport import EntropicOTFM
from src.models.components.simple_mlp import MLP
from src.models.components.solver import TrajectorySolver, simulate_trajectory, wasserstein
from src.models.components.plotting import log_causal_graph_matrices
import sys, os, time, pathlib
from pathlib import Path
import wandb
print("TOP-LEVEL IMPORT OK, cwd =", os.getcwd(), file=sys.stderr)
sys.stderr.flush()

T = 5
dataset = "dyn-TF"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SF2MNGM(nn.Module):
    """Encapsulates the NGM training pipeline for flow+score matching in one PyTorch module."""

    def __init__(
        self,
        datamodule,
        T=10,
        sigma=1.0,
        dt=0.1,
        batch_size=64,
        alpha=0.5,
        dyn_alpha=0.1,
        dyn_hidden=4,
        reg=1e-5,
        correction_reg_strength=1e-3,
        n_steps=2000,
        lr=1e-3,
        device=None,
        GL_reg=0.04,
        knockout_hidden=100,
        score_hidden=[100, 100],
        correction_hidden=[64, 64],
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

        # Detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

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
        self.dyn_alpha = dyn_alpha
        self.dyn_hidden = dyn_hidden

        # -----------------------
        # 1. Load the data
        # -----------------------
        self.data_loader = datamodule
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        self.adatas = self.data_loader.adatas
        self.kos = self.data_loader.kos
        self.ko_indices = self.data_loader.ko_indices
        self.true_matrix = self.data_loader.true_matrix.values

        self.adatas = [self.adatas[0]] # DEBUG: use only first dataset

        # Example shape from the first dataset
        self.n_genes = self.adatas[0].X.shape[1]

        wandb.init(project="sf2m-grn", config=dict(T=self.T, d=self.n_genes, model="BayesianDrift"))
        wandb.define_metric("trainer/step")
        wandb.define_metric("traj/*", step_metric="trainer/step")
        wandb.define_metric("loss/*", step_metric="trainer/step")
        wandb.define_metric("grn/*",  step_metric="trainer/step")

        self.conditionals = []
        for i, ko_name in enumerate(self.kos):
            cond_matrix = torch.zeros(self.batch_size, self.n_genes)
            if ko_name is not None:
                cond_matrix[:, self.ko_indices[i]] = 1
            self.conditionals.append(cond_matrix)

        self.knockout_masks = []
        for i, adata in enumerate(self.adatas):
            d = adata.X.shape[1]
            mask_i = self.build_knockout_mask(d, self.ko_indices[i])
            self.knockout_masks.append(mask_i)

        self.dims = [self.n_genes, 100, 1]
        # self.func_v = MLPODEFKO(
        #     dims=self.dims, GL_reg=GL_reg, bias=True, knockout_masks=self.knockout_masks
        # ).to(self.device)
        self.func_v = BayesianDrift(
            dims=self.dims, 
            n_ens=100, 
            deepens=True, 
            time_invariant=True, 
            k_hidden=dyn_hidden,
            alpha=self.dyn_alpha,
            knockout_masks=self.knockout_masks,
            step=100,
            hyper="mlp",
        ).to(self.device)

        self.score_net = CONDMLP(
            d=self.n_genes,
            hidden_sizes=score_hidden,
            time_varying=True,
            conditional=True,
            conditional_dim=self.n_genes,
        ).to(self.device)

        self.v_correction = MLP(
            d=self.n_genes, hidden_sizes=correction_hidden, time_varying=True
        ).to(self.device)

        self.otfms = self.build_entropic_otfms(
            self.adatas, T=self.T, sigma=self.sigma, dt=self.dt
        )


        self.func_v.to(self.device)
        self.score_net.to(self.device)
        self.v_correction.to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.func_v.parameters())
            + list(self.score_net.parameters())
            + list(self.v_correction.parameters()),
            lr=self.lr,
        )

        # For tracking losses
        self.loss_history = []
        self.score_loss_history = []
        self.flow_loss_history = []
        self.reg_loss_history = []
        self.reg_corr_loss_history = []

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

    def train_model(self, skip_time=None):
        """Combine flow matching + score matching with multiple datasets."""
        # Just a shorthand
        func_v = self.func_v
        func_s = self.score_net
        v_correction = self.v_correction
        optim = self.optimizer
        LOG_CSV = Path("traj_metrics.csv")

        for i in tqdm(range(self.n_steps)):
            # Randomly pick which dataset to train on
            #ds_idx = np.random.randint(0, len(self.adatas))
            ds_idx = 0  # For debugging with single dataset
            model = self.otfms[ds_idx]
            cond_vector = self.conditionals[ds_idx].to(self.device)

            # Sample bridging flows
            _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
                batch_size=self.batch_size, skip_time=skip_time
            )
            _x = _x.to(self.device)
            _s = _s.to(self.device)
            _u = _u.to(self.device)
            _t = _t.to(self.device)
            _t_orig = _t_orig.to(self.device)

            optim.zero_grad()

            s_input = _x.unsqueeze(1)
            v_input = _x.unsqueeze(1)
            t_input = _t.unsqueeze(1)
            B = _x.shape[0]

            cond_expanded = cond_vector.repeat(B // cond_vector.shape[0] + 1, 1)[:B]

            # Score net output
            s_fit = func_s(_t, _x, cond_expanded).squeeze(1)

            # Flow net output
            v_fit = func_v(t_input, v_input, ds_idx, step=i).squeeze(1) - (
                model.sigma**2 / 2
            ) * func_s(_t, _x, cond_expanded)

            # Losses
            L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
            L_flow = torch.mean((v_fit - (_u / self.dt)) ** 2)
            L_reg = func_v.l2_reg() + func_v.l1_reg()

            if i < 100:
                L = self.alpha * L_score
            else:
                L = self.alpha * L_score + (1 - self.alpha) * L_flow + self.reg * L_reg

            # Bookkeeping
            self.loss_history.append(L.item())
            self.score_loss_history.append(L_score.item())
            self.flow_loss_history.append(L_flow.item())
            self.reg_loss_history.append(L_reg.item())
            wandb.log({
                "trainer/step": i,
                "loss/train": float(L.item()),
                "loss/score": float(L_score.item()),
                "loss/flow": float(L_flow.item()),
            }, step=i)

            if i % 100 == 0:
                print(
                    f"Step={i}, ds={ds_idx}, "
                    f"L_score={L_score.item():.4f}, L_flow={L_flow.item():.4f}, "
                    f"Reg(Flow)={L_reg.item():.4f}"
                )
                sys.stdout.flush()

            if i % 500 == 0:
                with torch.no_grad():
                    per_time = []  # list of {'Time': t, 'Avg ODE': float, 'Avg SDE': float}

                    for t_bin in range(1, self.T):
                        per_ds = []
                        for ds_idx, adata in enumerate(self.adatas):
                            x0 = torch.from_numpy(adata.X[adata.obs["t"] == t_bin - 1]).float()
                            true_dist = torch.from_numpy(adata.X[adata.obs["t"] == t_bin]).float()

                            if len(x0) == 0 or len(true_dist) == 0:
                                continue

                            dev = next(self.func_v.parameters()).device
                            x0 = x0.to(dev)
                            true_dist = true_dist.to(dev)

                            cond_vector = self.conditionals[ds_idx]
                            if cond_vector is not None:
                                cond_vector = cond_vector[0].repeat(len(x0), 1).to(dev)

                            traj_ode = simulate_trajectory(
                                self.func_v, self.v_correction, self.score_net,
                                x0, dataset_idx=None, start_time=t_bin - 1, end_time=t_bin,
                                n_times=min(len(x0), len(true_dist)),
                                cond_vector=cond_vector, device=dev,
                            )
                            traj_sde = simulate_trajectory(
                                self.func_v, self.v_correction, self.score_net,
                                x0, dataset_idx=None, start_time=t_bin - 1, end_time=t_bin,
                                n_times=min(len(x0), len(true_dist)),
                                cond_vector=cond_vector, use_sde=True, device=dev,
                            )

                            pred_ode = traj_ode[-1].detach().cpu()
                            pred_sde = traj_sde[-1].detach().cpu()
                            td_cpu   = true_dist.detach().cpu()

                            w_dist_ode = wasserstein(pred_ode, td_cpu)
                            w_dist_sde = wasserstein(pred_sde, td_cpu)
                            per_ds.append((w_dist_ode, w_dist_sde))

                        if per_ds:
                            avg_ode = float(np.mean([a for a, _ in per_ds]))
                            avg_sde = float(np.mean([b for _, b in per_ds]))
                        else:
                            avg_ode = np.nan
                            avg_sde = np.nan

                        per_time.append({"Time": t_bin, "Avg ODE": avg_ode, "Avg SDE": avg_sde})
                        print(f"  Time {t_bin}: Avg ODE W={avg_ode}, Avg SDE W={avg_sde}")
                        sys.stdout.flush()

                    step_rows = []
                    for row in per_time:
                        t_bin = row["Time"]
                        step_rows.append({"Step": i, "Time": t_bin, "Kind": "ODE", "Value": row["Avg ODE"]})
                        step_rows.append({"Step": i, "Time": t_bin, "Kind": "SDE", "Value": row["Avg SDE"]})
                    df_step = pd.DataFrame(step_rows)
                    write_header = not LOG_CSV.exists()
                    df_step.to_csv(LOG_CSV, mode="a", header=write_header, index=False)

                    if wandb.run is not None:
                        log = {"trainer/step": i}
                        ode_vals, sde_vals = [], []

                        for row in per_time:
                            t = row["Time"]
                            v_ode = row["Avg ODE"]
                            v_sde = row["Avg SDE"]

                            if v_ode is not None and not np.isnan(v_ode):
                                log[f"traj/ODE/t{t}"] = float(v_ode)
                                ode_vals.append(v_ode)
                            if v_sde is not None and not np.isnan(v_sde):
                                log[f"traj/SDE/t{t}"] = float(v_sde)
                                sde_vals.append(v_sde)

                        if ode_vals:
                            log["traj/ODE/mean"] = float(np.mean(ode_vals))
                        if sde_vals:
                            log["traj/SDE/mean"] = float(np.mean(sde_vals))

                        wandb.log(log, step=i, commit=False)  # CHANGE: staged log

                        # KEEP (graph logging)
                        W_v = self.func_v.get_structure()
                        if W_v.ndim == 3:
                            W_v = W_v.mean(axis=0)
                        A_true = self.true_matrix
                        log_causal_graph_matrices(None, W_v, A_true, logger=None, global_step=i)

                        # NEW: ensure one final commit after the staged logs
                        wandb.log({}, step=i, commit=True)     # NEW

            L.backward()
            optim.step()

        print("Training complete.")

        if wandb.run is not None:
            try:
                last_ode_mean = float(np.nanmean([r["Avg ODE"] for r in per_time]))
            except Exception:
                last_ode_mean = float(np.nan)
            wandb.run.summary["traj/ODE/mean"] = last_ode_mean

    def forward(self, t, x):
        """
        Optionally define a forward pass if you want to do something
        like: output = model(t, x). Typically used in standard nn.Modules.
        """
        # Example: direct call to your flow function
        # This can be adapted to your needs.
        t_input = t.unsqueeze(1)
        x_input = x.unsqueeze(1)
        out = self.func_v(t_input, x_input)
        return out.squeeze(1)


def main():
    print("ENTERED main()", file=sys.stderr)
    sys.stderr.flush()
    time.sleep(2)

    model = SF2MNGM(
        datamodule=TrajectoryStructureDataModule(),
        T=5,
        sigma=1.0,
        dt=0.2,
        batch_size=164,
        alpha=0.1,
        dyn_alpha=0.1,
        reg=0,
        correction_reg_strength=1e-3,
        n_steps=15000,
        lr=1e-4,
        device=None  # Auto-detect
    )

    model.train_model(skip_time=None)
    n=8

    def maskdiag(A):
        return A * (1 - np.eye(n))

    import matplotlib.pyplot as plt

    # with torch.no_grad():
    #     A_estim = compute_global_jacobian(model.func_v, model.adatas, dt=1 / T, device=torch.device("cpu"))

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def maskdiag_np(A):
        A = to_numpy(A)
        n = A.shape[0]
        return A * (1 - np.eye(n, dtype=A.dtype))

    # plot loss history
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history, label="Total Loss")
    plt.plot(model.score_loss_history, label="Score Loss")
    plt.plot(model.flow_loss_history, label="Flow Loss")
    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_history.png")
    plt.show()

    W_v = model.func_v.get_structure()

    if W_v.ndim == 3:
        W_v = W_v[0]
    A_true = to_numpy(model.true_matrix)

    np.savetxt("A_estim_mlpoef.txt", maskdiag_np(W_v), fmt="%.6f")

    # Display both the estimated adjacency matrix and the learned causal graph
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 2)
    plt.imshow(maskdiag_np(W_v.T), cmap="Reds")
    plt.gca().invert_yaxis()
    plt.title("Causal Graph (from MLPODEF)")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(maskdiag(A_true), vmin=-1, vmax=1, cmap="RdBu_r")
    plt.gca().invert_yaxis()
    plt.title("A_true")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("causal_graph_mlpodef.png")
    plt.show()

    from sklearn.metrics import precision_recall_curve, average_precision_score
    plt.figure(figsize=(12, 5))
    y_true = np.abs(np.sign(maskdiag(A_true)).astype(int).flatten())
    # For MLPODEF-based estimation
    plt.subplot(1, 2, 2)
    y_pred_mlp = np.abs(maskdiag_np(W_v).flatten())
    prec, rec, thresh = precision_recall_curve(y_true, y_pred_mlp)
    avg_prec_mlp = average_precision_score(y_true, y_pred_mlp)
    plt.plot(rec, prec, label=f"MLPODEF-based (AP = {avg_prec_mlp:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve (MLPODEF)\nAUPR ratio = {avg_prec_mlp/np.mean(np.abs(A_true) > 0)}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pr_curve_mlpoef.png")
    plt.show()

if __name__ == "__main__":
    main()
