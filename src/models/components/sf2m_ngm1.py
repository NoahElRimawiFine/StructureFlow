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
from src.models.components.base import MLPODEFKO
from src.models.components.cond_mlp import MLP as CONDMLP
from src.models.components.distribution_distances import compute_distribution_distances
from src.models.components.optimal_transport import EntropicOTFM
from src.models.components.simple_mlp import MLP

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

        # Move models to device
        self.func_v.to(self.device)
        self.score_net.to(self.device)
        self.v_correction.to(self.device)

        # -----------------------
        # 6. Setup optimizer
        # -----------------------
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
    def train_model(self, skip_time=None):
        """Combine flow matching + score matching with multiple datasets."""
        # Just a shorthand
        func_v = self.func_v
        func_s = self.score_net
        v_correction = self.v_correction
        optim = self.optimizer

        for i in tqdm(range(self.n_steps)):
            # Randomly pick which dataset to train on
            ds_idx = np.random.randint(0, len(self.adatas))
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

            # Prepare inputs
            s_input = _x.unsqueeze(1)
            v_input = _x.unsqueeze(1)
            t_input = _t.unsqueeze(1)
            B = _x.shape[0]

            # Expand conditional vectors to match batch size
            cond_expanded = cond_vector.repeat(B // cond_vector.shape[0] + 1, 1)[:B]

            # Score net output
            s_fit = func_s(_t, _x, cond_expanded).squeeze(1)

            # Flow net output, with or without correction
            if i <= 500:
                # Warmup phase
                v_fit = func_v(t_input, v_input).squeeze(1) - (model.sigma**2 / 2) * func_s(
                    _t, _x, cond_expanded
                )
            else:
                # Full training phase with correction
                v_fit = func_v(t_input, v_input).squeeze(1) + v_correction(_t, _x)
                v_fit = v_fit - (model.sigma**2 / 2) * func_s(_t, _x, cond_expanded)

            # Losses
            L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
            L_flow = torch.mean((v_fit * model.dt - _u) ** 2)
            L_reg = func_v.l2_reg() + func_v.fc1_reg()
            L_reg_correction = self.mlp_l2_reg(v_correction)

            if i < 100:
                # Train only score initially
                L = self.alpha * L_score
            elif i <= 500:
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

            # Bookkeeping
            self.loss_history.append(L.item())
            self.score_loss_history.append(L_score.item())
            self.flow_loss_history.append(L_flow.item())
            self.reg_loss_history.append(L_reg.item())
            self.reg_corr_loss_history.append(L_reg_correction.item())

            if i % 100 == 0:
                print(
                    f"Step={i}, ds={ds_idx}, "
                    f"L_score={L_score.item():.4f}, L_flow={L_flow.item():.4f}, "
                    f"Reg(Flow)={L_reg.item():.4f}, Reg(Corr)={L_reg_correction.item():.4f}"
                )

            # Backprop and update
            L.backward()
            optim.step()

            # Proximal step (group-lasso style)
            self.proximal(func_v.fc1.weight, func_v.dims, lam=func_v.GL_reg, eta=0.01)

        print("Training complete.")

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


# def main():
#     model = SF2MNGM(
#         data_path="data/",
#         dataset_type="Synthetic",
#         T=5,
#         sigma=1.0,
#         dt=0.2,
#         batch_size=164,
#         alpha=0.1,
#         reg=1e-5,
#         correction_reg_strength=1e-3,
#         n_steps=15000,
#         lr=3e-3,
#         device=None  # Auto-detect
#     )

#     model.train_model(skip_time=None)
#     n=8

#     def maskdiag(A):
#         return A * (1 - np.eye(n))

#     import matplotlib.pyplot as plt

#     def compute_global_jacobian(v, adatas, dt, device=torch.device("cpu")):
#         """Compute a single adjacency from a big set of states across all datasets.

#         Returns a [d, d] numpy array representing an average Jacobian.
#         """

#         all_x_list = []
#         for ds_idx, adata in enumerate(adatas):
#             x0 = adata.X[adata.obs["t"] == 0]
#             all_x_list.append(x0)
#         if len(all_x_list) == 0:
#             return None

#         X_all = np.concatenate(all_x_list, axis=0)
#         if X_all.shape[0] == 0:
#             return None

#         X_all_torch = torch.from_numpy(X_all).float().to(device)

#         def get_flow(t, x):
#             x_input = x.unsqueeze(0).unsqueeze(0)
#             t_input = t.unsqueeze(0).unsqueeze(0)
#             return v(t_input, x_input).squeeze(0).squeeze(0)

#         # Or loop over multiple times if the model is time-varying
#         t_val = torch.tensor(0.0).to(device)

#         Ju = torch.func.jacrev(get_flow, argnums=1)

#         Js = []

#         batch_size = 256
#         for start in range(0, X_all_torch.shape[0], batch_size):
#             end = start + batch_size
#             batch_x = X_all_torch[start:end]

#             J_local = torch.vmap(lambda x: Ju(t_val, x))(batch_x)
#             J_avg = J_local.mean(dim=0)
#             Js.append(J_avg)

#         if len(Js) == 0:
#             return None
#         J_final = torch.stack(Js, dim=0).mean(dim=0)

#         A_est = J_final

#         return A_est.detach().cpu().numpy().T

#     with torch.no_grad():
#         A_estim = compute_global_jacobian(model.func_v, model.adatas, dt=1 / T, device=torch.device("cpu"))

#     W_v = model.func_v.causal_graph(w_threshold=0.0).T
#     A_true = model.true_matrix

#     # Display both the estimated adjacency matrix and the learned causal graph
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(maskdiag(A_estim), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
#     plt.gca().invert_yaxis()
#     plt.title("A_estim (from Jacobian)")
#     plt.colorbar()
#     plt.subplot(1, 3, 2)
#     plt.imshow(maskdiag(W_v), cmap="Reds")
#     plt.gca().invert_yaxis()
#     plt.title("Causal Graph (from MLPODEF)")
#     plt.colorbar()
#     plt.subplot(1, 3, 3)
#     plt.imshow(maskdiag(A_true), vmin=-1, vmax=1, cmap="RdBu_r")
#     plt.gca().invert_yaxis()
#     plt.title("A_true")
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()

#     maskdiag(W_v)

#     from sklearn.metrics import precision_recall_curve, average_precision_score
#     plt.figure(figsize=(12, 5))
#     # For Jacobian-based estimation
#     plt.subplot(1, 2, 1)
#     y_true = np.abs(np.sign(maskdiag(A_true)).astype(int).flatten())
#     y_pred = np.abs(maskdiag(A_estim).flatten())
#     prec, rec, thresh = precision_recall_curve(y_true, y_pred)
#     avg_prec = average_precision_score(y_true, y_pred)
#     plt.plot(rec, prec, label=f"Jacobian-based (AP = {avg_prec:.2f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(
#         f"Precision-Recall Curve (Jacobian)\nAUPR ratio = {avg_prec/np.mean(np.abs(A_true) > 0)}"
#     )
#     plt.legend()
#     plt.grid(True)
#     # For MLPODEF-based estimation
#     plt.subplot(1, 2, 2)
#     y_pred_mlp = np.abs(maskdiag(W_v).flatten())
#     prec, rec, thresh = precision_recall_curve(y_true, y_pred_mlp)
#     avg_prec_mlp = average_precision_score(y_true, y_pred_mlp)
#     plt.plot(rec, prec, label=f"MLPODEF-based (AP = {avg_prec_mlp:.2f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(
#         f"Precision-Recall Curve (MLPODEF)\nAUPR ratio = {avg_prec_mlp/np.mean(np.abs(A_true) > 0)}"
#     )
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
