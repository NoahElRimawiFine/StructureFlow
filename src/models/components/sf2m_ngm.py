import copy
import glob
import importlib
import os
import random
import sys

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
from lightning.pytorch import Trainer, seed_everything
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint
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


class DataLoader:
    def __init__(self, data_path, dataset_type="Synthetic"):
        """Initialize DataLoader.

        Args:
            data_path: Path to data directory
            dataset_type: Either "Synthetic" or "Curated"
        """
        self.data_path = os.path.join(data_path, dataset_type)
        self.dataset_type = dataset_type
        self.adatas = None
        self.kos = None
        self.true_matrix = None

    def load_data(self):
        """Load and preprocess data."""
        if self.dataset_type == "Synthetic":
            paths = glob.glob(os.path.join(self.data_path, f"{dataset}/{dataset}*-1")) + glob.glob(
                os.path.join(self.data_path, f"{dataset}_ko*/{dataset}*-1")
            )
        elif self.dataset_type == "Curated":
            paths = glob.glob(os.path.join(self.data_path, "HSC*/HSC*-1"))
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.adatas = [util.load_adata(p) for p in paths]

        df = pd.read_csv(os.path.join(os.path.dirname(paths[0]), "refNetwork.csv"))

        n_genes = self.adatas[0].n_vars

        # Create empty matrix with gene names
        self.true_matrix = pd.DataFrame(
            np.zeros((n_genes, n_genes), int),
            index=self.adatas[0].var.index,
            columns=self.adatas[0].var.index,
        )

        # Fill matrix with interaction values
        for i in range(df.shape[0]):
            _i = df.iloc[i, 1]  # target gene
            _j = df.iloc[i, 0]  # source gene
            _v = {"+": 1, "-": -1}[df.iloc[i, 2]]  # interaction type
            self.true_matrix.loc[_i, _j] = _v

        # Bin timepoints
        t_bins = np.linspace(0, 1, T + 1)[:-1]
        for adata in self.adatas:
            adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1

        # Get knockouts
        self.kos = []
        for p in paths:
            try:
                self.kos.append(os.path.basename(p).split("_ko_")[1].split("-")[0])
            except (IndexError, ValueError, AttributeError):
                self.kos.append(None)

        self.gene_to_index = {gene: idx for idx, gene in enumerate(self.adatas[0].var.index)}
        self.ko_indices = []
        for ko in self.kos:
            if ko is None:
                self.ko_indices.append(None)
            else:
                self.ko_indices.append(self.gene_to_index[ko])


def build_knockout_mask(d, ko_idx):
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


def build_entropic_otfms(adatas, T, sigma, dt):
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
            device=torch.device("cpu"),
        )
        otfms.append(model)
    return otfms


def train_with_fmot_scorematching(
    func_v,
    func_s,
    v_correction,
    adatas_list,
    otfms,
    cond_matrix,
    alpha=0.5,
    reg=1e-5,
    n_steps=2000,
    batch_size=64,
    correction_reg_strength=1e-3,
    device="cpu",
    lr=1e-3,
    true_mat=None,
    skip_time=None,
):
    """Combine flow matching + score matching with multiple datasets."""
    func_v.to(device)
    func_s.to(device)
    optim = torch.optim.AdamW(
        list(func_v.parameters()) + list(func_s.parameters()) + list(v_correction.parameters()),
        lr=lr,
    )

    loss_history = []
    score_loss_history = []
    flow_loss_history = []
    reg_loss_history = []
    reg_corr_loss_history = []

    def proximal(w, dims, lam=0.1, eta=0.1):
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    def mlp_l2_reg(mlp):
        l2_sum = 0.0
        for param in mlp.parameters():
            l2_sum += torch.sum(param**2)
        return l2_sum

    for i in tqdm(range(n_steps)):
        ds_idx = np.random.randint(0, len(adatas_list))
        model = otfms[ds_idx]
        cond_vector = cond_matrix[ds_idx]

        _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
            batch_size=batch_size, skip_time=skip_time
        )
        if i == 0:
            sample_data = {
                "x": _x.detach().cpu().numpy(),
                "s": _s.detach().cpu().numpy(),
                "u": _u.detach().cpu().numpy(),
                "t": _t.detach().cpu().numpy(),
                "t_orig": _t_orig.detach().cpu().numpy(),
            }
            import pandas as pd

            # Save each variable to its own CSV file.
            for key, value in sample_data.items():
                # If the array is more than 2D, you might need to reshape or flatten.
                # For example, if value is 3D, you could flatten the last two dimensions.
                if value.ndim > 2:
                    # This is just an example; adjust the reshaping logic as needed.
                    value = value.reshape(value.shape[0], -1)
                df = pd.DataFrame(value)
                df.to_csv(f"{key}_step0.csv", index=False)

        optim.zero_grad()

        # Reshape inputs for MLPODEF
        s_input = _x.unsqueeze(1)
        v_input = _x.unsqueeze(1)
        t_input = _t.unsqueeze(1)

        B = _x.shape[0]
        cond_expanded = cond_vector.repeat(B // cond_vector.shape[0] + 1, 1)[:B]

        # Get model outputs and reshape
        s_fit = func_s(_t, _x, cond_expanded).squeeze(1)
        # v_fit = v(t_input, v_input).squeeze(1)
        if i <= 500:
            v_fit = func_v(t_input, v_input).squeeze(1) - model.sigma**2 / 2 * func_s(
                _t, _x, cond_expanded
            )
        else:
            v_fit = func_v(t_input, v_input).squeeze(1) + v_correction(_t, _x)
            v_fit = v_fit - model.sigma**2 / 2 * func_s(_t, _x, cond_expanded)

        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * model.dt - _u) ** 2)

        L_reg = func_v.l2_reg() + func_v.fc1_reg()
        L_reg_correction = mlp_l2_reg(v_correction)
        if i < 100:  # train score for first few iters
            L = alpha * L_score
        elif i >= 100 and i <= 500:
            L = alpha * L_score + (1 - alpha) * L_flow + reg * L_reg
        else:
            L = (
                alpha * L_score
                + (1 - alpha) * L_flow
                + reg * L_reg
                + correction_reg_strength * L_reg_correction
            )

        with torch.no_grad():
            if i % 100 == 0:
                print(
                    f"step={i}, dataset={ds_idx}, L_score={L_score.item():.4f}, L_flow={L_flow.item():.4f}, "
                    f"NGM_Reg={L_reg.item():.4f}, MLP_Reg={L_reg_correction.item():.4f}"
                )
            loss_history.append(L.item())
            score_loss_history.append(L_score.item())
            flow_loss_history.append(L_flow.item())
            reg_loss_history.append(L_reg.item())
            reg_corr_loss_history.append(L_reg_correction.item())

        L.backward()
        optim.step()

        # proximal(s.fc1.weight, s.dims, lam=s.GL_reg, eta=0.01)
        proximal(func_v.fc1.weight, func_v.dims, lam=func_v.GL_reg, eta=0.01)

        if i % 1000 == 0:
            print(
                f"Step={i}, dataset={ds_idx}, L_score={L_score.item():.4f}, "
                f"L_flow={L_flow.item():.4f}, L_reg={L_reg:.4f}"
            )

    # plt.plot(loss_history)
    # plt.title("Score+Flow Matching Loss")
    # plt.xlabel("training step")
    # plt.ylabel("loss")
    # plt.show()

    return (
        loss_history,
        score_loss_history,
        flow_loss_history,
        reg_loss_history,
        reg_corr_loss_history,
        func_v,
        func_s,
        v_correction,
    )


def main():
    data_loader = DataLoader("data/", dataset_type="Synthetic")
    data_loader.load_data()
    adatas, kos, ko_indices, true_matrix = (
        data_loader.adatas,
        data_loader.kos,
        data_loader.ko_indices,
        data_loader.true_matrix.values,
    )
    batch_size = 164
    n = adatas[0].X.shape[1]

    # want to create a [8, n, 8] matrix that is one hot encoded and will be selected depending on dataset idx
    conditionals = []
    for i, ad in enumerate(kos):
        cond_matrix = torch.zeros(batch_size, n)
        if ad is not None:
            cond_matrix[:, i] = 1
        conditionals.append(cond_matrix)

    knockout_masks = []
    for i, ad in enumerate(adatas):
        d = ad.X.shape[1]
        mask_i = build_knockout_mask(d, ko_indices[i])  # returns [d,d]
        knockout_masks.append(mask_i)
    print(knockout_masks)

    wt_idx = [i for i, ko in enumerate(kos) if ko is None]
    ko_idx = [i for i, ko in enumerate(kos) if ko is not None]
    adatas_wt = [adatas[i] for i in wt_idx]
    adatas_ko = [adatas[i] for i in ko_idx]
    dims = [n, 100, 1]
    t = adatas[0].obs["t"].max()

    func_v = MLPODEFKO(dims=dims, GL_reg=0.04, bias=True, knockout_masks=knockout_masks)
    score_net = CONDMLP(
        d=n,
        hidden_sizes=[100, 100],
        time_varying=True,
        conditional=True,
        conditional_dim=n,
    )

    v_cor = MLP(d=n, hidden_sizes=[64, 64], time_varying=True)
    # score_net = fm.MLP(d=n, hidden_sizes = [64, 64], time_varying=True)

    otfms = build_entropic_otfms(adatas, T, sigma=1.0, dt=1 / T)

    (
        loss_history,
        score_loss_history,
        flow_loss_history,
        reg_loss_history,
        reg_corr_loss_history,
        flow_model,
        corr_model,
        score_model,
    ) = train_with_fmot_scorematching(
        func_v=func_v,
        func_s=score_net,
        v_correction=v_cor,
        adatas_list=adatas,
        otfms=otfms,
        cond_matrix=conditionals,
        alpha=0.1,
        reg=5e-6,
        n_steps=15000,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-3,
        true_mat=true_matrix,
    )

    def maskdiag(A):
        return A * (1 - np.eye(n))

    def compute_global_jacobian(v, adatas, dt, device=torch.device("cpu")):
        """Compute a single adjacency from a big set of states across all datasets.

        Returns a [d, d] numpy array representing an average Jacobian.
        """

        all_x_list = []
        for ds_idx, adata in enumerate(adatas):
            x0 = adata.X[adata.obs["t"] == 0]
            all_x_list.append(x0)
        if len(all_x_list) == 0:
            return None

        X_all = np.concatenate(all_x_list, axis=0)
        if X_all.shape[0] == 0:
            return None

        X_all_torch = torch.from_numpy(X_all).float().to(device)

        def get_flow(t, x):
            x_input = x.unsqueeze(0).unsqueeze(0)
            t_input = t.unsqueeze(0).unsqueeze(0)
            return v(t_input, x_input).squeeze(0).squeeze(0)

        # Or loop over multiple times if the model is time-varying
        t_val = torch.tensor(0.0).to(device)

        Ju = torch.func.jacrev(get_flow, argnums=1)

        Js = []

        batch_size = 256
        for start in range(0, X_all_torch.shape[0], batch_size):
            end = start + batch_size
            batch_x = X_all_torch[start:end]

            J_local = torch.vmap(lambda x: Ju(t_val, x))(batch_x)
            J_avg = J_local.mean(dim=0)
            Js.append(J_avg)

        if len(Js) == 0:
            return None
        J_final = torch.stack(Js, dim=0).mean(dim=0)

        A_est = J_final

        return A_est.detach().cpu().numpy().T

    with torch.no_grad():
        A_estim = compute_global_jacobian(func_v, adatas, dt=1 / T, device=torch.device("cpu"))

    W_v = func_v.causal_graph(w_threshold=0.0).T
    A_true = true_matrix

    # Display both the estimated adjacency matrix and the learned causal graph
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(maskdiag(A_estim), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    plt.gca().invert_yaxis()
    plt.title("A_estim (from Jacobian)")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(maskdiag(W_v), cmap="Reds")
    plt.gca().invert_yaxis()
    plt.title("Causal Graph (from MLPODEF)")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(maskdiag(A_true), vmin=-1, vmax=1, cmap="RdBu_r")
    plt.gca().invert_yaxis()
    plt.title("A_true")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    maskdiag(W_v)

    plt.figure(figsize=(12, 5))
    # For Jacobian-based estimation
    plt.subplot(1, 2, 1)
    y_true = np.abs(np.sign(maskdiag(A_true)).astype(int).flatten())
    y_pred = np.abs(maskdiag(A_estim).flatten())
    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)
    plt.plot(rec, prec, label=f"Jacobian-based (AP = {avg_prec:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve (Jacobian)\nAUPR ratio = {avg_prec/np.mean(np.abs(A_true) > 0)}"
    )
    plt.legend()
    plt.grid(True)
    # For MLPODEF-based estimation
    plt.subplot(1, 2, 2)
    y_pred_mlp = np.abs(maskdiag(W_v).flatten())
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
    plt.show()


if __name__ == "__main__":
    main()
