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
from torch.utils.data import DataLoader, Dataset, random_split
from torchdiffeq import odeint
from tqdm import tqdm

from src.datamodules.components import sc_dataset as util
from src.models.components.base import MLPODEFKO
from src.models.components.cond_mlp import MLP as CONDMLP
from src.models.components.distribution_distances import compute_distribution_distances
from src.models.components.optimal_transport import EntropicOTFM
from src.models.components.simple_mlp import MLP

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(seed, workers=True)


def build_knockout_mask(dim: int, ko_idx: int):
    """Build a [dim, dim] knockout mask.

    For a knockout of gene with index ko_idx, set the entire column to zero except the diagonal.
    For wild-type (ko_idx is None), return a matrix of ones.
    """
    mask = torch.ones((dim, dim), dtype=torch.float32)
    if ko_idx is not None:
        mask[:, ko_idx] = 0.0
        mask[ko_idx, ko_idx] = 1.0
    return mask


class AnnDataDataset(Dataset):
    """
    Wraps an AnnData object so that each sample is a dictionary with:
      - "X": the cell's expression data (as a torch tensor)
      - "t": the cell's pseudo-time (as a torch tensor)
      - "source_id": an identifier for the data source (optional)
    """

    def __init__(self, adata, source_id: int = None):
        self.adata = adata
        self.source_id = source_id
        # Convert expression matrix X to a dense torch tensor if necessary.
        if hasattr(adata.X, "toarray"):
            self.X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            self.X = torch.tensor(adata.X, dtype=torch.float32)
        # Assume pseudo-time is stored in adata.obs["t"]
        self.t = torch.tensor(adata.obs["t"].values, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx], "t": self.t[idx], "source_id": self.source_id}


class TrajectoryStructureDataModule(pl.LightningDataModule):
    """LightningDataModule for GRN structure inference.

    Loads data from disk, preprocesses (including binning time and computing knockout info), builds
    a ConcatDataset of AnnDataDataset objects, and splits into train/val/test.
    """

    def __init__(
        self,
        data_path: str = "data/",
        dataset: str = "dyn-TF",
        dataset_type: str = "Synthetic",
        batch_size: int = 64,
        num_workers: int = 4,
        train_val_test_split: tuple = (
            1.0,
            0.0,
            0.0,
        ),  # Use entire dataset for training for debugging
        T: int = 5,
    ):
        super().__init__()
        self.data_path = os.path.join(data_path, dataset_type)
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.T = T

        # Attributes to be filled in setup:
        self.adatas = None
        self.kos = None
        self.ko_indices = None
        self.true_matrix = None
        self.dim = None

        self._full_dataset = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.automatic_optimization = False

    def prepare_data(self):
        # If data download or extraction were needed, handle it here.
        pass

    def setup(self, stage=None):
        # Load and preprocess the data for training (or testing)
        if stage == "fit" or stage is None:
            paths = []
            if self.dataset_type == "Synthetic":
                paths = glob.glob(
                    os.path.join(self.data_path, f"{self.dataset}/{self.dataset}*-1")
                ) + glob.glob(
                    os.path.join(self.data_path, f"{self.dataset}_ko*/{self.dataset}*-1")
                )
            elif self.dataset_type == "Curated":
                paths = glob.glob(os.path.join(self.data_path, "HSC*/HSC*-1"))
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")

            self.adatas = [util.load_adata(p) for p in paths]

            # Build the true network matrix from a CSV.
            df = pd.read_csv(os.path.join(os.path.dirname(paths[0]), "refNetwork.csv"))
            n_genes = self.adatas[0].n_vars
            self.dim = n_genes
            self.true_matrix = pd.DataFrame(
                np.zeros((n_genes, n_genes), int),
                index=self.adatas[0].var.index,
                columns=self.adatas[0].var.index,
            )
            for i in range(df.shape[0]):
                _i = df.iloc[i, 1]  # target gene
                _j = df.iloc[i, 0]  # source gene
                _v = {"+": 1, "-": -1}[df.iloc[i, 2]]
                self.true_matrix.loc[_i, _j] = _v

            # Bin timepoints.
            t_bins = np.linspace(0, 1, self.T + 1)[:-1]
            for adata in self.adatas:
                adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1

            # Identify knockouts.
            self.kos = []
            for p in paths:
                try:
                    self.kos.append(os.path.basename(p).split("_ko_")[1].split("-")[0])
                except IndexError:
                    self.kos.append(None)

            self.gene_to_index = {gene: idx for idx, gene in enumerate(self.adatas[0].var.index)}
            self.ko_indices = []
            for ko in self.kos:
                if ko is None:
                    self.ko_indices.append(None)
                else:
                    self.ko_indices.append(self.gene_to_index[ko])

            # Build a ConcatDataset from AnnDataDataset wrappers.
            from torch.utils.data import ConcatDataset

            wrapped_datasets = [
                AnnDataDataset(adata, source_id=i) for i, adata in enumerate(self.adatas)
            ]
            self._dataset_lengths = [len(ds) for ds in wrapped_datasets]
            self._full_dataset = ConcatDataset(wrapped_datasets)

            self.dataset_train = self._full_dataset  # only reason for this is to debug
            self.dataset_val = None
            self.dataset_test = None

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader([], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader([], batch_size=self.batch_size)

    def get_subset_adatas(self, split: str = "train"):
        """Returns a list of AnnData objects corresponding to the cells in the specified split."""
        # For debugging, return entire set.
        return self.adatas


class SF2MLitModule(pl.LightningModule):
    def __init__(
        self,
        net,
        score_net,
        corr_net,
        alpha=0.1,
        corr_strength=1e-3,
        reg=5e-6,
        lr=3e-3,
        batch_size=64,
        skip_time=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "score_net", "corr_net"])
        self.net = net
        self.score_net = score_net
        self.corr_net = corr_net
        self.alpha = alpha
        self.corr_strength = corr_strength
        self.reg = reg
        self.lr = lr
        self.batch_size = batch_size
        self.skip_time = skip_time

        self.otfms = []
        self.cond_matrix = []

        self.automatic_optimization = False

    def mlp_l2_reg(self, mlp):
        """Compute L2 regularization for all parameters in an MLP."""
        l2_sum = 0.0
        for param in mlp.parameters():
            l2_sum += torch.sum(param**2)
        return l2_sum

    def proximal(self, w, dims, lam=0.1, eta=0.1):
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    def on_fit_start(self):
        print("on_fit_start: Building OT flow models and conditionals.")
        print("Existing otfms:", getattr(self, "otfms", None))
        print("Existing cond_matrix:", getattr(self, "cond_matrix", None))

    def forward(self, x, t, cond=None):
        v = self.net(t, x) + self.corr_net(t, x)
        s = self.score_net(t, x, cond)
        return v, s

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        # (1) Randomly select a dataset index.
        ds_idx = np.random.randint(0, len(self.trainer.datamodule.adatas))
        otfm_model = self.otfms[ds_idx]
        cond_vector = self.cond_matrix[ds_idx]

        # (2) Sample bridging flows.
        _x, _s, _u, _t, _t_orig = otfm_model.sample_bridges_flows(
            batch_size=self.batch_size, skip_time=self.skip_time
        )
        optimizer.zero_grad()

        # (3) Move data to the current device.
        _x = _x.to(self.device)
        _s = _s.to(self.device)
        _u = _u.to(self.device)
        _t = _t.to(self.device)
        _t_orig = _t_orig.to(self.device)

        # (4) Expand the conditional vector.
        B = _x.shape[0]
        cond_expanded = cond_vector.repeat(B // self.batch_size + 1, 1)[:B].to(self.device)

        # (5) Reshape inputs: unsqueeze to add a time dimension.
        s_input = _x.unsqueeze(1)
        v_input = _x.unsqueeze(1)
        t_input = _t.unsqueeze(1)

        # (6) Compute network outputs.
        s_fit = self.score_net(_t, _x, cond_expanded).squeeze(1)
        if self.global_step <= 500:
            v_fit = self.net(t_input, v_input).squeeze(
                1
            ) - otfm_model.sigma**2 / 2 * self.score_net(_t, _x, cond_expanded)
        else:
            v_fit = self.net(t_input, v_input).squeeze(1) + self.corr_net(_t, _x).squeeze(1)
            v_fit = v_fit - otfm_model.sigma**2 / 2 * self.score_net(_t, _x, cond_expanded)

        # (7) Compute loss components.
        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * otfm_model.dt - _u) ** 2)
        L_reg = self.net.l2_reg() + self.net.fc1_reg()
        L_reg_corr = self.mlp_l2_reg(self.corr_net)

        if self.global_step < 100:
            loss = self.alpha * L_score
        elif self.global_step <= 500:
            loss = self.alpha * L_score + (1 - self.alpha) * L_flow + self.reg * L_reg
        else:
            loss = (
                self.alpha * L_score
                + (1 - self.alpha) * L_flow
                + self.reg * L_reg
                + self.corr_strength * L_reg_corr
            )

        # (8) Log losses.
        self.log("train/score_loss", L_score, on_step=True, on_epoch=True)
        self.log("train/flow_loss", L_flow, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/reg_loss", L_reg, on_step=True, on_epoch=True)
        self.log("train/reg_corr_loss", L_reg_corr, on_step=True, on_epoch=True)

        if self.global_step % 100 == 0:
            self.print(
                f"step={self.global_step}, dataset={ds_idx}, L_score={L_score.item():.4f}, L_flow={L_flow.item():.4f}, "
                f"NGM_Reg={L_reg.item():.4f}, MLP_Reg={L_reg_corr.item():.4f}"
            )

        loss.backward()
        optimizer.step()
        self.proximal(self.net.fc1.weight, self.net.dims, lam=self.net.GL_reg, eta=0.01)

        return loss

    def configure_optimizers(self):
        params = (
            list(self.net.parameters())
            + list(self.score_net.parameters())
            + list(self.corr_net.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        return optimizer


def main():
    dm = TrajectoryStructureDataModule(
        data_path="data/",
        dataset="dyn-TF",
        dataset_type="Synthetic",
        batch_size=64,
        num_workers=4,
        train_val_test_split=(1.0, 0.0, 0.0),
        T=5,
    )

    dm.prepare_data()
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("X shape:", batch["X"].shape)
        print("t shape:", batch["t"].shape)
        print("Source ID:", batch["source_id"])
        break

    adatas = dm.get_subset_adatas(split="train")
    print(adatas)
    kos = dm.kos
    ko_indices = dm.ko_indices
    true_matrix = dm.true_matrix.values
    batch_size = 64
    n = adatas[0].X.shape[1]

    # Build conditional matrices.
    conditionals = []
    for i, ko in enumerate(kos):
        cond_matrix = torch.zeros(64, n)
        if ko is not None:
            cond_matrix[:, i] = 1
        conditionals.append(cond_matrix)

    # Build knockout masks.
    knockout_masks = []
    for i, ad in enumerate(adatas):
        d = ad.X.shape[1]
        mask_i = build_knockout_mask(d, ko_indices[i])
        knockout_masks.append(mask_i)

    print("Knockout masks:")
    print(knockout_masks)

    # Build Entropic OT Flow Models.
    T = dm.T

    def build_entropic_otfms(adatas, T, sigma, dt):
        otfms = []
        for adata in adatas:
            x_tensor = torch.tensor(adata.X, dtype=torch.float32)
            t_idx = torch.tensor(adata.obs["t"], dtype=torch.long)
            otfm = EntropicOTFM(
                x_tensor,
                t_idx,
                dt=dt,
                sigma=sigma,
                T=T,
                dim=x_tensor.shape[1],
                device=torch.device("cpu"),
            )
            otfms.append(otfm)
        return otfms

    otfms = build_entropic_otfms(adatas, T, sigma=1.0, dt=1 / T)

    # Define network dimensions.
    dims = [n, 100, 1]
    # Instantiate the neural ODE function with knockout masks
    func_v = MLPODEFKO(dims=dims, GL_reg=0.04, bias=True, knockout_masks=knockout_masks)
    # Instantiate the score network.
    score_net = CONDMLP(
        d=n, hidden_sizes=[100, 100], time_varying=True, conditional=True, conditional_dim=n
    )
    v_cor = MLP(d=n, hidden_sizes=[64, 64], time_varying=True)

    # Instantiate the Lightning Module.
    model = SF2MLitModule(
        net=func_v,
        score_net=score_net,
        corr_net=v_cor,
        alpha=0.1,
        corr_strength=1e-3,
        reg=5e-6,
        lr=3e-3,
        batch_size=64,
        skip_time=None,
    )

    model.otfms = build_entropic_otfms(adatas, T=dm.T, sigma=1.0, dt=1 / T)
    model.cond_matrix = conditionals

    # Instantiate the Trainer.
    trainer = Trainer(max_epochs=50)

    # Train the model.
    trainer.fit(model, datamodule=dm)

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
