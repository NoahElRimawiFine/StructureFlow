import numpy as np
from sklearn.decomposition import PCA
import torch
import sys

sys.path.append("../../")
import matplotlib.pyplot as plt
import NMC as models
import importlib
import os
import ot
import glob
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from tqdm import tqdm
from torchdiffeq import odeint
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score
import torchsde

# from src import util
from sf2m_utils import SDE, torch_wrapper, wasserstein
import fm
from linear_wsynthetic import DataLoader
import random
import argparse
import functools

parser = argparse.ArgumentParser(
    description="Run FMOT + score‑matching training for multiple datasets/seeds."
)

parser.add_argument("--seed", type=int, default=42, help="Single seed")

args = parser.parse_args()
print = functools.partial(print, flush=True)

# # loading the data
# data = torch.load("sim_BF_beta_5.0_N_100_T_10_c_0.5.pkl", weights_only=False)
# adata = ad.AnnData(data["x"], {"t_idx": data["t_idx"]})


# def load_boolODE_reference_network(path, genes):
#     df = pd.read_csv(path)
#     n_genes = len(genes)
#     A_ref = pd.DataFrame(np.zeros((n_genes, n_genes), int), index=genes, columns=genes)
#     for i in range(df.shape[0]):
#         _i = df.iloc[i, 1]
#         _j = df.iloc[i, 0]
#         _v = {"+": 1, "-": -1}[df.iloc[i, 2]]
#         A_ref.loc[_i, _j] = _v
#     return A_ref


# genes = ["g1", "g2", "g3", "g4", "g6", "g7", "g8"]
# A_ref = load_boolODE_reference_network(f"./data/simulation/refNetwork.csv", genes)

# breakpoint()


T = 5
seed = args.seed
dataset = "Curated"
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_map(pi, batch_size, replace=True):
    """
    Randomly pick (i, j) from the coupling matrix pi (shape [n0, n1]).
    Returns arrays of row indices i and column indices j.
    """
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(len(p), size=batch_size, replace=replace, p=p)
    i = choices // pi.shape[1]
    j = choices % pi.shape[1]
    return i, j


def sample_plan(x0, x1, pi, batch_size, device="cpu"):
    """
    Given x0 in [n0, d], x1 in [n1, d], and pi in [n0, n1],
    sample a batch of (x0, x1) pairs according to pi.
    """
    i, j = sample_map(pi, batch_size)
    return torch.tensor(x0[i], dtype=torch.float32, device=device), torch.tensor(
        x1[j], dtype=torch.float32, device=device
    )


def brownian_bridge(x0, x1, tau, sigma=0.1):
    """
    Construct a Brownian bridge from x0->x1 at fraction tau in [0,1].
    x0, x1: shape [batch_size, d]
    tau: shape [batch_size, 1]
    sigma: noise scale
    """
    mean_ = (1 - tau) * x0 + tau * x1
    var_ = (sigma**2) * tau * (1 - tau)
    # sample x(tau) = mean + sqrt(var)*epsilon
    eps = torch.randn_like(x0)
    x_tau = mean_ + torch.sqrt(var_.clamp_min(1e-10)) * eps

    # bridging score: s = -(x - mean)/var
    s_true = -(x_tau - mean_) / var_.clamp_min(1e-10)

    denom = 2 * tau * (1 - tau) + 1e-10
    u = ((1 - 2 * tau) / denom) * (x_tau - mean_) + (x1 - x0)
    return x_tau, s_true, u


def prepare_time_binned_data(adata, time_column="t"):
    """
    Groups cells by their time bins and returns a list of tensors.

    Args:
        adata (AnnData): The AnnData object containing cell data.
        time_column (str): The column in adata.obs indicating time bins.

    Returns:
        List[torch.Tensor]: A list where each element is a tensor of cells at a specific time bin.
    """
    num_time_bins = adata.obs[time_column].nunique()
    time_bins = sorted(adata.obs[time_column].unique())
    grouped_data = []
    for t in time_bins:
        cells_t = adata[adata.obs[time_column] == t].X
        if isinstance(cells_t, sp.spmatrix):
            cells_t = cells_t.toarray()
        grouped_data.append(torch.from_numpy(cells_t).float())
    return grouped_data


def normalize_data(grouped_data):
    """
    Applies Z-score normalization to each gene across all cells.

    Args:
        grouped_data (List[torch.Tensor]): List of tensors grouped by time bins.

    Returns:
        List[torch.Tensor]: Normalized data.
    """
    all_cells = torch.cat(grouped_data, dim=0)
    scaler = StandardScaler()
    all_cells_np = all_cells.numpy()
    scaler.fit(all_cells_np)

    normalized_data = []
    for tensor in grouped_data:
        normalized = torch.from_numpy(scaler.transform(tensor.numpy())).float()
        normalized_data.append(normalized)
    return normalized_data, scaler


def build_knockout_mask(d, ko_idx, device="cpu"):
    """
    Build a [d, d] adjacency mask for a knockout of gene ko_idx.
    If ko_idx is None, return a mask of all ones (wild-type).
    """
    if ko_idx is None:
        # No knockout => no edges removed
        return torch.ones((d, d), dtype=torch.float32).to(device)
    else:
        mask = torch.ones((d, d), dtype=torch.float32).to(device)
        g = ko_idx
        # Zero row g => remove outgoing edges from gene g
        # mask[g, :] = 0.0
        # Zero column g => remove incoming edges to gene g
        mask[:, g] = 0.0
        mask[g, g] = 1.0
        return mask


def build_entropic_otfms(adatas, T, sigma, dt):
    """
    Returns a list of EntropicOTFM objects, one per dataset.
    """
    otfms = []
    for adata in adatas:
        x_tensor = torch.tensor(adata.X, dtype=torch.float32)
        x_pca_tensor = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        t_idx = torch.tensor(adata.obs["t"], dtype=torch.long)
        model = fm.EntropicOTFM(
            x=x_tensor,
            x_pca=x_pca_tensor,
            t_idx=t_idx,
            dt=dt,
            sigma=sigma,
            T=T,
            dim=x_tensor.shape[1],
            device="cuda" if torch.cuda.is_available() else "cpu",
            model="eot",
            lamda=1,
        )
        otfms.append(model)
    return otfms


def compute_pi_entropic_fixed(
    x0, x1, reg=1e-2, numItermax=10000, ko_index=None, cost=1e9
):
    """
    Computes an entropic OT plan between x0 and x1 using the Sinkhorn algorithm.
    """
    x0_np = x0.cpu().numpy()
    x1_np = x1.cpu().numpy()
    a = ot.unif(x0_np.shape[0])  # uniform distribution over rows
    b = ot.unif(x1_np.shape[0])  # uniform distribution over columns
    # Cost matrix: squared Euclidean distance
    M = np.sum((x0_np[:, None, :] - x1_np[None, :, :]) ** 2, axis=2)
    # if ko_index is not None:
    #     ko0 = (x0_np[:, ko_index] < 1)
    #     ko1 = (x1_np[:, ko_index] < 1)
    #     mismatch = (ko0[:,None] != ko1[None, :])
    #     M[mismatch] = cost
    pi = ot.sinkhorn(a, b, M, reg=reg, numItermax=numItermax)
    return pi


def compute_all_pis_fixed(adata, t, reg=1e-2, ko_index=None):
    """
    Precompute entropic OT for each time bin using a fixed plan (single dataset).

    Returns:
        all_pis: list of length t, where all_pis[time_bin] = pi_matrix (or None)
    """
    all_pis = []
    for time_bin in range(t):
        # Extract cells belonging to time_bin and time_bin+1
        x0 = adata.X[adata.obs["t"] == time_bin]
        x1 = adata.X[adata.obs["t"] == time_bin + 1]

        # Convert to torch tensors
        x0 = torch.from_numpy(x0).float()
        x1 = torch.from_numpy(x1).float()

        if x0.size(0) == 0 or x1.size(0) == 0:
            pi = None
        else:
            pi = compute_pi_entropic_fixed(x0, x1, reg=reg, ko_index=ko_index)

        all_pis.append(pi)
    return all_pis


class MLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.GELU,
        time_varying=True,
        conditional=False,
        conditional_dim=0,  # dimension of the knockout or condition
    ):
        super(MLP, self).__init__()
        self.time_varying = time_varying
        self.conditional = conditional

        input_dim = d
        if self.time_varying:
            input_dim += 1
        if self.conditional:
            input_dim += conditional_dim

        hidden_sizes = copy.copy(hidden_sizes)
        hidden_sizes.insert(0, input_dim)  # first layer's input size
        hidden_sizes.append(d)  # final layer is dimension d

        layers = []
        for i in range(len(hidden_sizes) - 1):
            in_size = hidden_sizes[i]
            out_size = hidden_sizes[i + 1]
            layers.append(nn.Linear(in_size, out_size))
            # activation except for the last layer
            if i < len(hidden_sizes) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

        # Weight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x, cond=None):
        inputs = [x]
        if self.time_varying:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inputs.append(t)

        if self.conditional:
            if cond is None:
                raise ValueError(
                    "Conditional flag = True, but no 'cond' input provided."
                )
            Bx = x.shape[0]
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(Bx, -1)
            elif cond.shape[0] != Bx:
                raise ValueError(
                    f"cond batch size ({cond.shape[0]}) != x batch size ({Bx}). "
                )
            inputs.append(cond)

        # cat along dim=1 => shape [batch_size, (d + time + cond_dim)]
        net_in = torch.cat(inputs, dim=1)
        return self.net(net_in)


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
    correction_reg_strength=0,
    device="cpu",
    lr=1e-3,
    true_mat=None,
    skip_time=None,
    freeze_flow=None,
    freeze_score=None,
):
    """
    Combine flow matching + score matching with multiple datasets
    """
    func_v.to(device)
    func_s.to(device)
    optim = torch.optim.AdamW(
        list(func_v.parameters())
        + list(func_s.parameters())
        + list(v_correction.parameters()),
        lr=lr,
    )

    loss_history = []
    score_loss_history = []
    flow_loss_history = []
    reg_loss_history = []
    reg_corr_loss_history = []

    save_dir = "training_visuals"
    os.makedirs(save_dir, exist_ok=True)

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

        if freeze_score is not None and i == freeze_score:
            print(f"Step {i}: Freezing Score Model (s)")
            for param in func_s.parameters():
                param.requires_grad = False
            optim = torch.optim.AdamW(
                filter(
                    lambda p: p.requires_grad,
                    list(func_v.parameters()) + list(func_s.parameters()),
                ),
                lr=lr,
            )

        if freeze_flow is not None and i == freeze_flow:
            print(f"Step {i}: Freezing Flow Model (v)")
            for param in func_v.parameters():
                param.requires_grad = False
            optim = torch.optim.AdamW(
                filter(
                    lambda p: p.requires_grad,
                    list(func_v.parameters()) + list(func_s.parameters()),
                ),
                lr=lr,
            )

        ds_idx = np.random.randint(0, len(adatas_list))
        model = otfms[ds_idx]
        cond_vector = cond_matrix[ds_idx]

        _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
            batch_size=batch_size, skip_time=skip_time
        )
        optim.zero_grad()
        _x = _x.to(device)
        _s = _s.to(device)
        _u = _u.to(device)
        _t = _t.to(device)
        _t_orig = _t_orig.to(device)

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
            v_fit = func_v(t_input, v_input, ds_idx).squeeze(
                1
            ) - model.sigma**2 / 2 * func_s(_t, _x, cond_expanded)
        else:
            v_fit = func_v(t_input, v_input, ds_idx).squeeze(1) + 0 * v_correction(
                _t, _x
            )
            v_fit = v_fit - model.sigma**2 / 2 * func_s(_t, _x, cond_expanded)

        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * model.dt - _u) ** 2)

        current_alpha = alpha
        if freeze_score is not None and i >= freeze_score:
            current_alpha = 0.0  # Ignore score loss, focus on flow
        elif freeze_flow is not None and i >= freeze_flow:
            current_alpha = 1.0  # Ignore flow loss, focus on score

        L_reg = func_v.l2_reg() + func_v.fc1_reg()
        L_reg_correction = mlp_l2_reg(v_correction)
        if i < 100:  # train score for first few iters
            L = current_alpha * L_score
        elif i >= 100 and i <= 500:
            L = current_alpha * L_score + (1 - current_alpha) * L_flow + reg * L_reg
        else:
            L = (
                current_alpha * L_score + (1 - current_alpha) * L_flow + reg * L_reg
            )  # + correction_reg_strength * L_reg_correction

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
        if (freeze_flow is None) or (i < freeze_flow):
            proximal(func_v.fc1.weight, func_v.dims, lam=func_v.GL_reg, eta=0.01)

        if i % 1000 == 0:
            print(
                f"Step={i}, dataset={ds_idx}, L_score={L_score.item():.4f}, "
                f"L_flow={L_flow.item():.4f}, L_reg={L_reg:.4f}"
            )

    plt.plot(loss_history)
    plt.title("Score+Flow Matching Loss")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.show()

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


def train_with_fmot(
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
    correction_reg_strength=0,
    device="cpu",
    lr=1e-3,
    true_mat=None,
    skip_time=None,
    freeze_flow=None,
    freeze_score=None,
):
    """
    Combine flow matching + score matching with multiple datasets
    """
    func_v.to(device)
    func_s.to(device)
    optim = torch.optim.AdamW(
        list(func_v.parameters()) + list(func_s.parameters()), lr=lr
    )

    loss_history = []
    score_loss_history = []
    flow_loss_history = []
    reg_loss_history = []
    reg_corr_loss_history = []

    save_dir = "training_visuals"
    os.makedirs(save_dir, exist_ok=True)

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
        optim.zero_grad()
        _x = _x.to(device)
        _s = _s.to(device)
        _u = _u.to(device)
        _t = _t.to(device)
        _t_orig = _t_orig.to(device)

        # Reshape inputs for MLPODEF
        s_input = _x.unsqueeze(1)
        v_input = _x.unsqueeze(1)
        t_input = _t.unsqueeze(1)

        B = _x.shape[0]
        cond_expanded = cond_vector.repeat(B // cond_vector.shape[0] + 1, 1)[:B]

        if i <= 500:
            v_fit = func_v(t_input, v_input).squeeze(1) - model.sigma**2 / 2 * func_s(
                _t, _x, cond_expanded
            )
        else:
            v_fit = func_v(t_input, v_input).squeeze(1) - model.sigma**2 / 2 * func_s(
                _t, _x, cond_expanded
            )

        L_flow = torch.mean((v_fit * model.dt - _u) ** 2)
        L_score = torch.mean(
            (_t_orig * (1 - _t_orig)) * (func_s(_t, _x, cond_expanded) - _s) ** 2
        )

        current_alpha = alpha
        if freeze_score is not None and i >= freeze_score:
            current_alpha = 0.0  # Ignore score loss, focus on flow
        elif freeze_flow is not None and i >= freeze_flow:
            current_alpha = 1.0  # Ignore flow loss, focus on score

        L = L_flow + L_score  # + correction_reg_strength * L_reg_correction

        with torch.no_grad():
            if i % 100 == 0:
                print(
                    f"step={i}, dataset={ds_idx}, L_flow={L_flow.item():.4f}, L_score={L_score.item():.4f}"
                )
            loss_history.append(L.item())
            flow_loss_history.append(L_flow.item())
            score_loss_history.append(L_score.item())

        L.backward()
        optim.step()

        if i % 1000 == 0:
            print(
                f"Step={i}, dataset={ds_idx}, "
                f"L_flow={L_flow.item():.4f}, L_score={L_score.item():.4f}"
            )

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


def simulate_trajectory(
    flow_model,
    corr_model,
    score_model,
    x0,
    dataset_idx,
    start_time,
    end_time,
    n_times=400,
    sigma=1.0,
    device="cpu",
    use_sde=False,
    cond_vector=None,
):
    x0 = x0.to(device)
    dt = 1 / T
    t_start = start_time * dt
    t_end = end_time * dt
    ts = torch.linspace(t_start, t_end, n_times, device=device)

    if use_sde:

        class FlowSDE(torch.nn.Module):
            def __init__(self, flow_model, corr_model, score_model, sigma):
                super().__init__()
                self.flow_model = flow_model
                self.corr_model = corr_model
                self.score_model = score_model
                self.sigma = sigma
                self.noise_type = "diagonal"
                self.sde_type = "ito"

            def f(self, t, x):
                t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
                flow_out = self.flow_model(
                    t_batch, x.unsqueeze(1), dataset_idx
                ).squeeze(1)
                corr_out = self.corr_model(t_batch.unsqueeze(1), x)
                score_out = self.score_model(t_batch, x, cond_vector)
                return flow_out + corr_out

            def g(self, t, x):
                return self.sigma * torch.ones_like(x)

        sde = FlowSDE(flow_model, corr_model, score_model, sigma)
        with torch.no_grad():
            trajectory = torchsde.sdeint(sde, x0, ts, method="euler")

    else:

        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
            flow_out = flow_model(t_batch, x.unsqueeze(1), dataset_idx).squeeze(1)
            corr_out = corr_model(t_batch.unsqueeze(1), x)
            score_out = score_model(t_batch, x, cond_vector)
            return flow_out + corr_out - (sigma**2 / 2) * score_out

        with torch.no_grad():
            trajectory = odeint(ode_func, x0, ts, method="dopri5")

    return trajectory.cpu()


def train_and_evaluate_with_holdout(
    adatas,
    held_out_time,
    num_variables,
    kos,
    ko_indices,
    true_matrix,
    hidden_dim=200,
    n_steps=5000,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train the model on data excluding one timepoint and then evaluate trajectory simulation
    on the held-out time.

    We simulate trajectories over a relative time interval.

    Args:
        adatas: List of AnnData objects.
        held_out_time: The timepoint to hold out.
        num_variables: Number of variables (e.g., genes).
        hidden_dim: Hidden layer dimension.
        n_steps: Number of training steps.
        device: Computation device.

    Returns:
        avg_distances: Dictionary with average Wasserstein distances for ODE and SDE simulations.
        flow_model: Trained flow model.
        score_model: Trained score model.
    """
    batch_size = 164
    n = adatas[0].X.shape[1]

    # want to create a [8, n, 8] matrix that is one hot encoded and will be selected depending on dataset idx
    conditionals = []
    for i, ad in enumerate(kos):
        cond_matrix = torch.zeros(batch_size, 8)
        if ad is not None:
            cond_matrix[:, i] = 1
        conditionals.append(cond_matrix)

    knockout_masks = []
    for i, ad in enumerate(adatas):
        d = ad.X.shape[1]
        mask_i = build_knockout_mask(d, ko_indices[i])  # returns [d,d]
        knockout_masks.append(mask_i)

    wt_idx = [i for i, ko in enumerate(kos) if ko is None]
    ko_idx = [i for i, ko in enumerate(kos) if ko is not None]
    adatas_wt = [adatas[i] for i in wt_idx]
    adatas_ko = [adatas[i] for i in ko_idx]
    dims = [n, 100, 1]
    t = adatas[0].obs["t"].max()

    func_v = models.MLPODEF1(
        dims=dims, GL_reg=0.04, bias=True, knockout_masks=knockout_masks
    )
    score_net = MLP(
        d=n,
        hidden_sizes=[100, 100, 100],
        time_varying=True,
        conditional=True,
        conditional_dim=n,
    )

    v_cor = fm.MLP(d=n, hidden_sizes=[64, 64], time_varying=True)

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
        reg=3e-5,
        n_steps=n_steps,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-3,
        true_mat=true_matrix,
        skip_time=held_out_time,
    )

    if held_out_time == 4:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

        # Fit PCA once using wildtype data
        all_data = np.vstack([adata.X for adata in adatas])
        pca = PCA(n_components=2)
        pca.fit(all_data)

        # Pick 3 random knockout indices
        ko_indices = [1, 2, 3, 4, 5, 6]

        # Create plots for three different knockouts
        for plot_idx, (ax, ko_idx_local) in enumerate(zip([ax1, ax2, ax3], ko_indices)):
            # Get initial conditions for current knockout
            x0 = torch.from_numpy(
                adatas_ko[ko_idx_local].X[adatas_ko[ko_idx_local].obs["t"] == T - 2]
            ).float()
            dataset_idx = ko_idx[ko_idx_local]  # Get the actual dataset index
            cond_vector = conditionals[dataset_idx]
            if cond_vector is not None:
                cond_vector = cond_vector[0].repeat(len(x0), 1)

            # Simulate trajectory
            traj = simulate_trajectory(
                flow_model,
                corr_model,
                score_model,
                x0,
                dataset_idx=dataset_idx,
                start_time=T - 2,
                end_time=T - 1,
                n_times=400,
                sigma=1.0,
                use_sde=True,
                cond_vector=cond_vector,
            )

            # Get data for current knockout condition
            ko_data = adatas_ko[ko_idx_local].X
            ko_times = adatas_ko[ko_idx_local].obs["t"]

            # Transform data using the PCA fit on wildtype
            true_ko_pca = pca.transform(ko_data)
            final_predictions = traj[-1]
            if isinstance(final_predictions, torch.Tensor):
                final_predictions = final_predictions.cpu().numpy()
            pred_pca = pca.transform(final_predictions)

            # Create scatter plot
            scatter = ax.scatter(
                true_ko_pca[:, 0],
                true_ko_pca[:, 1],
                c=ko_times,
                cmap="viridis",
                label="True trajectory" if plot_idx == 0 else None,
            )
            ax.scatter(
                pred_pca[:, 0],
                pred_pca[:, 1],
                c="salmon",
                label="Model predictions" if plot_idx == 0 else None,
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2" if plot_idx == 0 else "")
            ax.set_title(f"Gene {ko_idx[ko_idx_local]} KO")
            ax.set_aspect("equal")

        # After all plots are created, set the same limits for all axes
        all_axes = [ax1, ax2, ax3]
        x_min = min(ax.get_xlim()[0] for ax in all_axes)
        x_max = max(ax.get_xlim()[1] for ax in all_axes)
        y_min = min(ax.get_ylim()[0] for ax in all_axes)
        y_max = max(ax.get_ylim()[1] for ax in all_axes)

        # Set the same limits for all plots
        for ax in all_axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Add colorbar with specific position
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(scatter, cax=cbar_ax, label="Time")

        # Add legend
        fig.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

        plt.suptitle(
            "PCA of True Trajectory vs Model Predictions\n(trained with final timepoint withheld)",
            y=1.05,
        )

        # Adjust layout while maintaining equal sizes
        plt.tight_layout(
            rect=[0, 0, 0.9, 0.95]
        )  # Adjust the right margin to make room for colorbar
        plt.show()
    distances = []
    for i, adata in enumerate(adatas):
        x0 = torch.from_numpy(adata.X[adata.obs["t"] == held_out_time - 1]).float()
        true_dist = torch.from_numpy(adata.X[adata.obs["t"] == held_out_time]).float()
        cond_vector = conditionals[i]
        if cond_vector is not None:
            cond_vector = cond_vector[0].repeat(len(x0), 1)

        if len(x0) == 0 or len(true_dist) == 0:
            continue

        traj_ode = simulate_trajectory(
            flow_model,
            corr_model,
            score_model,
            x0,
            dataset_idx=i,
            start_time=held_out_time - 1,
            end_time=held_out_time,
            use_sde=False,
            cond_vector=cond_vector,
        )
        traj_sde = simulate_trajectory(
            flow_model,
            corr_model,
            score_model,
            x0,
            dataset_idx=i,
            start_time=held_out_time - 1,
            end_time=held_out_time,
            use_sde=True,
            cond_vector=cond_vector,
        )

        w_dist_ode = wasserstein(traj_ode[-1], true_dist)
        w_dist_sde = wasserstein(traj_sde[-1], true_dist)

        distances.append({"ode": w_dist_ode, "sde": w_dist_sde})

    avg_distances = {
        "ode": np.mean([d["ode"] for d in distances]),
        "sde": np.mean([d["sde"] for d in distances]),
    }

    return avg_distances, flow_model, score_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader("data", dataset_type="Synthetic", dataset="dyn-TF")
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
        cond_matrix = torch.zeros(batch_size, n).to(device)
        if ad is not None:
            cond_matrix[:, i] = 1
        conditionals.append(cond_matrix)

    knockout_masks = []
    for i, ad in enumerate(adatas):
        d = ad.X.shape[1]
        mask_i = build_knockout_mask(d, ko_indices[i], device)
        knockout_masks.append(mask_i)

    wt_idx = [i for i, ko in enumerate(kos) if ko is None]
    ko_idx = [i for i, ko in enumerate(kos) if ko is not None]
    adatas_wt = [adatas[i] for i in wt_idx]
    adatas_ko = [adatas[i] for i in ko_idx]
    dims = [n, 100, 1]
    t = adatas[0].obs["t"].max()

    func_v = models.MLPODEF1(
        dims=dims, GL_reg=0.04, bias=True, knockout_masks=knockout_masks
    ).to(device)
    # func_v = fm.MLP(
    #     d=n, hidden_sizes=[128], time_varying=False).to(device)

    score_net = MLP(
        d=n,
        hidden_sizes=[100, 100],
        time_varying=True,
        conditional=True,
        conditional_dim=n,
    ).to(device)

    # count params
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Flow model params: {count_params(func_v)}")
    print(f"Score model params: {count_params(score_net)}")

    v_cor = fm.MLP(d=n, hidden_sizes=[128, 128], time_varying=True).to(device)
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
    ) = train_with_fmot(
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

    def plot_loss_components(
        iterations,
        score_loss_history,
        flow_loss_history,
        total_loss_history,
        reg_loss_history,
        reg_corr_loss_history,
    ):
        """
        Plots the different loss components over training iterations.

        Args:
            iterations (list or np.array): The iteration numbers.
            score_loss_history (list): History of score loss values.
            flow_loss_history (list): History of flow loss values.
            total_loss_history (list): History of total loss values.
            reg_loss_history (list): History of regularization (L_reg) values.
            reg_corr_loss_history (list): History of correction regularization (L_reg_corr) values.
        """
        plt.figure(figsize=(12, 8))
        plt.plot(iterations, score_loss_history, label="Score Loss", linewidth=2)
        plt.plot(iterations, flow_loss_history, label="Flow Loss", linewidth=2)
        plt.plot(iterations, reg_loss_history, label="Reg Loss", linewidth=2)
        plt.plot(iterations, reg_corr_loss_history, label="Corr Reg Loss", linewidth=2)
        plt.plot(
            iterations,
            total_loss_history,
            label="Total Loss",
            linewidth=3,
            linestyle="--",
        )
        plt.xlabel("Training Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Components over Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    iterations = range(len(loss_history))

    def maskdiag(A):
        return A * (1 - np.eye(n))

    def compute_global_jacobian(v, adatas, dt, device=torch.device("cpu")):
        """
        Compute a single adjacency from a big set of states across all datasets.
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
        A_estim = compute_global_jacobian(func_v, adatas, dt=1 / T, device=device)

    W_v = func_v.causal_graph(w_threshold=0.0).T
    W_v = W_v
    A_true = true_matrix

    pd.DataFrame(A_true).to_csv("A_true.csv")

    # Display both the estimated adjacency matrix and the learned causal graph
    def save_adj_heat(mat, title, out_name, cmap="RdBu_r"):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(maskdiag(mat), cmap=cmap)
        ax.set_title(title)
        ax.invert_yaxis()
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(f"{out_name}.pdf", dpi=300)
        plt.close(fig)

    save_adj_heat(A_estim, "A_estim (Jacobian)", f"A_estim_")
    save_adj_heat(
        W_v, "StructureFlow (TF)", f"StructureFlow_nerpy_wt{seed}", cmap="Reds"
    )
    save_adj_heat(A_true, "true matrix", f"A_true_")

    # Compute and display precision-recall curves for both methods
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
    )

    # plt.figure(figsize=(12, 5))
    # # For Jacobian-based estimation
    # plt.subplot(1, 2, 1)
    y_true = np.abs(np.sign(maskdiag(A_true)).astype(int).flatten())
    # print(np.array(y_true).reshape(n,n))
    y_pred = np.abs(maskdiag(A_estim).flatten())
    # prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    # avg_prec = average_precision_score(y_true, y_pred)
    # plt.plot(rec, prec, label=f"Jacobian-based (AP = {avg_prec:.2f})")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title(
    #     f"Precision-Recall Curve (Jacobian)\nAUPR ratio = {avg_prec/np.mean(np.abs(A_true) > 0)}"
    # )
    # plt.legend()
    # plt.grid(True)
    # # For MLPODEF-based estimation
    # plt.subplot(1, 2, 2)
    y_pred_mlp = np.abs(maskdiag(W_v).flatten())
    matrix = np.array(y_pred_mlp).reshape(n, n)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.to_csv(f"StructFlow_wt_{seed}y.csv", index=False)
    prec, rec, thresh = precision_recall_curve(y_true, y_pred_mlp)
    avg_prec_mlp = average_precision_score(y_true, y_pred_mlp)
    auc_w_v = roc_auc_score(y_true, y_pred_mlp)
    print(f"AUC for W_v: {auc_w_v:.4f}")
    print(f"Average Precision for W_v: {avg_prec_mlp:.4f}")

    # plt.plot(rec, prec, label=f"MLPODEF-based (AP = {avg_prec_mlp:.2f})")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title(
    #     f"Precision-Recall Curve (MLPODEF)\nAUPR ratio = {avg_prec_mlp/np.mean(np.abs(A_true) > 0)}"
    # )
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


def main_with_holdout(n_steps=1000):
    """
    Run leave-one-out evaluation for the linear model.
    """
    data_loader = DataLoader("data", dataset_type="Synthetic")
    data_loader.load_data()
    adatas, kos, ko_indices, true_matrix = (
        data_loader.adatas,
        data_loader.kos,
        data_loader.ko_indices,
        data_loader.true_matrix.values,
    )

    num_variables = 8
    results = []

    for held_out_time in range(1, T):
        # for held_out_time in [4]:
        print(f"\n=== Training with timepoint {held_out_time} held out ===")

        avg_distances, flow_model, score_model = train_and_evaluate_with_holdout(
            adatas=adatas,
            held_out_time=held_out_time,
            num_variables=num_variables,
            kos=kos,
            ko_indices=ko_indices,
            true_matrix=true_matrix,
            n_steps=n_steps,
        )

        results.append({"held_out_time": held_out_time, "distances": avg_distances})

        print(f"Results for held-out timepoint {held_out_time}:")
        print(f"ODE distance: {avg_distances['ode']:.4f}")
        print(f"SDE distance: {avg_distances['sde']:.4f}")

    return results


if __name__ == "__main__":
    # main_with_holdout(n_steps=15000)
    main()
