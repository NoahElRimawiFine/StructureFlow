import glob
import numpy as np
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
import NMC as models
import importlib
import os
import ot
import scipy.sparse as sp
from tqdm import tqdm
from torchdiffeq import odeint
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score
import torchsde
from sf2m_utils import SDE, torch_wrapper, wasserstein
from src import util


T = 5


class DataLoader:
    def __init__(self, data_path="data", dataset_type="Synthetic", dataset="dyn-TF"):
        """
        Initialize DataLoader

        Args:
            data_path: Path to data directory
            dataset_type: Either "Synthetic" or "Curated"
        """
        self.data_path = os.path.join(data_path, dataset_type)
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.adatas = None
        self.kos = None
        self.true_matrix = None

    def load_data(self):
        # breakpoint()
        """Load and preprocess data"""
        if self.dataset_type == "Synthetic":
            paths = glob.glob(
                os.path.join(self.data_path, f"{self.dataset}/{self.dataset}*-1")
            ) + glob.glob(
                os.path.join(self.data_path, f"{self.dataset}_ko*/{self.dataset}*-1")
            )
        elif self.dataset_type == "Curated":
            paths = glob.glob(
                os.path.join(self.data_path, f"HSC/HSC-1000-1")
            ) + glob.glob(os.path.join(self.data_path, f"HSC_ko*/HSC*-1"))
            print(paths[0])
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
            except:
                self.kos.append(None)

        self.gene_to_index = {
            gene: idx for idx, gene in enumerate(self.adatas[0].var.index)
        }
        self.ko_indices = []
        for ko in self.kos:
            if ko is None:
                self.ko_indices.append(None)
            else:
                self.ko_indices.append(self.gene_to_index[ko])


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


def compute_pi_entropic_fixed(x0, x1, reg=1e-2, numItermax=5000):
    """
    Computes an entropic OT plan between x0 and x1 using the Sinkhorn algorithm.
    """
    x0_np = x0.cpu().numpy()
    x1_np = x1.cpu().numpy()
    a = ot.unif(x0_np.shape[0])  # uniform distribution over rows
    b = ot.unif(x1_np.shape[0])  # uniform distribution over columns
    # Cost matrix: squared Euclidean distance
    M = np.sum((x0_np[:, None, :] - x1_np[None, :, :]) ** 2, axis=2)
    pi = ot.sinkhorn(a, b, M, reg=reg, numItermax=numItermax)
    return pi


def compute_all_pis_fixed(adata, t, reg=1e-2):
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
            pi = compute_pi_entropic_fixed(x0, x1, reg=reg)

        all_pis.append(pi)
    return all_pis


class MLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
        time_varying=True,
    ):
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        self.time_varying = time_varying
        assert len(hidden_sizes) > 0
        hidden_sizes = copy.copy(hidden_sizes)

        # Input dimension is: original dimension + time (if time_varying) + one-hot KO vector
        input_dim = d
        if time_varying:
            input_dim += 1  # Add time dimension
        input_dim += d  # Add one-hot KO vector dimension

        hidden_sizes.insert(0, input_dim)
        hidden_sizes.append(d)

        for i in range(len(hidden_sizes) - 1):
            self.net.add_module(
                name=f"L{i}", module=nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )
            if i < len(hidden_sizes) - 2:
                self.net.add_module(name=f"A{i}", module=activation())

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x, ko_vector=None):
        """
        Args:
            t: time tensor [batch_size]
            x: input tensor [batch_size, d]
            ko_vector: one-hot encoded KO vector [batch_size, d] or None (defaults to all zeros)
        """
        if ko_vector is None:
            ko_vector = torch.zeros_like(x)  # Default to all zeros for wildtype

        if self.time_varying:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            # Concatenate input, time, and KO vector
            inp = torch.cat([x, t, ko_vector], dim=1)
        else:
            # Concatenate just input and KO vector
            inp = torch.cat([x, ko_vector], dim=1)

        return self.net(inp)


def convert_ko_to_index(ko_name):
    if ko_name is None:
        return None
    return int(ko_name[1:]) - 1


def train_with_fmot_scorematching(
    func_v,
    func_s,
    adatas_list,
    all_pis_list,
    t,
    kos_list,
    sigma=0.1,
    dt=1.0,
    alpha=0.5,
    reg=1e-5,
    n_steps=2000,
    batch_size=64,
    device="cpu",
    lr=1e-3,
):
    """
    Combine flow matching + score matching with multiple datasets
    """
    func_v.to(device)
    func_s.to(device)
    optimizer = torch.optim.AdamW(
        list(func_v.parameters()) + list(func_s.parameters()), lr=lr
    )

    loss_history = []

    def proximal(w, dims, lam=0.1, eta=0.01):
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    for step in tqdm(range(n_steps)):
        # for step in range(n_steps):
        # Randomly select a dataset
        dataset_idx = np.random.randint(0, len(adatas_list))
        adata = adatas_list[dataset_idx]
        pis = all_pis_list[dataset_idx]
        ko = kos_list[dataset_idx]

        # Randomly select a time bin
        tb = np.random.randint(0, t)

        pi_matrix = pis[tb]
        if pi_matrix is None:
            continue

        cells_t0 = adata.X[(adata.obs["t"] == tb).values, :]
        cells_t1 = adata.X[(adata.obs["t"] == tb + 1).values, :]
        n0, n1 = cells_t0.shape[0], cells_t1.shape[0]
        if n0 == 0 or n1 == 0:
            continue

        x0, x1 = sample_plan(cells_t0, cells_t1, pi_matrix, batch_size, device=device)

        tau = torch.rand(batch_size, 1, device=device)
        x_tau, s_true, u = brownian_bridge(x0, x1, tau, sigma=sigma)

        s_input = x_tau
        B = s_input.shape[0]
        t_tensor = float(tb) + tau.squeeze()

        ko = kos_list[dataset_idx]
        ko_vector = torch.zeros(batch_size, x0.shape[1], device=device)
        if ko is not None:  # If this is a knockout condition
            ko_vector[:, ko] = 1.0

        s_pred = func_s(t_tensor, s_input, ko_vector)

        v_input = x_tau.unsqueeze(1)
        # v_pred = func_v(tb, v_input).squeeze(1)
        v_pred = func_v(tb, v_input).squeeze(1) - sigma**2 / 2 * s_pred

        weight_ = tau * (1 - tau)
        L_score = torch.mean(weight_ * (s_pred - s_true) ** 2)
        L_flow = torch.mean((v_pred * dt - u) ** 2)

        L_reg = 0.0
        if hasattr(func_v, "l2_reg"):
            L_reg += func_v.l2_reg()
        if hasattr(func_v, "fc1_reg"):
            L_reg += func_v.fc1_reg()

        L = alpha * L_score + (1 - alpha) * L_flow + reg * L_reg

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        if hasattr(func_v, "fc1") and hasattr(func_v, "dims"):
            proximal(func_v.fc1.weight, func_v.dims, lam=func_v.GL_reg, eta=0.01)

        loss_history.append(L.item())

    return loss_history, func_v, func_s


def train_with_flowmatching(
    func_v,
    adatas_list,  # list of datasets
    all_pis_list,  # list of OT plans for each dataset
    t,
    sigma=0.1,
    dt=1.0,
    reg=1e-5,
    n_steps=2000,
    batch_size=64,
    device="cpu",
    lr=1e-3,
):
    """
    Train using only flow matching (no score matching).

    This function trains the flow model (func_v) using the flow matching loss.
    The loss is given by the mean squared error between the predicted
    flow (multiplied by dt) and the target flow (u) computed via the
    Brownian bridge.

    Args:
        func_v: Flow model
        adatas_list: list of datasets
        all_pis_list: list of OT plans for each dataset
        t: number of time bins (0,...,t)
        sigma: noise scale used for the Brownian bridge
        dt: time step length for scaling the flow
        reg: regularization coefficient
        n_steps: number of training steps
        batch_size: number of samples per batch
        device: device to use ("cpu" or "cuda")
        lr: learning rate for the optimizer

    Returns:
        loss_history: a list of loss values at each (or every few) training steps
        func_v: the updated flow model
    """
    func_v.to(device)
    optimizer = torch.optim.AdamW(func_v.parameters(), lr=lr)
    loss_history = []

    def proximal(w, dims, lam=0.1, eta=0.01):
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    for step in tqdm(range(n_steps)):
        # for step in range(n_steps):
        # Randomly select a dataset
        dataset_idx = np.random.randint(0, len(adatas_list))
        adata = adatas_list[dataset_idx]
        pis = all_pis_list[dataset_idx]

        # Randomly select a time bin
        tb = np.random.randint(0, t)

        pi_matrix = pis[tb]
        if pi_matrix is None:
            continue

        cells_t0 = adata.X[(adata.obs["t"] == tb).values, :]
        cells_t1 = adata.X[(adata.obs["t"] == tb + 1).values, :]
        n0, n1 = cells_t0.shape[0], cells_t1.shape[0]
        if n0 == 0 or n1 == 0:
            continue

        x0, x1 = sample_plan(cells_t0, cells_t1, pi_matrix, batch_size, device=device)

        # Obtain the Brownian bridge outputs. Even though brownian_bridge returns (x_tau, s_true, u),
        # the s_true value is not used in flow matching.
        tau = torch.rand(batch_size, 1, device=device)
        x_tau, _, u = brownian_bridge(x0, x1, tau, sigma=sigma)

        # Compute the predicted flow
        t_tensor = float(tb) + tau.squeeze()
        v_input = x_tau.unsqueeze(1)
        v_pred = func_v(t_tensor, v_input).squeeze(1)

        # Compute the flow matching loss
        L_flow = torch.mean((v_pred * dt - u) ** 2)

        # Compute any additional regularization loss if provided by the model
        L_reg = 0.0
        if hasattr(func_v, "l2_reg"):
            L_reg += func_v.l2_reg()
        if hasattr(func_v, "fc1_reg"):
            L_reg += func_v.fc1_reg()

        L = L_flow + reg * L_reg

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        if hasattr(func_v, "fc1") and hasattr(func_v, "dims"):
            proximal(func_v.fc1.weight, func_v.dims, lam=func_v.GL_reg, eta=0.01)

        loss_history.append(L.item())

    return loss_history, func_v


def simulate_trajectory(
    flow_model, score_model, x0, n_times=400, sigma=0.1, device="cpu", use_sde=False
):
    """
    Simulate trajectory using either ODE or SDE integration.

    Args:
        flow_model: The trained flow model
        score_model: The trained score model
        x0: Initial conditions [batch_size, n_features]
        n_times: Number of timepoints to simulate
        sigma: Noise scale for SDE
        device: Device to run simulation on
        use_sde: Whether to use SDE (True) or ODE (False)

    Returns:
        trajectory: Simulated trajectory [n_times, batch_size, n_features]
    """
    x0 = x0.to(device)
    ts = torch.linspace(0, 1, n_times, device=device)

    if use_sde:
        # Define drift and diffusion functions for SDE
        class FlowSDE(torch.nn.Module):
            def __init__(self, flow_model, score_model, sigma):
                super().__init__()
                self.flow_model = flow_model
                self.score_model = score_model
                self.sigma = sigma
                self.noise_type = "diagonal"
                self.sde_type = "ito"

            def f(self, t, x):
                # Drift term
                t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
                flow = self.flow_model(t_batch, x.unsqueeze(1)).squeeze(1)
                # score = self.score_model(t_batch, x)
                # return flow - (self.sigma**2 / 2) * score
                return flow

            def g(self, t, x):
                # Diffusion term
                return self.sigma * torch.ones_like(x)

        sde = FlowSDE(flow_model, score_model, sigma)
        with torch.no_grad():
            trajectory = torchsde.sdeint(sde, x0, ts, method="euler", dt=1e-2)

    else:
        # ODE integration
        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
            flow = flow_model(t_batch, x.unsqueeze(1)).squeeze(1)
            score = score_model(t_batch, x)
            return flow - (sigma**2 / 2) * score

        with torch.no_grad():
            trajectory = odeint(ode_func, x0, ts, method="dopri5")

    return trajectory.cpu()


def train_and_evaluate_with_holdout(
    adatas,
    held_out_time,
    num_variables,
    hidden_dim=200,
    n_steps=5000,
    kos_list=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train the model with one timepoint held out and evaluate performance.
    """
    dims = [num_variables, hidden_dim, 1]
    func_v = models.MLPODEF(dims=dims, GL_reg=0.015, bias=True)
    score_net = MLP(d=num_variables, hidden_sizes=[hidden_dim], time_varying=True)

    # Filter out data from held_out_time
    filtered_adatas = []
    for adata in adatas:
        mask = adata.obs["t"] != held_out_time
        filtered_adata = adata[mask].copy()
        filtered_adatas.append(filtered_adata)

    # Compute pis for filtered datasets
    all_pis_list = []
    for adata in filtered_adatas:
        pis = compute_all_pis_fixed(adata, adata.obs["t"].max(), reg=1e-1)
        all_pis_list.append(pis)

    # Train the model
    loss_history, flow_model, score_model = train_with_fmot_scorematching(
        func_v=func_v,
        func_s=score_net,
        adatas_list=filtered_adatas,
        all_pis_list=all_pis_list,
        t=filtered_adatas[0].obs["t"].max(),
        kos_list=kos_list,
        sigma=0.1,
        dt=1.0,
        alpha=0.05,
        reg=1e-7,
        n_steps=n_steps,
        batch_size=164,
        device=device,
        lr=3e-3,
    )

    # Evaluate on held-out timepoint
    distances = []
    for adata in adatas:
        x0 = torch.from_numpy(adata.X[adata.obs["t"] == held_out_time - 1]).float()
        true_dist = torch.from_numpy(adata.X[adata.obs["t"] == held_out_time]).float()

        if len(x0) == 0 or len(true_dist) == 0:
            continue

        # Simulate trajectory
        traj_ode = simulate_trajectory(flow_model, score_model, x0, use_sde=False)
        traj_sde = simulate_trajectory(flow_model, score_model, x0, use_sde=True)

        # Calculate Wasserstein distance
        w_dist_ode = wasserstein(traj_ode[held_out_time], true_dist)
        w_dist_sde = wasserstein(traj_sde[held_out_time], true_dist)

        distances.append({"ode": w_dist_ode, "sde": w_dist_sde})

    # Calculate average distances
    avg_distances = {
        "ode": np.mean([d["ode"] for d in distances]),
        "sde": np.mean([d["sde"] for d in distances]),
    }

    return avg_distances, flow_model, score_model


def plot_heatmap_comparison(
    result_matrix, true_matrix, gene_names, model_name, vmin=-2.5, vmax=2.5
):
    """Plot heatmap comparison of a model result against the true matrix.

    Args:
        result_matrix: Results from the model
        true_matrix: True interaction matrix
        gene_names: List of gene names for axis labels
        model_name: Name of the model for the title
        vmin: Minimum value for colorbar scale (default: -2.5)
        vmax: Maximum value for colorbar scale (default: 2.5)
    """
    import seaborn as sb

    plt.figure(figsize=(10, 4))

    # Plot result matrix
    plt.subplot(1, 2, 1)
    df = pd.DataFrame(result_matrix, index=gene_names, columns=gene_names)
    sb.heatmap(df, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    plt.gca().invert_yaxis()
    plt.title(model_name)

    # Plot true matrix
    plt.subplot(1, 2, 2)
    df_true = pd.DataFrame(true_matrix, index=gene_names, columns=gene_names)
    sb.heatmap(df_true, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    plt.gca().invert_yaxis()
    plt.title("True Matrix")

    plt.suptitle(f"Comparison of {model_name} vs True Matrix", y=1.05, fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    data_loader = DataLoader("data", dataset_type="Curated")
    data_loader.load_data()
    adatas, kos, ko_indices, true_matrix = (
        data_loader.adatas,
        data_loader.kos,
        data_loader.ko_indices,
        data_loader.true_matrix.values,
    )

    wt_idx = [i for i, ko in enumerate(kos) if ko is None]
    ko_idx = [i for i, ko in enumerate(kos) if ko is not None]
    adatas_wt = [adatas[i] for i in wt_idx]
    adatas_ko = [adatas[i] for i in ko_idx]
    num_variables = 8
    hidden_dim = 200
    dims = [num_variables, hidden_dim, 1]
    t = adatas[0].obs["t"].max()

    func_v = models.MLPODEF(dims=dims, GL_reg=0.015, bias=True)
    score_net = MLP(d=num_variables, hidden_sizes=[hidden_dim], time_varying=True)

    # Compute pis for all datasets
    all_pis_list = []
    for adata in adatas:
        pis = compute_all_pis_fixed(adata, t, reg=1e-1)
        all_pis_list.append(pis)

    kos_indices = [convert_ko_to_index(ko) for ko in kos]
    loss_history, flow_model, score_model = train_with_fmot_scorematching(
        func_v=func_v,
        func_s=score_net,
        adatas_list=adatas,
        all_pis_list=all_pis_list,
        t=t,
        kos_list=kos_indices,  # Pass the knockout information
        sigma=1.0,
        dt=1.0,
        alpha=0.1,
        reg=1e-5,
        n_steps=1_000,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-3,
    )
    # loss_history, flow_model = train_with_flowmatching(
    #     func_v=func_v,
    #     adatas_list=adatas,
    #     all_pis_list=all_pis_list,
    #     t=t,
    #     sigma=0.1,
    #     dt=1.0,
    #     reg=1e-7,
    #     n_steps=75000,
    #     batch_size=164,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     lr=3e-3,
    # )

    # Rest of the visualization code remains the same
    graph_sm = flow_model.causal_graph() * (1 - np.eye(num_variables))

    plot_heatmap_comparison(graph_sm, true_matrix, adatas[0].var.index, "Flow-based")

    plt.figure(figsize=(8, 6))
    y_true = np.abs(np.sign(true_matrix).astype(int).flatten())
    y_pred = np.abs(graph_sm.flatten())
    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)
    plt.plot(rec, prec, label=f"Flow-based (AP = {avg_prec:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve\nAUPR ratio = {avg_prec/np.mean(np.abs(true_matrix) > 0):.2f}"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def main_with_holdout():
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
    kos_indices = [convert_ko_to_index(ko) for ko in kos]

    num_variables = 8
    results = []

    for held_out_time in range(1, T):
        print(f"\n=== Training with timepoint {held_out_time} held out ===")

        avg_distances, flow_model, score_model = train_and_evaluate_with_holdout(
            adatas=adatas,
            held_out_time=held_out_time,
            num_variables=num_variables,
            n_steps=10000,
            kos_list=kos_indices,
        )

        results.append({"held_out_time": held_out_time, "distances": avg_distances})

        print(f"Results for held-out timepoint {held_out_time}:")
        print(f"ODE distance: {avg_distances['ode']:.4f}")
        print(f"SDE distance: {avg_distances['sde']:.4f}")

    # Calculate and print averages
    avg_ode = sum(r["distances"]["ode"] for r in results) / len(results)
    avg_sde = sum(r["distances"]["sde"] for r in results) / len(results)
    print("\n=== Average Results Across All Timepoints ===")
    print(f"Average ODE distance: {avg_ode:.4f}")
    print(f"Average SDE distance: {avg_sde:.4f}")

    return results


if __name__ == "__main__":
    # main_with_holdout()
    main()
