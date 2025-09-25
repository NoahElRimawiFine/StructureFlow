import numpy as np
import scipy as sp
import scanpy as sc
import anndata as ad
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
import scanpy as sc
import sys
sys.path.append("./src/models/components/")
import importlib
import rf #type: ignore
importlib.reload(rf)
import torch
import dcor # type: ignore
import glob
import networkx as nx
import random
import argparse
import torch.nn.init as init
from scipy.sparse.linalg import cg

os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(
        description="Run FMOT + score‑matching training for multiple datasets/seeds."
)

parser.add_argument("--seed", type=int, default=1,
                    help="Single seed or comma‑separated list, e.g. 0,1,2,3,4")
parser.add_argument("--mix", type=bool, default=False,
                    help="mix batches or not")
args = parser.parse_args()

mixed_batches = args.mix
seed = args.seed


def set_global_seed(seed: int = 3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)      # PyTorch 1.11+
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seed(seed)

PLT_CELL = 3.5
adata = sc.read_h5ad('./data/Renge/hipsc.h5ad')

selected_genes = [
    "MYC", "TCF7L1", "ZNF398", "UBTF", "NR5A2", "DNMT1", "MED1", "KDM5B",
    "TCF3", "SALL4", "CHD7", "CTNNB1", "JARID2", "FOXH1", "NANOG",
    "SOX2", "PRDM14", "POU5F1",
]
gene_strings = adata.var["gene"].astype(str)  

missing = [g for g in selected_genes if g not in gene_strings.values]
if missing:
    raise ValueError(f"These genes weren’t found in adata.var['gene']: {missing}")

adata.var_names = gene_strings
adata.var_names_make_unique() 

adata18 = adata[:, selected_genes].copy()

valid_targets  = set(selected_genes)
is_panel_ko    = adata18.obs["ko"].isin(valid_targets)

adata_sub = adata18[is_panel_ko, :].copy()
print(f"Retaining {adata_sub.n_obs} cells out of {adata.n_obs}")

X = pd.read_csv('./data/Renge/X_renge_d2_80.csv', index_col=0)
E = pd.read_csv('./data/Renge/E_renge_d2_80.csv', index_col=0)

adata_ = ad.AnnData(E)
adata_.obs["condition"] = None
adata_.obs.loc[X.index[X.iloc[:, :-1].T.sum(0) == 0], "condition"] = "wt"
idx_ko = X.index[X.iloc[:, :-1].T.sum(0) == 1]
adata_.obs.loc[idx_ko, "condition"] = X.columns[np.argmax(X.loc[idx_ko, :].iloc[:, :-1], -1)]
sc.pp.pca(adata_)
sc.pp.neighbors(adata_)
adata_.obs["t"] = X.t

# sc.pl.scatter(adata, basis = "pca", color = "t")

options = {
    "lr" : 0.1,
    "reg_sinkhorn" : 0.05,
    "reg_A" : 0.5e-3, 
    "reg_A_elastic" : 0.5,
    "iter" : 100,
    "ot_coupling" : True,
    "optimizer" : torch.optim.Adam,
    "n_pca_components" : 50
}

dists_ko = pd.Series({k : dcor.energy_distance(adata.obsm["X_pca"][adata.obs.ko == "WT", :], adata.obsm["X_pca"][adata.obs.ko == k, :]) for k in adata.obs.ko.unique()})

adata_tf = adata_sub.copy()
mask = adata_tf.var["gene"].isin(E.columns).values
adata_tf = adata_tf[:, mask].copy()
adata_tf.var_names = adata_tf.var["gene"]

_kos = list(dists_ko.sort_values()[::-1][range(8)].index) 
_kos = ['MYC', 'SOX2', 'NANOG', 'PRDM14', 'POU5F1']

_adata = adata[adata.obs.ko.isin(_kos), :]


_adatas = []
for k in _kos:
    _adatas.append(adata_tf[adata_tf.obs.ko == k, :].copy())
    _adatas[-1].X = np.asarray(_adatas[-1].X.todense(), dtype = np.float64)
    _adatas[-1].obs.t -= 2
    _adatas[-1].var.index = _adatas[-1].var.gene
if _kos[0] == "WT":
    _kos[0] = None

# Construct reference 
refs = {}
for f in glob.glob("chip_1kb/*.tsv"):
    gene = os.path.splitext(os.path.basename(f))[0].split(".")[0]
    df = pd.read_csv(f, sep = "\t")
    df.index = df.Target_genes
    # if len(df.columns[df.columns.str.contains("iPS_cells|ES_cells")]) == 0:
    #     print(pd.unique(df.columns.str.split("|").str[1]))
    y = pd.Series(df.loc[:, df.columns.str.contains("iPS_cells")].values.mean(-1), index = df.index)
    # y = pd.Series(df.iloc[:, 2:].values.mean(-1), index = df.index) 
    # y = pd.Series(df.iloc[:, 1], index = df.index) 
    refs[gene] = y

A_ref = pd.DataFrame(refs).T
A_ref[np.isnan(A_ref.values)] = 0

tfs = A_ref.index
tfs_no_ko = [i for i in tfs if i not in _kos]
tfs_ko = [i for i in tfs if i in _kos]

def build_knockout_mask(d, ko_idx):
    """
    Build a [d, d] adjacency mask for a knockout of gene ko_idx.
    If ko_idx is None, return a mask of all ones (wild-type).
    """
    print(ko_idx)
    if ko_idx is None:
        # No knockout => no edges removed
        return np.ones((d, d), dtype=np.float32)
    else:
        mask = np.ones((d, d), dtype=np.float32)
        g = ko_idx
        # Zero row g => remove outgoing edges from gene g
        # mask[g, :] = 0.0
        # Zero column g => remove incoming edges to gene g
        mask[:, g] = 0.0
        mask[g, g] = 1.0
        return mask
    
import ot

class BridgeMatcher:
    def __init__(self):
        pass

    def sample_map(self, pi, batch_size, replace=True):
        p = pi.flatten()
        p = p / p.sum()
        choices = torch.multinomial(p, num_samples=batch_size, replacement=replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, pi, batch_size, replace=True):
        i, j = self.sample_map(pi, batch_size, replace=replace)
        return x0[i], x1[j]

    def sample_bridge_and_flow(self, x0, x1, ts, sigma):
        # Sample Brownian bridges between paired entries of [x0, x1] at times ts \in [0, 1].
        means = (1 - ts) * x0 + ts * x1
        vars = (sigma**2) * ts * (1 - ts)
        x = means + torch.sqrt(vars.clamp_min(1e-8)) * torch.randn_like(x0)
        s = (-1 / vars.clamp_min(1e-8)) * (x - means)
        u = (1 - 2 * ts) / (2 * ts * (1 - ts) + 1e-8) * (x - means) + x1 - x0
        return means, vars, x, s, u


class EntropicOTFM:
    def __init__(
        self, x, t_idx, dt, sigma, T, dim, device, held_out_time=None, normalize_C=False
    ):
        def entropic_ot_plan(x0, x1, eps, normalize_C=False):
            C = ot.utils.euclidean_distances(x0, x1, squared=True) / 2

            if normalize_C:
                C = C / C.max()
            p, q = torch.full((x0.shape[0],), 1 / x0.shape[0]), torch.full(
                (x1.shape[0],), 1 / x1.shape[0]
            )
            sinkhorn_ot = ot.sinkhorn(p, q, C, eps, method="sinkhorn", numItermax=5000)

            return sinkhorn_ot

        self.sigma = sigma
        self.bm = BridgeMatcher()
        self.x = x
        self.t_idx = t_idx
        self.dt = dt
        self.T = T
        self.dim = dim
        self.device = device
        self.Ts = []
        self.held_out_time = held_out_time
        self.has_bridge_over_held_out = False
        self.normalize_C = normalize_C

        # construct EOT plans
        for i in range(self.T - 1):
            if self.held_out_time is not None and (
                i == self.held_out_time or i + 1 == self.held_out_time
            ):
                self.Ts.append(None)

                # Create a bridge over the held-out time if it's the first encounter
                if i == self.held_out_time and not self.has_bridge_over_held_out:
                    self.bridge_over_held_out = entropic_ot_plan(
                        self.x[self.t_idx == i - 1, :],
                        self.x[self.t_idx == i + 1, :],
                        2 * self.dt * self.sigma**2,
                        self.normalize_C,
                    )
                    self.has_bridge_over_held_out = True
            else:
                self.Ts.append(
                    entropic_ot_plan(
                        self.x[self.t_idx == i, :],
                        self.x[self.t_idx == i + 1, :],
                        self.dt * self.sigma**2,
                        self.normalize_C,
                    )
                )

    def sample_bridges_flows(self, batch_size=64, skip_time=None):
        _x = []
        _t = []
        _t_orig = []
        _s = []
        _u = []
        i = 0
        while i < self.T - 1:
            if skip_time is not None and (i == skip_time or i + 1 == skip_time):
                if i == skip_time and self.has_bridge_over_held_out:
                    # Use the bridge spanning the held-out timepoint
                    with torch.no_grad():
                        x0, x1 = self.bm.sample_plan(
                            self.x[self.t_idx == i - 1, :],
                            self.x[self.t_idx == i + 1, :],
                            self.bridge_over_held_out,
                            batch_size,
                        )
                    ts = torch.rand_like(x0[:, :1])
                    _, _, x, s, u = self.bm.sample_bridge_and_flow(
                        x0, x1, ts, (2 * self.sigma**2 * self.dt) ** 0.5
                    )
                    _x.append(x)
                    _s.append(s)
                    _t.append(
                        (i - 1 + ts * 2) * self.dt
                    )  # Scale ts to span 2 timesteps
                    _t_orig.append(ts)
                    _u.append(u)
                    i += 1
                else:
                    i += 1
            else:
                with torch.no_grad():
                    x0, x1 = self.bm.sample_plan(
                        self.x[self.t_idx == i, :],
                        self.x[self.t_idx == i + 1, :],
                        self.Ts[i],
                        batch_size,
                    )
                ts = torch.rand_like(x0[:, :1])
                _, _, x, s, u = self.bm.sample_bridge_and_flow(
                    x0, x1, ts, (self.sigma**2 * self.dt) ** 0.5
                )
                _x.append(x)
                _s.append(s)
                _t.append((i + ts) * self.dt)
                _t_orig.append(ts)
                _u.append(u)
                i += 1
        return (
            torch.vstack(_x),
            torch.vstack(_s),
            torch.vstack(_u),
            torch.vstack(_t),
            torch.vstack(_t_orig),
        )

# class BridgeMatcher:
#     def __init__(self):
#         pass

#     def sample_map(self, pi, batch_size, replace=True):
#         p = pi.flatten()
#         p = p / p.sum()
#         choices = torch.multinomial(p, num_samples=batch_size, replacement=replace)
#         return np.divmod(choices, pi.shape[1])

#     def sample_plan(self, x0, x1, pi, batch_size, replace=True):
#         i, j = self.sample_map(pi, batch_size, replace=replace)
#         return x0[i], x1[j]

#     def sample_bridge_and_flow(self, x0, x1, ts, sigma):
#         # Sample Brownian bridges between paired entries of [x0, x1] at times ts \in [0, 1].
#         means = (1 - ts) * x0 + ts * x1
#         vars = (sigma**2) * ts * (1 - ts)
#         # x = means + torch.sqrt(vars) * torch.randn_like(x0)
#         x = means + torch.sqrt(vars.clamp_min(1e-10)) * torch.randn_like(x0)
#         s = (-1 / vars.clamp_min(1e-10)) * (x - means)
#         u = (1 - 2 * ts) / (2 * ts * (1 - ts) + 1e-10) * (x - means) + x1 - x0
#         return means, vars, x, s, u

# def pins_ot_plan(x0, x1, eps, *,
#                  K=3, rho=0.10, t1=5_000, t2=20,
#                  tol_outer=1e-8, tol_cg=1e-6,
#                  device=None, dtype=torch.float64):
#     if device is None:
#         device = x0.device
#     x0 = x0.to(device, dtype)
#     x1 = x1.to(device, dtype)

#     n, m = x0.size(0), x1.size(0)
#     p = torch.full((n,), 1./n, dtype=dtype, device=device)
#     q = torch.full((m,), 1./m, dtype=dtype, device=device)

#     C = 0.5 * torch.cdist(x0, x1, p=2).pow(2)
#     C = C / C.mean()

#     def log_sinkhorn(C_, eps_, f=None, g=None, iters=t1):
#         K_ = (-C_ / eps_).exp()                 # Gibbs
#         if f is None:
#             f = torch.zeros_like(p)
#             g = torch.zeros_like(q)
#         u = torch.exp(f / eps_)
#         v = torch.exp(g / eps_)
#         for _ in range(iters):
#             u = p / (K_ @ v + 1e-200)
#             v = q / (K_.t() @ u + 1e-200)
#         f = eps_ * torch.log(u);  g = eps_ * torch.log(v)
#         P = torch.diag(u) @ K_ @ torch.diag(v)
#         return P, f, g

#     # outer EPPA
#     X, *_ = log_sinkhorn(C, eps, iters=100)
#     f = g = None
#     for _ in range(K):
#         Ck = C - eps * torch.log(X + 1e-300)

#         # Phase‑A
#         X, f, g = log_sinkhorn(Ck, eps, f, g, iters=t1)

#         # Phase‑B
#         for _ in range(t2):
#             r = p - X.sum(1)
#             c = q - X.sum(0)
#             grad = torch.cat([r, c])
#             if grad.norm() < tol_outer:
#                 break

#             d_r = X.sum(1);           d_c = X.sum(0)
#             H_rr = torch.diag(d_r) / eps
#             H_cc = torch.diag(d_c) / eps
#             H_rc = X / eps            # (n,m)

#             # off‑diagonal sparsity
#             k_keep = int(rho * n * m)
#             flat = H_rc.flatten()
#             if k_keep < flat.numel():
#                 thresh = flat.abs().kthvalue(flat.numel() - k_keep).values
#                 mask = flat.abs() >= thresh
#                 H_rc = torch.where(mask.view_as(H_rc), H_rc, torch.zeros_like(H_rc))

#             def Hv(v):
#                 v_r, v_c = v[:n], v[n:]
#                 return torch.cat([H_rr @ v_r + H_rc @ v_c,
#                                   H_rc.t() @ v_r + H_cc @ v_c])

#             delta, _ = cg(Hv, -grad, tol=tol_cg, maxiter=500)
#             f_r, f_c = delta[:n], delta[n:]

#             alpha = 1.0
#             while alpha > 1e-4:
#                 f_new = f + alpha * f_r if f is not None else alpha * f_r
#                 g_new = g + alpha * f_c if g is not None else alpha * f_c
#                 X_new = torch.exp(((f_new[:, None] + g_new[None, :]) - Ck) / eps).clamp_min(1e-300)
#                 if max((X_new.sum(1)-p).abs().max(),
#                        (X_new.sum(0)-q).abs().max()) < (1-alpha/2) * grad.abs().max():
#                     f, g, X = f_new, g_new, X_new
#                     break
#                 alpha *= .5

#         if grad.norm() < tol_outer:
#             break

#     return X

# class EntropicOTFM:
#     def __init__(self, x, t_idx, dt, sigma, T, dim, device):
#         # def entropic_ot_plan(x0, x1, eps):
#         #     C = ot.utils.euclidean_distances(x0, x1, squared=True) / 2
#         #     p, q = torch.full((x0.shape[0],), 1 / x0.shape[0]), torch.full(
#         #         (x1.shape[0],), 1 / x1.shape[0]
#         #     )
#         #     return ot.sinkhorn(p, q, C, eps, method="sinkhorn", numItermax=10000)
#         # def entropic_ot_plan(x0, x1, eps):
#         #     C = ot.utils.euclidean_distances(x0, x1, squared = True) / 2
#         #     p, q = torch.full((x0.shape[0], ), 1 / x0.shape[0]), torch.full((x1.shape[0], ), 1 / x1.shape[0])
#         #     return ot.bregman.sinkhorn_epsilon_scaling(p.double(), q.double(), C.double(), eps, numItermax=10000).float()
        
#         # def entropic_ot_plan(x0, x1, device):
#         #     x0 = x0.to(device=device, dtype=torch.float64)
#         #     x1 = x1.to(device=device, dtype=torch.float64)

#         #     C = 0.5 * torch.cdist(x0, x1).pow(2)
#         #     C = C / C.mean()

#         #     n, m = C.shape
#         #     p = torch.full((n,), 1 / n, device=device, dtype=torch.float64)
#         #     q = torch.full((m,), 1 / m, device=device, dtype=torch.float64)

#         #     f = g = None                    # dual warm‑start
#         #     for eps in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
#         #         π, log = ot.bregman.sinkhorn_log(
#         #             p, q, C, reg=eps,
#         #             numItermax=10_000, stopThr=1e-9,
#         #             log=True,
#         #             init_dual=(f, g) if f is not None else None,
#         #         )
#         #         # extract duals for the next round
#         #         f, g = log["u"], log["v"]

#         #     return π 
#         self.sigma = sigma
#         self.bm = BridgeMatcher()
#         self.x = x
#         self.t_idx = t_idx
#         self.dt = dt
#         self.T = T
#         self.dim = dim
#         self.device = device
#         self.Ts = []
#         # construct EOT plans
#         for i in range(self.T - 1):
#             # self.Ts.append(
#             #     entropic_ot_plan(
#             #         self.x[self.t_idx == i, :],
#             #         self.x[self.t_idx == i + 1, :],
#             #         self.dt * self.sigma**2,
#             #     )
#             # )
#             self.Ts.append(
#                 pins_ot_plan(
#                     self.x[self.t_idx == i, :],
#                     self.x[self.t_idx == i + 1, :],
#                     eps=self.dt * self.sigma ** 2,     # same ε you used before
#                     device=self.device
#                 )
#             )

    # def sample_bridges_flows(self, batch_size=64, skip_time=None):
    #     _x = []
    #     _t = []
    #     _t_orig = []
    #     _s = []
    #     _u = []
    #     for i in range(self.T - 1):
    #         if skip_time is not None and (i == skip_time or i + 1 == skip_time):
    #             continue

    #         with torch.no_grad():
    #             x0, x1 = self.bm.sample_plan(
    #                 self.x[self.t_idx == i, :],
    #                 self.x[self.t_idx == i + 1, :],
    #                 self.Ts[i],
    #                 batch_size,
    #             )
    #         ts = torch.rand_like(x0[:, :1])
    #         _, _, x, s, u = self.bm.sample_bridge_and_flow(
    #             x0, x1, ts, (self.sigma**2 * self.dt) ** 0.5
    #         )
    #         _x.append(x)
    #         _s.append(s)
    #         _t.append((i + ts) * self.dt)
    #         _t_orig.append(ts)
    #         _u.append(u)
    #     return (
    #         torch.vstack(_x),
    #         torch.vstack(_s),
    #         torch.vstack(_u),
    #         torch.vstack(_t),
    #         torch.vstack(_t_orig),
    #     )
    
def build_entropic_otfms(adatas, T, sigma, dt):
    """
    Returns a list of EntropicOTFM objects, one per dataset.
    """
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
            normalize_C=True,
        )
        otfms.append(model)
    return otfms

import math

import torch
import torch.nn as nn


class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input_.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_linear={}, in_features={}, out_features={}, bias={}".format(
            self.num_linear, self.in_features, self.out_features, self.bias is not None
        )

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint as odeint


class NNODEF(nn.Module):
    def __init__(self, in_dim, hid_dim, time_invariant=True):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim + 1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):

        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out


class MLPODEF(nn.Module):
    def __init__(self, dims, GL_reg=0.01, bias=True, time_invariant=True):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super(MLPODEF, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # adaptive lasso parameter

        if time_invariant:
            self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        else:
            self.fc1 = nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias)

        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(
                LocallyConnected(dims[0], dims[l + 1], dims[l + 2], bias=bias)
            )
        self.fc2 = nn.ModuleList(layers)
        self.elu = nn.ELU(inplace=True)

        # Initialize a mask buffer for knockout. By default, no mask is applied
        self.register_buffer("ko_mask", torch.ones(self.fc1.weight.shape))

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]

        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        # we do a linear operation with the masked weights
        w = self.fc1.weight * self.ko_mask
        x = F.linear(x, w, self.fc1.bias)  # x: [n,1,d], w: [d*m1, d]

        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = fc(self.elu(x))  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        x = x.unsqueeze(dim=1)  # [n, 1, d]
        return x

    def l2_reg(self):
        """L2 regularization on all parameters"""
        reg = 0.0
        fc1_weight = self.fc1.weight  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def fc1_reg(self):
        """L1 regularization on input layer parameters"""
        return torch.sum(torch.abs(self.fc1.weight))

    def group_weights(self, gamma=0.5):
        """Group lasso weights"""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def causal_graph(self, w_threshold=0.3):  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i] or [j * m1, i+1] if time-varying

        if not self.time_invariant:
            # Remove the time dimension (last column) before reshaping
            fc1_weight = fc1_weight[:, :-1]  # [j * m1, i]

        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        W = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        W[np.abs(W) < w_threshold] = 0
        return np.round(W, 2)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()


class MLPODEF1(nn.Module):
    def __init__(
        self, dims, GL_reg=0.01, bias=True, time_invariant=True, knockout_masks=None
    ):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super(MLPODEF1, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims = dims
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # adaptive lasso parameter
        if time_invariant:
            self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        else:
            self.fc1 = nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(
                LocallyConnected(dims[0], dims[l + 1], dims[l + 2], bias=bias)
            )
        self.fc2 = nn.ModuleList(layers)
        self.elu = nn.ELU(inplace=True)
        self.knockout_masks = None
        if knockout_masks is not None:
            self.knockout_masks = [
                torch.tensor(m, dtype=torch.float32) for m in knockout_masks
            ]

        self._init_weights()

    def _init_weights(self):
        """Custom weight/bias initialisation."""
        for m in self.modules():
            # 1) nn.Linear ----------------------------------------------------
            if isinstance(m, nn.Linear):
                # He (fan_in) works well with ELU
                init.kaiming_normal_(m.weight, mode="fan_in",
                                    nonlinearity="relu")  # ELU≈ReLU for init
                if m.bias is not None:
                    init.zeros_(m.bias)

            # 2) LocallyConnected (same shape semantics as Linear) ------------
            elif isinstance(m, LocallyConnected):
                init.kaiming_normal_(m.weight, mode="fan_in",
                                     nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, t, x, dataset_idx=None):  # [n, 1, d] -> [n, 1, d]
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
        if dataset_idx is not None and self.knockout_masks is not None:
            mask = self.knockout_masks[dataset_idx].to(x.device)  # [d, d]
            x = x.squeeze(1)
            d = self.dims[0]
            m = self.dims[1]
            if self.time_invariant:
                w_raw = self.fc1.weight
                w_reshaped = w_raw.view(d, m, d)
                masked_w = w_reshaped * mask.unsqueeze(1)
                x_out = torch.einsum("rhd,nd->nrh", masked_w, x)
            else:
                w_raw = self.fc1.weight  # [d*m, d+1]
                w_vars = w_raw[:, :d]  # [d*m, d]
                w_time = w_raw[:, d:]  # [d*m, 1]
                w_vars_reshaped = w_vars.view(d, m, d)  # [d, m, d]
                mask = self.knockout_masks.to(x.device)  # [d, d]
                masked_w_vars = w_vars_reshaped * mask.unsqueeze(1)  # [d, m, d]
                x_vars = x[:, :d]  # [n, d]
                x_time = x[:, d:]  # [n, 1]
                out_vars = torch.einsum(
                    "rhd,nd->nrh", masked_w_vars, x_vars
                )  # [n, d, m]
                w_time_reshaped = w_time.view(d, m, 1)  # [d, m, 1]
                out_time = w_time_reshaped * x_time.unsqueeze(1)  # [n, d, m]
                x_out = out_vars + out_time
            if self.fc1.bias is not None:
                bias = self.fc1.bias.view(d, m)  # reshape bias to [d, m]
                x_out = x_out + bias.unsqueeze(0)  # broadcast over batch dimension
        else:
            x = self.fc1(x)
            x_out = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x_out = fc(self.elu(x_out))  # [n, d, m2]
        x_out = x_out.squeeze(dim=2)  # [n, d]
        x_out = x_out.unsqueeze(dim=1)  # [n, 1, d]
        return x_out

    def l2_reg(self):
        """L2 regularization on all parameters"""
        reg = 0.0
        fc1_weight = self.fc1.weight  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def fc1_reg(self):
        """L1 regularization on input layer parameters"""
        return torch.sum(torch.abs(self.fc1.weight))

    def group_weights(self, gamma=0.5):
        """Group lasso weights"""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def causal_graph(self, w_threshold=0.3):  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        W = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
        W = W.cpu().detach().cpu().numpy()  # [i, j]
        W[np.abs(W) < w_threshold] = 0
        return np.round(W, 2)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()

import copy
class MLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
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
    
class FMMLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
        time_varying=True,
    ):
        super().__init__()
        self.net = nn.Sequential()
        self.time_varying = time_varying
        assert len(hidden_sizes) > 0
        hidden_sizes = copy.copy(hidden_sizes)
        if time_varying:
            hidden_sizes.insert(0, d + 1)
        else:
            hidden_sizes.insert(0, d)
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

    def forward(self, t, x):
        if self.time_varying:
            return self.net(torch.hstack([x, t.expand(*x.shape[:-1], 1)]))
        else:
            return self.net(x)
        

from tqdm import tqdm # type: ignore

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
    mix_batches=True,
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
        MIX_BATCHES = mix_batches

        if not MIX_BATCHES:
            ds_idx = np.random.randint(0, len(adatas_list))
            model = otfms[ds_idx]
            cond_vector = cond_matrix[ds_idx]

            _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
                batch_size=batch_size, skip_time=skip_time
            )

            B = _x.shape[0]
            cond_expanded = cond_vector.repeat(B // cond_vector.shape[0] + 1, 1)[:B]
        else:
            all_x, all_s, all_u, all_t, all_t_orig = [], [], [], [], []
            all_cond_vectors = []
            n_datasets = len(adatas_list)
            base_batch_size = batch_size // n_datasets
            remainder = batch_size % n_datasets
            # Create batch sizes list with equal distribution
            batch_sizes = [base_batch_size] * n_datasets

            # Distribute remainder randomly
            if remainder > 0:
                indices = torch.randperm(n_datasets)[:remainder]
                for idx in indices:
                    batch_sizes[idx] += 1

            for ds_idx in range(n_datasets):
                model = otfms[ds_idx]
                cond_vector = cond_matrix[ds_idx]

                _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
                    batch_size=batch_sizes[ds_idx], skip_time=skip_time
                )
                actual_batch_size = _x.shape[0]
                all_x.append(_x)
                all_s.append(_s)
                all_u.append(_u)
                all_t.append(_t)
                all_t_orig.append(_t_orig)

                cond_expanded = cond_vector.repeat(
                    actual_batch_size // cond_vector.shape[0] + 1, 1
                )[:actual_batch_size]
                all_cond_vectors.append(cond_expanded)
            # Combine all samples using vstack
            _x = torch.vstack(all_x)
            _s = torch.vstack(all_s)
            _u = torch.vstack(all_u)
            _t = torch.vstack(all_t)
            _t_orig = torch.vstack(all_t_orig)
            cond_expanded = torch.vstack(all_cond_vectors)

            B = _x.shape[0]

        optim.zero_grad()
        # Reshape inputs for MLPODEF
        s_input = _x.unsqueeze(1)
        v_input = _x.unsqueeze(1)
        t_input = _t.unsqueeze(1)

        # Get model outputs and reshape
        s_fit = func_s(_t, _x, cond_expanded).squeeze(1)
        # v_fit = v(t_input, v_input).squeeze(1)
        if i <= 500:
            v_fit = func_v(t_input, v_input).squeeze(1) - model.sigma**2/2 * func_s(_t, _x, cond_expanded)
        else:
            v_fit = func_v(t_input, v_input).squeeze(1) + v_correction(_t, _x)
            v_fit = v_fit - model.sigma**2/2 * func_s(_t, _x, cond_expanded)

        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * model.dt - _u) ** 2)

        L_reg = func_v.l2_reg() + func_v.fc1_reg()
        L_reg_correction = mlp_l2_reg(v_correction)
        if i < 100: # train score for first few iters 
            L = alpha * L_score
        elif i >= 100 and i <= 500:
            L = alpha * L_score + (1 - alpha) * L_flow + reg * L_reg 
        else:
            L = alpha * L_score + (1 - alpha) * L_flow + reg * L_reg + correction_reg_strength * L_reg_correction

        with torch.no_grad():
            if i % 100 == 0:
                print(f"step={i}, dataset={ds_idx}, L_score={L_score.item():.4f}, L_flow={L_flow.item():.4f}, "
                    f"NGM_Reg={L_reg.item():.4f}, MLP_Reg={L_reg_correction.item():.4f}")
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

    return loss_history, score_loss_history, flow_loss_history, reg_loss_history, reg_corr_loss_history, func_v, func_s, v_correction


def main():
    estim_alt = rf.Estimator(_adatas, _kos, 
                lr = options["lr"],
                reg_sinkhorn = options["reg_sinkhorn"], 
                reg_A = options["reg_A"], 
                reg_A_elastic = options["reg_A_elastic"], 
                iter = options["iter"], 
                ot_coupling = options["ot_coupling"],
                optimizer = options["optimizer"], 
                n_pca_components = options["n_pca_components"])
    estim_alt.fit(print_iter=100, alg = "alternating", update_couplings_iter=500)

    _A = pd.DataFrame(estim_alt.A, index = adata_tf.var.index, columns=adata_tf.var.index)
    genes_common = pd.Index(set(A_ref.columns).intersection(set(_A.index)))
    tfs = A_ref.index
    tfs_no_ko = [i for i in tfs if i not in _kos]
    tfs_ko = [i for i in tfs if i in _kos]

    _tfs = tfs
    _thresh = 0

    A_renge = pd.read_csv("./data/Renge/A_renge_output.csv", index_col=0)
    
    batch_size = 128
    n = _adatas[0].X.shape[1]
    ko_indices = {ko: idx for idx, ko in enumerate(_kos)}

    _tfs = tfs
    _thresh = 0

    conditionals = []
    for i, ad in enumerate(_kos):
        cond_matrix = torch.zeros(batch_size, n)
        if ad is not None:
            cond_matrix[:, i] = 1
        conditionals.append(cond_matrix)

    knockout_masks = []
    for i, ad in enumerate(_adatas):
        d = ad.X.shape[1]
        mask_i = build_knockout_mask(d, list(ko_indices.values())[i])  # returns [d,d]
        knockout_masks.append(mask_i)

    dims = [n, 128, 1]
    t = 4

    func_v = MLPODEF1(
            dims=dims, GL_reg= 0.04, bias=True, knockout_masks=knockout_masks
        )
    score_net = MLP(
        d=n,
        hidden_sizes=[128, 128],
        time_varying=True,
        conditional=True,
        conditional_dim=n,
    )

    v_cor = FMMLP(d=n, hidden_sizes=[128,128], time_varying=True)

    otfms = build_entropic_otfms(_adatas, t, sigma=1.0, dt=1 / t)

    loss_history, score_loss_history, flow_loss_history, reg_loss_history, reg_corr_loss_history, flow_model, corr_model, score_model = train_with_fmot_scorematching(
            func_v=func_v,
            func_s=score_net,
            v_correction=v_cor,
            adatas_list=_adatas,
            otfms=otfms,
            cond_matrix=conditionals,
            alpha=0.1,
            reg=1e-6,
            n_steps=10000,
            batch_size=batch_size,
            device="cpu",
            lr=3e-3,
            mix_batches=mixed_batches,
        )
    
    W_v = func_v.causal_graph()
    W_v_df = pd.DataFrame(W_v, index = adata_tf.var.index, columns=adata_tf.var.index)

    W_v_df.to_csv(f"W_v_sf2m__mix:{mixed_batches}_{seed}_ogsetting.csv", index=True, header=True)

    vals = np.abs(W_v_df.loc[_tfs, genes_common]).values.flatten()

    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score, auc
    ys = {}
    y_true = (A_ref.loc[_tfs, genes_common] > _thresh).values.flatten()
    ys["RF (KO)"] = np.abs(_A.loc[_tfs, genes_common]).values.flatten()
    ys["RENGE"] = np.abs(A_renge.loc[_tfs, genes_common]).values.flatten()
    ys["StrcutureFlow"] = np.abs(W_v_df.loc[_tfs, genes_common]).values.flatten()

    records = []
    plt.figure(figsize = (2*PLT_CELL, PLT_CELL))
    for (k, y) in ys.items():
        prec, rec, thresh = precision_recall_curve(y_true, y)
        avg_prec = average_precision_score(y_true, y)
        plt.plot(rec, prec, label=f'{k} (AUPR = {avg_prec:.2f}, AUPR ratio = {avg_prec / y_true.mean():.2f})')
        # PR
        aupr = auc(rec, prec)
        rocauc = roc_auc_score(y_true, y)

        records.append({
        "method":   k,
        "AP":       avg_prec,
        "AUPR":     aupr,
        "ROC_AUC":  rocauc,
        "AP Ratio": (avg_prec / y_true.mean()),
        "ROC_AUC_ratio": (rocauc / 0.5)
        })

    # build DataFrame and save
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(f"curve_metrics__mix:{mixed_batches}_{seed}_ogsetting.csv", index=False)
    print("Saved metrics to curve_metrics.csv")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./scCRISPR_AUPR_all_{seed}_ogsetting.pdf")

    # A = pd.DataFrame(np.asarray(W_v), index = adata_tf.var.index, columns = adata_tf.var.index)
    # g = nx.DiGraph(np.abs(A))
    # centralities = nx.centrality.eigenvector_centrality(g.reverse(), weight = 'weight') # calcualte first centrality without thresholding
    # _centralities = pd.Series(centralities)
    # plt.figure(figsize = (PLT_CELL, PLT_CELL))
    # sb.barplot(_centralities.sort_values()[::-1][:25], orient = 'h')
    # plt.yticks(fontsize=8);
    # ax = plt.gca()
    # y_ticks = ax.get_yticks()
    # y_tick_labels = ax.get_yticklabels()
    # for label in y_tick_labels:
    #     if label.get_text() in _kos:
    #         label.set_color('red')
    # ax.set_yticklabels(y_tick_labels);
    # plt.xlabel("Out-edge eigencentrality")
    # plt.tight_layout()
    # plt.savefig("./scCRISPR_centrality.pdf")

    # sb.clustermap(np.log1p(A_ref > _thresh).loc[tfs, genes_common].iloc[cg.dendrogram_row.reordered_ind, cg.dendrogram_col.reordered_ind], figsize = (5, 5), row_cluster = False, col_cluster= False)
    
if __name__ =="__main__":
    main()
