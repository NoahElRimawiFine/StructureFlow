from __future__ import annotations
import numpy as np
import pandas as pd
import sys
import sklearn
import copy
from pathlib import Path
import anndata as ad, scanpy as sc
from typing import Dict, Any
import argparse
import torch
import random
import dcor
import glob
import os
import scipy.sparse as sp
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from functools import partial
import ot
from sklearn.metrics.pairwise import pairwise_kernels
import math

from otvelo.utils_Velo import *

from otvelo.utils import *
import scipy
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
import umap

parser = argparse.ArgumentParser(description="OTVelo baseline script")

parser.add_argument(
    "--subset",
    choices=["wt", "ko", "all"],
    default="wt",
    help="which replicates to load: " "'wt' (wild‑type), 'ko' (knock‑outs), or 'all'",
)
parser.add_argument(
    "--backbone",
    type=str,
    default="dyn-BF",
    help="backbone name, e.g. 'dyn-BF', 'dyn-TF'",
)

parser.add_argument(
    "--seed", type=int, default=42, help="random seed for reproducibility"
)

parser.add_argument(
    "--dataset",
    choices=["synthetic", "renge"],
    default="synthetic",
    help="'synthetic' (default) or 'renge' real dataset",
)


args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random.seed(seed)
sklearn.utils.check_random_state(seed)


if args.backbone not in ["dyn-BF", "dyn-TF", "dyn-SW", "dyn-CY", "dyn-LL"]:
    ROOT_SYN = Path(__file__).resolve().parents[2] / "data" / "Curated"
else:
    ROOT_SYN = Path(__file__).resolve().parents[2] / "data" / "Synthetic"
backbone = args.backbone


def discover_subset(root: Path, backbone: str, subset: str):
    """
    Parameters
    ----------
    root      Synthetic/ directory that contains all backbones
    backbone  e.g. 'dyn-BF', 'dyn-TF'
    subset    'wt' | 'ko' | 'all'
    Returns   list[Path] replicate directories
    """

    wt_dirs = [root / backbone] if (root / backbone).is_dir() else []

    ko_dirs = [
        d for d in root.iterdir() if d.is_dir() and d.name.startswith(f"{backbone}_ko")
    ]

    print(f"[INFO] found {len(wt_dirs)} WT dir(s) and {len(ko_dirs)} KO dir(s)")

    if subset == "wt":
        return wt_dirs
    elif subset == "ko":
        return ko_dirs
    elif subset == "all":
        return wt_dirs + ko_dirs
    else:
        raise ValueError("--subset must be 'wt', 'ko', or 'all'")


paths = discover_subset(ROOT_SYN, backbone=backbone, subset=args.subset)
print(f"[INFO] loading {len(paths)} replicate(s): {[p for p in paths]}")


def _actual_dataset_dir(ds: Path) -> Path:
    """
    Handle the extra nesting like  data/Synthetic/dyn-BF/dyn-BF-1000-1/.
    If ds contains a single sub‑directory with ExpressionData.csv inside,
    we descend automatically. Otherwise we assume ds is already correct.
    """
    subdirs = [p for p in ds.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / "ExpressionData.csv").exists():
        return subdirs[0]
    return ds


def load_adata(path: Path) -> ad.AnnData:
    adata = ad.AnnData(pd.read_csv(path / "ExpressionData.csv", index_col=0).T)
    df_pt = pd.read_csv(path / "PseudoTime.csv", index_col=0)
    df_pt[np.isnan(df_pt)] = 0
    adata.obs["t_sim"] = np.max(df_pt.to_numpy(), -1)
    sc.pp.log1p(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    return adata


def bin_timepoints(adata: ad.AnnData, t_bins: np.ndarray) -> None:
    adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1


def dense_array(mat) -> np.ndarray:
    """Return a plain 2-D float32 ndarray from any AnnData .X slice."""
    if sp.issparse(mat):  # CSR/CSC etc.
        return mat.toarray().astype(np.float32)
    if isinstance(mat, np.matrix):  # numpy.matrix
        return np.asarray(mat, dtype=np.float32)
    return np.asarray(mat, dtype=np.float32)


def to_otvelo_arrays(adata: ad.AnnData):
    counts_all = []
    labels_vec = []
    for tb in sorted(adata.obs["t"].unique()):
        sl = adata[adata.obs.t == tb]
        X = dense_array(sl.X)  # (n_cells × n_genes)
        if X.ndim != 2:
            raise ValueError(f"slice at t={tb} is not 2-D, got shape {X.shape}")

        X = X.T
        counts_all.append(X)
        labels_vec.extend([tb] * X.shape[1])

    counts = np.concatenate(counts_all, axis=1)
    labels = np.asarray(labels_vec)[None, :]
    Nt = len(counts_all)
    return counts_all, counts, labels, Nt


def load_renge_dataset(
    n_bins: int = 4, data_dir: Path = Path("../../data/Renge")
) -> dict[str, Any]:
    """
    Returns the same dict keys as load_synth_dataset()
    """
    X = pd.read_csv("../../data/Renge/X_renge_d2_80.csv", index_col=0)
    E = pd.read_csv("../../data/Renge/E_renge_d2_80.csv", index_col=0)

    adata_ = ad.AnnData(E)
    adata_.obs["condition"] = None
    adata_.obs.loc[X.index[X.iloc[:, :-1].T.sum(0) == 0], "condition"] = "wt"
    idx_ko = X.index[X.iloc[:, :-1].T.sum(0) == 1]
    adata_.obs.loc[idx_ko, "condition"] = X.columns[
        np.argmax(X.loc[idx_ko, :].iloc[:, :-1], -1)
    ]
    sc.pp.pca(adata_)
    sc.pp.neighbors(adata_)
    adata_.obs["t"] = X.t
    adata = sc.read_h5ad(data_dir / "hipsc.h5ad")
    # use gene symbols as var_names
    dists_ko = pd.Series(
        {
            k: dcor.energy_distance(
                adata.obsm["X_pca"][adata.obs.ko == "WT", :],
                adata.obsm["X_pca"][adata.obs.ko == k, :],
            )
            for k in adata.obs.ko.unique()
        }
    )
    adata_tf = adata.copy()
    mask = adata_tf.var["gene"].isin(E.columns).values
    adata_tf = adata_tf[:, mask].copy()
    adata_tf.var_names = adata_tf.var["gene"]

    _kos = list(dists_ko.sort_values()[::-1][range(8)].index)

    _adatas = []
    for k in _kos:
        _adatas.append(adata_tf[adata_tf.obs.ko == k, :].copy())
        _adatas[-1].X = np.asarray(_adatas[-1].X.todense(), dtype=np.float64)
        _adatas[-1].obs.t -= 2
        _adatas[-1].var.index = _adatas[-1].var.gene
    if _kos[0] == "WT":
        _kos[0] = None

    # Construct reference
    refs = {}
    for f in glob.glob("../../data/Renge/chip_1kb/*.tsv"):
        print(f)
        gene = os.path.splitext(os.path.basename(f))[0].split(".")[0]
        df = pd.read_csv(f, sep="\t")
        df.index = df.Target_genes
        # if len(df.columns[df.columns.str.contains("iPS_cells|ES_cells")]) == 0:
        #     print(pd.unique(df.columns.str.split("|").str[1]))
        y = pd.Series(
            df.loc[:, df.columns.str.contains("iPS_cells")].values.mean(-1),
            index=df.index,
        )
        # y = pd.Series(df.iloc[:, 2:].values.mean(-1), index = df.index)
        # y = pd.Series(df.iloc[:, 1], index = df.index)
        refs[gene] = y

    A_ref = pd.DataFrame(refs).T
    A_ref[np.isnan(A_ref.values)] = 0
    print(A_ref)

    tfs = A_ref.index
    tfs_no_ko = [i for i in tfs if i not in _kos]
    tfs_ko = [i for i in tfs if i in _kos]

    # build counts_all
    print(adata_tf)
    counts_all, counts, labels, Nt = to_otvelo_arrays(adata_tf)
    print("counts_all lens :", [c.shape for c in counts_all])
    print("counts shape     :", counts.shape)
    print("labels shape     :", labels.shape)
    print("Nt               :", Nt)
    assert counts.shape[1] == labels.shape[1]

    # no ground–truth GRN available → None
    return dict(
        adata=adata_tf,
        counts_all=counts_all,
        counts=counts,
        labels=labels,
        Nt=Nt,
        ref_network=A_ref,
    )


def load_synth_dataset(name: str, n_bins: int = 5) -> Dict[str, Any]:
    """
    Main entry point.
    Parameters
    ----------
    name     e.g. 'dyn-BF', 'dyn-BF_ko_g2', ...
    Returns
    -------
    dict with
        adata          AnnData (log1p counts, t_sim, t_bin)
        counts_all     List[ndarray]  (n_genes × n_cells_t)
        counts         ndarray (concat counts)
        labels         1 × n_cells int64
        Nt             number of time bins
        ref_network    pd.DataFrame or None (if refNetwork.csv exists)
    """
    base = ROOT_SYN / name
    ds_dir = _actual_dataset_dir(base)
    if not (ds_dir / "ExpressionData.csv").exists():
        raise FileNotFoundError(f"No ExpressionData.csv in {ds_dir}")

    t_bins = np.linspace(0, 1, n_bins + 1)[:-1]

    adata = load_adata(ds_dir)

    # loading the reference network
    true_mat = None
    ref_path = ds_dir / "refNetwork.csv"
    if ref_path.exists():
        df_ref = pd.read_csv(ref_path)
        genes = adata.var.index.to_list()
        g2i = {g: i for i, g in enumerate(genes)}

        true_mat = np.zeros((len(genes), len(genes)), dtype=np.int8)

        for tgt, src, sign in df_ref.itertuples(index=False):
            if src not in g2i or tgt not in g2i:
                continue
            i, j = g2i[tgt], g2i[src]
            true_mat[i, j] = 1 if sign == "+" else -1

    bin_timepoints(adata, t_bins=t_bins)

    counts_all, counts, labels, Nt = to_otvelo_arrays(adata)

    print("counts_all lens :", [c.shape for c in counts_all])
    print("counts shape     :", counts.shape)
    print("labels shape     :", labels.shape)
    print("Nt               :", Nt)
    assert counts.shape[1] == labels.shape[1]

    return dict(
        adata=adata,
        counts_all=counts_all,
        counts=counts,
        labels=labels,
        Nt=Nt,
        ref_network=true_mat,
    )


def loocv_score_single(
    left_out_idx,
    counts,  # genes × cells
    labels,  # 1 × cells
    Nt,
    dt,
    pca,
    alpha,
    eps_samp,
):
    """
    Remove one column from `counts`, re-fit couplings, then predict the
    left-out column’s next-time-step expression profile via barycentric
    projection.  Return reconstruction MSE in the original gene space.
    """
    keep_mask = np.ones(counts.shape[1], dtype=bool)
    keep_mask[left_out_idx] = False

    counts_train = counts[:, keep_mask]
    labels_train = labels[:, keep_mask]

    Ts_prior, _ = solve_prior(
        counts_train, counts_train, Nt, labels_train, eps_samp=eps_samp, alpha=alpha
    )

    t0 = labels[0, left_out_idx]  # its time bin
    if t0 == Nt - 1:  # last bin → skip
        return np.nan

    same_bin_mask = (labels_train == t0).flatten()
    X_bin = counts_train[:, same_bin_mask]  # genes × n_bin
    proj = X_bin @ Ts_prior[t0].T
    pred_col = proj[
        :, np.argmin(np.linalg.norm(proj - counts[:, left_out_idx][:, None], axis=0))
    ]

    mse_gene = mean_squared_error(counts[:, left_out_idx], pred_col)
    mse_pca = mean_squared_error(
        pca.transform(counts[:, left_out_idx][None, :]),
        pca.transform(pred_col[None, :]),
    )
    return mse_gene, mse_pca


def wasserstein(
    x0: torch.Tensor, x1: torch.Tensor, method: str = "exact", reg: float = 0.05
) -> float:
    """Compute Wasserstein-2 distance between two distributions."""
    # Set up the OT function
    if method == "exact":
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get uniform weights for the samples
    a = ot.unif(x0.shape[0])
    b = ot.unif(x1.shape[0])

    # Reshape if needed
    if x0.ndim > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.ndim > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    # Compute cost matrix (squared Euclidean distance)
    M = torch.cdist(x0, x1) ** 2

    # Compute Wasserstein distance
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)

    # Return square root for W2 distance
    return math.sqrt(ret)


def rbf_kernel(X, Y, gamma=None):
    if X.dim() > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2:
        Y = Y.reshape(Y.shape[0], -1)
    d = X.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    dist_sq = torch.cdist(X, Y) ** 2
    K = torch.exp(-gamma * dist_sq)
    return K, gamma


def mmd_squared(X, Y, kernel=rbf_kernel, sigma_list=None, **kernel_args):
    if X.dim() > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2:
        Y = Y.reshape(Y.shape[0], -1)

    if sigma_list is None:
        sigma_list = [0.01, 0.1, 1, 10, 100]

    mmd_values = []

    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)

        K_XX, _ = kernel(X, X, gamma=gamma)
        K_YY, _ = kernel(Y, Y, gamma=gamma)
        K_XY, _ = kernel(X, Y, gamma=gamma)

        term1 = K_XX.mean()
        term2 = K_YY.mean()
        term3 = K_XY.mean()

        mmd2 = term1 + term2 - 2 * term3
        mmd_values.append(mmd2.clamp(min=0))

    avg_mmd = torch.stack(mmd_values).mean().item()
    return avg_mmd


def solve_prior_strict(
    counts, counts_pca, Nt, labels, eps_samp, alpha=0.5, normalize_C=False
):
    """
    Like utils_Velo.solve_prior but:
      - skips any interval where one side is empty
      - if do_bridge=True, builds an fGW bridge on the first missing interval
    """
    import ot
    from ot.gromov import entropic_fused_gromov_wasserstein

    Ts, log = [None] * (Nt - 1), [np.nan] * (Nt - 1)
    bridged = False

    for t in range(Nt - 1):
        idx_t = np.where(labels == t)[1]
        idx_t1 = np.where(labels == t + 1)[1]

        # Case A: both sides nonempty -> normal fGW
        if len(idx_t) and len(idx_t1):
            X1 = counts[:, idx_t].T
            X2 = counts[:, idx_t1].T
            M = ot.dist(counts_pca[:, idx_t].T, counts_pca[:, idx_t1].T)
            if normalize_C:
                M /= M.max()
            # graph distances
            nnb = max(1, min(int(0.2 * X1.shape[0]), int(0.2 * X2.shape[0]), 50))
            D1 = compute_graph_distances(X1, nnb, metric="euclidean")
            D2 = compute_graph_distances(X2, nnb, metric="euclidean")
            if D1.max() and D2.max():
                D1 /= D1.max()
                D2 /= D2.max()
            else:
                a = 0.0
            T, logt = entropic_fused_gromov_wasserstein(
                M, D1, D2, epsilon=eps_samp, alpha=alpha, log=True
            )
            Ts[t] = T / T.sum()
            log[t] = logt["fgw_dist"]
            continue

        # Case B: interval touches held-out -> bridge *once* if possible
        if not bridged and len(idx_t) == 0 and len(idx_t1):
            # build bridge between (t-1) and (t+1)
            if (
                t > 0
                and t + 1 < Nt
                and np.any(labels == t - 1)
                and np.any(labels == t + 1)
            ):
                idx_prev = np.where(labels == t - 1)[1]
                idx_next = np.where(labels == t + 1)[1]
                X1 = counts[:, idx_prev].T
                X2 = counts[:, idx_next].T
                M = ot.dist(counts_pca[:, idx_prev].T, counts_pca[:, idx_next].T)
                if normalize_C:
                    M /= M.max()
                nnb = max(1, min(int(0.2 * X1.shape[0]), int(0.2 * X2.shape[0]), 50))
                D1 = compute_graph_distances(X1, nnb, metric="euclidean")
                D2 = compute_graph_distances(X2, nnb, metric="euclidean")
                if D1.max() and D2.max():
                    D1 /= D1.max()
                    D2 /= D2.max()
                else:
                    a = 0.0
                T, logt = entropic_fused_gromov_wasserstein(
                    M, D1, D2, epsilon=eps_samp, alpha=alpha, log=True
                )
                Ts[t] = T / T.sum()
                log[t] = logt["fgw_dist"]
                bridged = True
            # otherwise leave Ts[t] = None
        # else leave Ts[t] = None

    return Ts, log


def predict_via_velocity(counts_all, Ts, t_star):
    """
    Predict x_t from x_(t-1) using velocity estimated from bridge coupling.

    OTVelo-style approach:
    1. Use the bridge coupling (t-1 → t+1) to estimate velocity
    2. Take one Euler step from t-1 to predict t

    Args:
        counts_all: List of count matrices (genes × n_cells) for each time point
        Ts: List of coupling matrices from solve_prior_strict
        t_star: The time point to predict

    Returns:
        X_pred_t: Predicted gene expression at time t_star (genes × n_cells)
    """
    T_bridge = Ts[t_star]
    if T_bridge is None:
        raise RuntimeError(f"No bridge coupling for t*={t_star}")

    X_tminus1 = counts_all[t_star - 1]  # genes × n_cells (at t-1)
    X_tplus1 = counts_all[t_star + 1]  # genes × n_cells (at t+1)
    n_src = X_tminus1.shape[1]
    n_tgt = X_tplus1.shape[1]

    # Project t-1 cells forward to t+1 using the coupling
    # This gives us where each cell at t-1 would end up at t+1
    if T_bridge.shape == (n_src, n_tgt):
        # Standard: rows = src (t-1), cols = tgt (t+1)
        X_projected_tplus1 = (T_bridge @ X_tplus1.T).T  # (genes × n_src)
    elif T_bridge.shape == (n_tgt, n_src):
        # Transposed: rows = tgt, cols = src
        X_projected_tplus1 = (T_bridge.T @ X_tplus1.T).T  # (genes × n_src)
    else:
        raise ValueError(
            f"Coupling shape {T_bridge.shape} doesn't match "
            f"(n_src={n_src}, n_tgt={n_tgt})"
        )

    # Estimate velocity: change from t-1 to t+1, divided by 2 time steps
    # velocity = (X_tplus1 - X_tminus1) / (2 * dt)
    # Since dt=1, velocity = (X_projected_tplus1 - X_tminus1) / 2
    velocity = (X_projected_tplus1 - X_tminus1) / 2.0

    # Take ONE Euler step from t-1 to t
    # x_t = x_(t-1) + velocity * dt (dt=1 for one time step)
    X_pred_t = X_tminus1 + velocity

    return X_pred_t  # (genes × n_cells)


def otvelo_loto_one_fold(counts_all, adatas, t_star, *, eps=1e-2, alpha=0.5, pca):
    Nt = len(counts_all)

    # drop held-out t* cells for training
    mask = np.concatenate(
        [np.full(blk.shape[1], tt != t_star) for tt, blk in enumerate(counts_all)]
    )
    counts_flat = np.hstack(counts_all)[:, mask]
    labels_flat = np.concatenate(
        [np.full(blk.shape[1], tt) for tt, blk in enumerate(counts_all)]
    )[mask][None, :]
    counts_pca = pca.transform(counts_flat.T).T

    # strict solve_prior with skip+bridge
    Ts, _ = solve_prior_strict(
        counts_flat,
        counts_pca,
        Nt,
        labels_flat,
        eps_samp=eps,
        alpha=alpha,
        normalize_C=True,
    )

    # predict via velocity-based one-step prediction
    X_pred = predict_via_velocity(counts_all, Ts, t_star)
    X_true = counts_all[t_star]

    # compute per-dataset metrics
    cell_counts = [adata[adata.obs.t == t_star].shape[0] for adata in adatas]

    per_dataset_metrics = []
    start_idx = 0

    for i, cell_count in enumerate(cell_counts):
        end_idx = start_idx + cell_count

        X_pred_i = X_pred[:, start_idx:end_idx]
        X_true_i = X_true[:, start_idx:end_idx]

        Xp_i = pca.transform(X_pred_i.T)
        Xt_i = pca.transform(X_true_i.T)
        Xp_i = torch.from_numpy(Xp_i).float()
        Xt_i = torch.from_numpy(Xt_i).float()

        w2_i = wasserstein(Xp_i, Xt_i)
        _, gamma_i = rbf_kernel(Xp_i, Xt_i)
        mmd_i = mmd_squared(Xp_i, Xt_i, gamma=gamma_i)

        per_dataset_metrics.append(
            {
                "dataset_idx": i,
                "w2": w2_i,
                "mmd2": mmd_i,
            }
        )

        print(f"  Dataset {i}: W₂={w2_i:.4f}, MMD²={mmd_i:.4e}")

        start_idx = end_idx

    # Average metrics across datasets
    avg_w2 = np.mean([m["w2"] for m in per_dataset_metrics])
    avg_mmd = np.mean([m["mmd2"] for m in per_dataset_metrics])

    return avg_w2, avg_mmd, X_pred, per_dataset_metrics


def create_multi_ko_pca_plot_wgrey(
    full_adatas,
    predictions_dict,
    ko_names,
    held_out_time,
    folder_path,
    model_type,
    dataset_type="Synthetic",
):
    os.makedirs(folder_path, exist_ok=True)

    if len(ko_names) < 3:
        print(
            "Not enough knockout conditions to create multi-KO plot (need at least 3)"
        )
        return

    ko_indices_to_plot = list(range(min(3, len(ko_names))))

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))

    all_data = np.vstack([adata.X for adata in full_adatas])

    if dataset_type == "Renge":
        print("Computing UMAP for Renge data (need to project predictions)")

        pca_reducer = PCA(n_components=min(50, all_data.shape[0], all_data.shape[1]))
        all_data_pca = pca_reducer.fit_transform(all_data)

        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
        )
        reducer.fit(all_data_pca)
        reduction_method = "UMAP"

        reducer.pca_reducer = pca_reducer
    else:
        reducer = PCA(n_components=2)
        reducer.fit(all_data)
        reduction_method = "PCA"

    plt.rcParams.update(
        {
            "font.size": 38,
            "axes.titlesize": 44,
            "axes.labelsize": 44,
            "xtick.labelsize": 38,
            "ytick.labelsize": 38,
            "legend.fontsize": 38,
        }
    )

    highlight_color = "#E41A1C"
    prediction_color = "#377EB8"

    if dataset_type == "Renge":
        all_data_reduced = reducer.transform(reducer.pca_reducer.transform(all_data))
    else:
        all_data_reduced = reducer.transform(all_data)
    x_min, x_max = all_data_reduced[:, 0].min(), all_data_reduced[:, 0].max()
    y_min, y_max = all_data_reduced[:, 1].min(), all_data_reduced[:, 1].max()

    for i, (ax, ko_idx) in enumerate(zip(axes, ko_indices_to_plot)):
        adata = full_adatas[ko_idx]
        ko_name = ko_names[ko_idx]

        # Clean up knockout name for display (extract just "gX" from "dyn-TF_ko_gX")
        is_knockout = ko_name and "_ko_" in ko_name
        if is_knockout:
            # Extract the gene identifier after "_ko_"
            ko_display_name = ko_name.split("_ko_")[-1]
        else:
            ko_display_name = ko_name

        times = adata.obs["t"].values
        if dataset_type == "Renge":
            ko_data_reduced = reducer.transform(reducer.pca_reducer.transform(adata.X))
        else:
            ko_data_reduced = reducer.transform(adata.X)

        if ko_idx in predictions_dict:
            predictions = predictions_dict[ko_idx]
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()

            if dataset_type == "Renge":
                pred_reduced = reducer.transform(
                    reducer.pca_reducer.transform(predictions)
                )
            else:
                pred_reduced = reducer.transform(predictions)
        else:
            print(f"No predictions available for KO: {ko_name}")
            continue

        if dataset_type == "Renge":
            data_start_idx = 0
            max_time_global = max(
                max(adata_temp.obs["t"].values) for adata_temp in full_adatas
            )

            for adata_idx, adata_temp in enumerate(full_adatas):
                temp_times = adata_temp.obs["t"].values
                temp_data_size = adata_temp.X.shape[0]
                temp_data_reduced = all_data_reduced[
                    data_start_idx : data_start_idx + temp_data_size
                ]

                for t in sorted(set(temp_times)):
                    t_mask = temp_times == t
                    shade = 0.3 + 0.4 * (t / max_time_global)
                    gray_color = str(1 - shade)

                    ax.scatter(
                        temp_data_reduced[t_mask, 0],
                        temp_data_reduced[t_mask, 1],
                        c=gray_color,
                        s=20,
                        alpha=0.6,
                        label=(
                            f"All data t={t}"
                            if i == 0 and adata_idx == 0 and t == min(temp_times)
                            else None
                        ),
                    )

                data_start_idx += temp_data_size

            is_held_out = times == held_out_time
            if any(is_held_out):
                ax.scatter(
                    ko_data_reduced[is_held_out, 0],
                    ko_data_reduced[is_held_out, 1],
                    c=highlight_color,
                    s=120,
                    edgecolors="black",
                    linewidth=1,
                    label=f"KO {ko_display_name} t={held_out_time}" if i == 0 else None,
                )

            ax.scatter(
                pred_reduced[:, 0],
                pred_reduced[:, 1],
                c=prediction_color,
                s=120,
                marker="x",
                linewidth=3,
                label="Predictions" if i == 0 else None,
            )

        else:
            is_held_out = times == held_out_time

            for t in sorted(set(times)):
                if t == held_out_time:
                    continue

                t_mask = times == t
                shade = 0.3 + 0.5 * (t / max(times))
                gray_color = str(1 - shade)

                ax.scatter(
                    ko_data_reduced[t_mask, 0],
                    ko_data_reduced[t_mask, 1],
                    c=gray_color,
                    s=60,
                    label=f"t={t}" if i == 0 and t == min(times) else None,
                )

            if any(is_held_out):
                ax.scatter(
                    ko_data_reduced[is_held_out, 0],
                    ko_data_reduced[is_held_out, 1],
                    c=highlight_color,
                    s=80,
                    label=f"t={held_out_time} (held out)" if i == 0 else None,
                )

            ax.scatter(
                pred_reduced[:, 0],
                pred_reduced[:, 1],
                c=prediction_color,
                s=100,
                marker="x",
                linewidth=2,
                label="Predictions" if i == 0 else None,
            )

        component1_label = "UMAP1" if reduction_method == "UMAP" else "PC1"
        component2_label = "UMAP2" if reduction_method == "UMAP" else "PC2"

        ax.set_xlabel(component1_label, fontsize=44)
        if i == 0:
            ax.set_ylabel(component2_label, fontsize=44)

        ko_label = f"Knockout {ko_display_name}" if is_knockout else "Observational"
        ax.set_title(ko_label, pad=15, fontsize=44)

        ax.grid(True, alpha=0.3, linestyle="--")

        for spine in ax.spines.values():
            spine.set_visible(True)

    for ax in axes:
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    plt.tight_layout()

    plt.subplots_adjust(bottom=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=min(len(handles), 3),
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=38,
    )

    filename_base = f"multi_ko_comparison_{model_type}_holdout_{held_out_time}"
    plt.savefig(os.path.join(folder_path, f"{filename_base}.pdf"), bbox_inches="tight")
    plt.savefig(
        os.path.join(folder_path, f"{filename_base}.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()

    plt.rcParams.update(plt.rcParamsDefault)


def main():
    if args.dataset == "synthetic":
        ds = [load_synth_dataset(p, n_bins=5) for p in paths]
        adatas = [d["adata"] for d in ds]

        adata_big = ad.concat(
            adatas,
            axis=0,
            join="inner",
            label="rep",
            keys=[p.name for p in paths],
            index_unique=None,
        )

        true_mat = ds[0]["ref_network"].T

        counts_all = []
        labels_vec = []

        for t in range(5):
            sl = adata_big[adata_big.obs.t == t]
            X = (sl.X.A if hasattr(sl.X, "A") else sl.X).T
            counts_all.append(X)
            labels_vec.extend([t] * X.shape[1])

        counts = np.hstack(counts_all)
        labels = np.asarray(labels_vec)[None, :]
        Nt = 5

        dt = [1] * (Nt - 1)
        group_labels = [str(i) for i in range(Nt)]

        counts_pca, pca = visualize_pca(counts, labels, group_labels, viz_opt="pca")

        output_dir = f"otvelo_results/{backbone}_{args.subset}_seed{seed}"
        os.makedirs(output_dir, exist_ok=True)
        pca_folder = os.path.join(output_dir, "pca_plots")
        os.makedirs(pca_folder, exist_ok=True)

        folds = []
        predictions_dict = {}
        for held_out_t in range(1, Nt - 1):
            print(f"\n===== OTVelo – hold-out t={held_out_t} =====")

            w2, mmd, X_pred, per_dataset_metrics = otvelo_loto_one_fold(
                counts_all, adatas, held_out_t, eps=1e-2, alpha=0.5, pca=pca
            )
            folds.append({"t": held_out_t, "w2": w2, "mmd2": mmd})
            print(f"t={held_out_t}:  W₂={w2:.4f}   MMD²={mmd:.4e}")

            if held_out_t == Nt - 2:
                # Split X_pred back into individual dataset predictions
                # X_pred is (genes × total_cells_at_t), we need to split by dataset
                cell_counts = [
                    adata[adata.obs.t == held_out_t].shape[0] for adata in adatas
                ]

                start_idx = 0
                for i in range(len(adatas)):
                    end_idx = start_idx + cell_counts[i]
                    # X_pred is (genes × cells), so we slice along axis 1
                    dataset_pred = X_pred[:, start_idx:end_idx].T  # Now (cells × genes)
                    predictions_dict[i] = dataset_pred
                    start_idx = end_idx

        if predictions_dict and len(adatas) >= 3:
            ko_names = [p.name for p in paths]
            create_multi_ko_pca_plot_wgrey(
                adatas,
                predictions_dict,
                ko_names,
                Nt - 2,
                pca_folder,
                "otvelo",
                "Synthetic",
            )
            print(f"Multi-KO comparison plots saved to: {pca_folder}")

        df = pd.DataFrame(folds)
        print("\nMean W₂  :", df.w2.mean())
        print("Mean MMD²:", df.mmd2.mean())

        df.to_csv(
            os.path.join(output_dir, "trajectory_inference_results.csv"), index=False
        )

    elif args.dataset == "renge":
        ds = load_renge_dataset()
        ds = copy.deepcopy(ds)
        counts_all = ds["counts_all"]
        counts = ds["counts"]
        labels = ds["labels"]
        Nt = ds["Nt"]
        true_mat = ds["ref_network"]
        adata_tf = ds["adata"]

        cells_per_bin = [X.shape[1] for X in counts_all]
        old_to_new = {old: new for new, old in enumerate(sorted(np.unique(labels)))}

        labels = np.vectorize(old_to_new.get)(labels)
        Nt = len(old_to_new)
        assert labels.min() == 0 and labels.max() == Nt - 1

        output_dir = f"otvelo_results/Renge_seed{seed}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError("--dataset must be 'synthetic' or 'renge'")

    Ts_prior, _ = solve_prior(counts, counts, Nt, labels, eps_samp=1e-2, alpha=0.5)

    vel_all, vel_all_signed = solve_velocities(
        counts_all, Ts_prior, order=1, stimulation=False
    )

    vel = np.hstack(vel_all)
    vel_sign = np.hstack(vel_all_signed)

    Tv_corr, _ = OT_lagged_correlation(
        vel_all_signed,
        vel_sign,
        Ts_prior,
        stimulation=True,
        elastic_Net=False,
        tune=False,
        signed=True,
        return_slice=True,
    )

    Tv_corr -= np.diag(np.diag(Tv_corr))

    if args.dataset != "renge":
        true_mat -= np.diag(np.diag(true_mat))
    else:
        A_renge = pd.read_csv("../../data/Renge/A_renge_output.csv", index_col=0)
        genes_common = pd.Index(set(true_mat.columns).intersection(set(A_renge.index)))
        print(genes_common)
        tfs = true_mat.index
        y_true = (true_mat.loc[tfs, genes_common] > 0).values.flatten()

    Tv_Granger, Tv_Granger_slices = OT_lagged_correlation(
        vel_all_signed,
        vel_sign,
        Ts_prior,
        stimulation=True,
        elastic_Net=True,
        l1_opt=0.1,
        tune=False,
        signed=True,
        return_slice=True,
    )

    Tv_Granger = Tv_Granger - np.diag(np.diag(Tv_Granger))

    from sklearn.metrics import average_precision_score, roc_auc_score

    if args.dataset != "renge":
        mask = ~np.eye(Tv_corr.shape[0], dtype=bool)
        y_true = (true_mat[mask] != 0).astype(int)
        y_score = np.abs(Tv_corr[mask])
        y_score_granger = np.abs(Tv_Granger[mask])
    else:
        y_true = (true_mat.loc[tfs, genes_common] > 0).values.flatten()
        genes_panel = ds["adata"].var.index
        Tv_corr_df = pd.DataFrame(Tv_corr, index=genes_panel, columns=genes_panel)
        Tv_Granger_df = pd.DataFrame(Tv_Granger, index=genes_panel, columns=genes_panel)
        y_score = np.abs(Tv_corr_df.loc[tfs, genes_common]).values.flatten()
        y_score_granger = np.abs(Tv_Granger_df.loc[tfs, genes_common]).values.flatten()

    aupr = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    print(f"AUROC OT_corr (unsigned edges) : {auroc:.4f}")
    print(f"AUPR OT_corr (unsigned edges) : {aupr:.4f}")
    aupr_granger = average_precision_score(y_true, y_score_granger)
    auroc_granger = roc_auc_score(y_true, y_score_granger)
    print(f"AUROC OT_Granger (unsigned edges) : {auroc_granger:.4f}")
    print(f"AUPR OT_Granger (unsigned edges) : {aupr_granger:.4f}")

    results_df = pd.DataFrame(
        {
            "method": ["OT_corr", "OT_Granger"],
            "auroc": [auroc, auroc_granger],
            "aupr": [aupr, aupr_granger],
            "dataset": [args.dataset, args.dataset],
            "seed": [seed, seed],
        }
    )
    results_df.to_csv(
        os.path.join(output_dir, "grn_inference_results.csv"), index=False
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
