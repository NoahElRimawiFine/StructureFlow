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

from otvelo.utils_Velo import *

from otvelo.utils import *
import scipy
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="OTVelo baseline script")

parser.add_argument("--subset",
                    choices=["wt", "ko", "all"], default="wt",
                    help="which replicates to load: "
                         "'wt' (wild‑type), 'ko' (knock‑outs), or 'all'")
parser.add_argument("--backbone", type=str, default="dyn-BF",
                    help="backbone name, e.g. 'dyn-BF', 'dyn-TF'")

parser.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducibility")

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

    ko_dirs = [d for d in root.iterdir()
               if d.is_dir() and d.name.startswith(f"{backbone}_ko")]

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
    sc.pp.log1p(adata); sc.tl.pca(adata); sc.pp.neighbors(adata)
    return adata

def bin_timepoints(adata: ad.AnnData, t_bins: np.ndarray) -> None:
    adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1

def to_otvelo_arrays(adata: ad.AnnData):
    counts_all = []
    labels_vec  = []
    for tb in sorted(adata.obs["t"].unique()):
        sl = adata[adata.obs.t == tb]
        X  = sl.X.A if hasattr(sl.X, "A") else sl.X         
        X  = X.T                                           
        counts_all.append(X)
        labels_vec.extend([tb] * X.shape[1])

    counts   = np.hstack(counts_all)                
    labels   = np.asarray(labels_vec)[None, :]        
    Nt       = len(counts_all)
    return counts_all, counts, labels, Nt


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
    
    t_bins = np.linspace(0, 1, n_bins+1)[:-1]
    
    adata = load_adata(ds_dir)

    # loading the reference network
    true_mat = None
    ref_path = ds_dir / "refNetwork.csv"
    if ref_path.exists():
        df_ref = pd.read_csv(ref_path)              
        genes  = adata.var.index.to_list()         
        g2i    = {g: i for i, g in enumerate(genes)}

        true_mat = np.zeros((len(genes), len(genes)), dtype=np.int8)

        for tgt, src, sign in df_ref.itertuples(index=False):
            if src not in g2i or tgt not in g2i:
                continue           
            i, j = g2i[tgt], g2i[src]          
            true_mat[i, j] = 1 if sign == '+' else -1

    bin_timepoints(adata, t_bins=t_bins)

    counts_all, counts, labels, Nt = to_otvelo_arrays(adata)

    return dict(
        adata=adata,
        counts_all=counts_all,
        counts=counts,
        labels=labels,
        Nt=Nt,
        ref_network=true_mat,
    )

def main(): 
    ds = [load_synth_dataset(p, n_bins=5) for p in paths]
    adatas     = [d["adata"] for d in ds]  

    adata_big = ad.concat(adatas, axis=0, join="inner",
                      label="rep", keys=[p.name for p in paths],
                      index_unique=None)
    
    true_mat = ds[0]["ref_network"].T

    counts_all = []     
    labels_vec = []      

    for t in range(5):
        sl  = adata_big[adata_big.obs.t == t]
        X   = (sl.X.A if hasattr(sl.X, "A") else sl.X).T   
        counts_all.append(X)
        labels_vec.extend([t] * X.shape[1])

    counts = np.hstack(counts_all)           
    labels = np.asarray(labels_vec)[None, :]  
    Nt     = 5                                

    print("counts shape :", counts.shape)     

    Ts_prior, _ = solve_prior(counts, counts,
                          Nt, labels,
                          eps_samp=1e-2, alpha=0.5)

    vel_all, vel_all_signed = solve_velocities(counts_all, Ts_prior,
                                            order=1, stimulation=False)

    vel      = np.hstack(vel_all)
    vel_sign = np.hstack(vel_all_signed)

    Tv_corr, _ = OT_lagged_correlation(
                    vel_all_signed, vel_sign, Ts_prior, stimulation=True, 
                                 elastic_Net=False,tune=False,signed=True, return_slice=True)

    Tv_corr -= np.diag(np.diag(Tv_corr))
    true_mat -= np.diag(np.diag(true_mat))

    Tv_Granger, Tv_Granger_slices = OT_lagged_correlation(vel_all_signed, vel_sign, Ts_prior, stimulation=True, 
                                 elastic_Net=True,l1_opt=0.5,tune=False,signed=True, return_slice=True )       

    Tv_Granger = Tv_Granger - np.diag( np.diag(Tv_Granger) )
    plt.imshow(Tv_Granger, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.gca().invert_yaxis()
    plt.title('OTVelo‑Granger')
    plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # im0 = ax[0].imshow(Tv_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    # ax[0].invert_yaxis()
    # ax[0].set_title('OTVelo‑Corr')

    # im1 = ax[1].imshow(true_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    # ax[1].invert_yaxis()
    # ax[1].set_title('Reference network')

    # plt.tight_layout()
    # plt.show()


    from sklearn.metrics import average_precision_score, roc_auc_score

    mask = ~np.eye(Tv_corr.shape[0], dtype=bool)

    y_true  = (true_mat[mask] != 0).astype(int)    
    y_score = np.abs(Tv_corr[mask])
    y_score_granger = np.abs(Tv_Granger[mask])


    aupr = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    print(f"AUROC OT_corr (unsigned edges) : {auroc:.4f}")
    print(f"AUPR OT_corr (unsigned edges) : {aupr:.4f}")
    aupr_granger = average_precision_score(y_true, y_score_granger)
    auroc_granger = roc_auc_score(y_true, y_score_granger)
    print(f"AUROC OT_Granger (unsigned edges) : {auroc_granger:.4f}")
    print(f"AUPR OT_Granger (unsigned edges) : {aupr_granger:.4f}")


if __name__ == "__main__":
    main()







