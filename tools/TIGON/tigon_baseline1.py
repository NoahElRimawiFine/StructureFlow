from __future__ import annotations
import numpy as np
import pandas as pd
import sys
import sklearn
import copy
from pathlib import Path
import anndata as ad, scanpy as sc
from typing import Dict, Any
import scipy.sparse as sp
import argparse
import torch
import random
import torch.optim as optim
from tigon.utility import *
import scipy
import matplotlib.pyplot as plt
import math
import ot
from TorchDiffEqPack import odesolve

import warnings
warnings.filterwarnings("ignore")
_parser = argparse.ArgumentParser(description="TIGON synthetic runner")

# dataset layout
_parser.add_argument("--backbone", default="dyn-TF",
                     help="dyn‑BF, dyn‑TF, … (folder name inside data/Synthetic)")
_parser.add_argument("--subset", choices=["wt", "ko", "all"], default="wt",
                     help="replicate subset to use")
# training hyper‑params
_parser.add_argument("--timepoints", default="0.0,0.25,0.5,0.75,1.0",
                     help="comma‑sep list of numeric times; length = #bins")
_parser.add_argument("--niters", type=int, default=100)
_parser.add_argument("--lr", type=float, default=3e-3)
_parser.add_argument("--hidden-dim", type=int, default=16)
_parser.add_argument("--n-hiddens", type=int, default=4)
_parser.add_argument("--activation", choices=["Tanh", "relu", "elu", "leakyrelu"],
                     default="Tanh")
_parser.add_argument("--num-samples", type=int, default=100,
                     help="#points to sample per epoch")
# I/O
_parser.add_argument("--input-dir", default="Input/")
_parser.add_argument("--save-dir", default="Output/")
_parser.add_argument("--gpu", type=int, default=0)
_parser.add_argument("--seed", type=int, default=1)
# misc
_parser.add_argument("--n-bins", type=int, default=5,
                     help="how many pseudotime bins to make")

args = _parser.parse_args()
args.timepoints = [float(x) for x in args.timepoints.split(",")]

run_id = f"{args.backbone}_{args.subset}_{args.seed}"
args.save_dir = Path(args.save_dir) / run_id
args.save_dir.mkdir(parents=True, exist_ok=True)

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
    sc.pp.log1p(adata); sc.tl.pca(adata); sc.pp.neighbors(adata); sc.tl.umap(adata, n_components=3, random_state=seed)
    return adata

def bin_timepoints(adata: ad.AnnData, t_bins: np.ndarray) -> None:
    adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1

def to_tigon_coords(adata: ad.AnnData):
    coords_all = []
    for tb in sorted(adata.obs["t"].unique()):
        sl = adata[adata.obs.t == tb]
        X = sl.X.A if sp.issparse(sl.X) else sl.X   
        coords_all.append(X)
    return coords_all


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

    coords_all = to_tigon_coords(adata)

    return dict(
        adata=adata,
        coords_all=coords_all,
        ref_network=true_mat
    )

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
    if X.dim() > 2: X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2: Y = Y.reshape(Y.shape[0], -1)
    d = X.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    dist_sq = torch.cdist(X, Y)**2
    K = torch.exp(-gamma * dist_sq)
    return K, gamma

def mmd_squared(X, Y, kernel=rbf_kernel, sigma_list=None, **kernel_args):
    if X.dim() > 2: X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2: Y = Y.reshape(Y.shape[0], -1)
    
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


def validate_one_bin(func,
                     data_all,            # full list (len=Nt) of tensors
                     t_star,              # int index of held-out bin
                     train_bins,          # list[int] actually used to fit
                     args,
                     options,
                     device):
    """
    • Generates model samples at t_star by integrating from the nearest
      *observed* bin on the *training* side.
    • Returns Wasserstein-2 and MMD scores vs. the real cells in bin t_star.
    """
    # ----- true cells -------------------------------------------------- #
    z_true = data_all[t_star]                               # (n_val, d)
    print(f"[INFO] validating bin {t_star} with {z_true.shape[0]} cells")

    # ----- pick a 'source' bin to shoot from (nearest in time) --------- #
    src = min(train_bins, key=lambda t: abs(t - t_star))
    z_src = data_all[src]
    print(f"[INFO] using source bin {src} with {z_src.shape[0]} cells")

    # match sample sizes for fair comparison
    n_val = z_true.shape[0]
    idx   = torch.randperm(z_src.shape[0])[:n_val]
    z0    = z_src[idx].clone().to(device)

    # dummy g, logp  (TIGON's signature)
    g0  = torch.zeros(n_val, 1, device=device)
    lp0 = torch.zeros_like(g0)

    # integrate flow  src_time  →  t_star
    opts = options.copy()
    opts.update({"t0": args.timepoints[src],
                 "t1": args.timepoints[t_star],
                 "t_eval": [args.timepoints[t_star]]})

    z_pred, _, _ = odesolve(func, y0=(z0, g0, lp0), options=opts)
    print(z_pred.shape, z_true.shape)

    # ----- metrics ----------------------------------------------------- #
    wd   = wasserstein(z_pred.detach(), z_true.detach())
    mmd2 = mmd_squared(z_pred.detach(), z_true.detach())

    return {"t_star": t_star,
            "src": src,
            "WD2": wd,
            "MMD": mmd2}
    


def main(): 
    ds = [load_synth_dataset(p, n_bins=5) for p in paths]
    adatas     = [d["adata"] for d in ds]  

    adata_big = ad.concat(adatas, axis=0, join="inner",
                      label="rep", keys=[p.name for p in paths],
                      index_unique=None)
    print(adata_big)
    coords_all = to_tigon_coords(adata_big)
    
    true_mat = ds[0]["ref_network"].T

    all_metrics = []
    Nt = len(coords_all)                   # number of time bins
    print(f"[INFO] {Nt} time bins found in the dataset")
    device = 'cpu'
    mse = nn.MSELoss()
    data_all   = [torch.tensor(coords_all[t],      
                           dtype=torch.float32,
                           device=device)
              for t in range(Nt)] 

    # for t_star in range(1,4):
    #      # configure training options
    #     options = {}
    #     options.update({'method': 'Dopri5'})
    #     options.update({'h': None})
    #     options.update({'rtol': 1e-3})
    #     options.update({'atol': 1e-5})
    #     options.update({'print_neval': False})
    #     options.update({'neval_max': 1000000})
    #     options.update({'safety': None})

    #     print(f"\n===  Leaving out time bin {t_star}/{Nt-1}  ===")

    #     # ---------------------------
    #     # Data split
    #     # ---------------------------
    #     leave_1_out = [t_star]
    #     train_time  = [t for t in range(Nt) if t not in leave_1_out]
    #     print(f"Training on time bins: {train_time}")

    #     data_train  = [torch.tensor(coords_all[t], dtype=torch.float32,
    #                                 device=device) for t in train_time]
    #     z_val       = torch.tensor(coords_all[t_star],
    #                             dtype=torch.float32, device=device)
    #     integral_time = [args.timepoints[t] for t in train_time]

    #     func = UOT(in_out_dim=data_train[0].shape[1],
    #             hidden_dim=args.hidden_dim,
    #             n_hiddens=args.n_hiddens,
    #             activation=args.activation).to(device)
    #     func.apply(initialize_weights)
    #     optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay=0.01)
    #     lr_adjust = optim.lr_scheduler.MultiStepLR(
    #         optimizer, milestones=[args.niters-400, args.niters-200],
    #         gamma=0.5, last_epoch=-1)

    #     sigma_now = 1.0
    #     for itr in range(1, args.niters + 1):
    #         optimizer.zero_grad()
    #         loss, *_ = train_model(mse, func, args, data_all,
    #                             train_time, integral_time,
    #                             sigma_now, options, device, itr)
    #         loss.backward()
    #         optimizer.step()
    #         lr_adjust.step()

    #     metric_dict = validate_one_bin(func,
    #                            data_train,
    #                            t_star,
    #                            train_time,
    #                            args,
    #                            options,
    #                            device)
    #     all_metrics.append(metric_dict)
    #     print(f"bin {t_star}: WD2={metric_dict['WD2']:.4f} | MMD={metric_dict['MMD']:.4f}")
    

    data_train = [torch.tensor(c, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu') for c in coords_all]
    
    for i, arr in enumerate(coords_all):
        print(f"bin {i}: {arr.shape[0]} cells   dim={arr.shape[1]}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] using device {device}")

    integral_time = args.timepoints

    time_pts = range(len(coords_all))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i != leave_1_out]

    #model 
    func = UOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim, n_hiddens=args.n_hiddens,
               activation=args.activation).to(device)
    func.apply(initialize_weights)

     # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 100000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay= 0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters-400,args.niters-200], gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []
    
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = Path(args.save_dir) / "ckpt.pth"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        sigma_now = 1
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            
            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)

            
            loss.backward()
            optimizer.step()
            lr_adjust.step()

            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())
            
            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
            
            
            if itr % 500 == 0:
                ckpt_path = Path(args.save_dir) / "ckpt_itr{}.pth".format(itr)
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))
                
            
            

    except KeyboardInterrupt:
        if args.save_dir is not None:
            ckpt_path = Path(args.save_dir) / "ckpt.pth"
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))
    
    
    ckpt_path = Path(args.save_dir) / "ckpt.pth"
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS':LOSS,
        'TRANS':Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = Path(args.save_dir) / "ckpt.pth"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
            func.load_state_dict(checkpoint['func_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))
    
    time_pt = 0
    z_t = data_train[time_pt]
    gene_list = adata_big.var.index.to_list()
    jac = plot_jac_v(func,z_t,time_pt,'Average_jac_d0.pdf', gene_list,args,device)

    def save_adj_heat(mat, title, out_name, cmap="RdBu_r"):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(abs(mat), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=20); ax.invert_yaxis()
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(f"{out_name}.pdf", dpi=300)
        plt.close(fig)

    save_adj_heat(jac, "TIGON", f"tigon_{args.backbone}", cmap="Reds")

    from sklearn.metrics import average_precision_score, roc_auc_score

    true_mat -= np.diag(np.diag(true_mat))

    y_true  = (true_mat != 0).astype(int)
    y_pred  = (jac != 0).astype(int)

    print(y_true, '\n', y_pred)

    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(f"AP: {ap:.4f}, AUC: {auc:.4f}")



if __name__ == "__main__":
    main()







