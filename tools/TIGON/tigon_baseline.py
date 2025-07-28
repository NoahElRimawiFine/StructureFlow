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

import warnings
warnings.filterwarnings("ignore")
_parser = argparse.ArgumentParser(description="TIGON synthetic runner")

# dataset layout
_parser.add_argument("--backbone", default="dyn-BF",
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


def main(): 
    ds = [load_synth_dataset(p, n_bins=5) for p in paths]
    adatas     = [d["adata"] for d in ds]  

    adata_big = ad.concat(adatas, axis=0, join="inner",
                      label="rep", keys=[p.name for p in paths],
                      index_unique=None)
    print(adata_big)
    coords_all = to_tigon_coords(adata_big)
    
    true_mat = ds[0]["ref_network"].T

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
    options.update({'neval_max': 1000000})
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







