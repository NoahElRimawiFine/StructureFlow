#!/usr/bin/env python
# infer_perturb.py
"""
Run dynGENIE3, GLASSO, SINCERITIES, SCODE on a perturb-seq dataset and write:

results/
└── <dataset>/                        # dataset folder  (dyn-TF-1000-1, HSC-…)
    ├── output_scode_<dataset>/      # full SCODE raw outputs
    ├── A_dyngenie3_*.csv            # one CSV per model …    } duplicated
    ├── A_glasso_*.csv               # … both here and in …   }   inside
    ├── A_sincerities_*.csv          #                       regime sub-dir
    ├── A_scode_*.csv                #                       (wt / full)
    ├── metrics_*.csv                # concatenated AP scores
    ├── wt/                          # WT-only run
    │   └── <all PNG/CSV from WT>    #  ─┐
    └── full/                        # full (concatenated)   ─┘ identical layout
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sklearn as sk
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve,
    roc_auc_score, 
    roc_curve,
    auc
)
import random

matplotlib.use("Agg")


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str,
                    help="Path to the WT replicate folder (dyn-TF-… or curated)")
parser.add_argument("--T", type=int, default=5,
                    help="Number of pseudo-time bins (default: 5)")
parser.add_argument("--curated", action="store_true",
                    help="Interpret the path as a curated dataset")
parser.add_argument("--concat_all", action="store_true",
                    help="Stack every AnnData before GRN inference")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility (default: None)")
args = parser.parse_args()


script_dir   = Path(__file__).resolve().parent
dataset_name = Path(args.path).name                    
regime_tag   = "full" if args.concat_all else "wt"     

dataset_root = script_dir / "results-curated" / dataset_name   
results_dir  = dataset_root / regime_tag               
results_dir.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)


def save_csv(df: pd.DataFrame, stem: str, **to_csv_kwargs) -> Path:
    """
    Write <stem>_<regime>_<dataset>.csv BOTH in
      • results_dir (wt/full)  and
      • dataset_root           (top-level)
    Returns the path inside results_dir.
    """
    fname    = f"{stem}_{regime_tag}_{dataset_name}.csv"
    run_path = results_dir   / fname
    root_path = dataset_root / fname

    df.to_csv(run_path,  **to_csv_kwargs)
    if run_path != root_path:           # duplicate upward
        shutil.copy2(run_path, root_path)

    return run_path


def discover_replicates(backbone_dir: str) -> list[Path]:
    bb = Path(backbone_dir).resolve()
    backbone = bb.name
    wt_csvs  = sorted(bb.glob(f"{backbone}-*/ExpressionData.csv"))
    if not wt_csvs:
        raise FileNotFoundError("No WT replicate found under", bb)
    wt_rep   = wt_csvs[0].parent

    ko_csvs  = sorted(bb.parent.glob(f"{backbone}_ko_*/*/ExpressionData.csv"))
    ko_reps  = [p.parent for p in ko_csvs]
    return [wt_rep, *ko_reps]


def load_adata(path: Path) -> ad.AnnData:
    adata = ad.AnnData(pd.read_csv(path / "ExpressionData.csv", index_col=0).T)
    df_pt = pd.read_csv(path / "PseudoTime.csv", index_col=0)
    df_pt[np.isnan(df_pt)] = 0
    adata.obs["t_sim"] = np.max(df_pt.to_numpy(), -1)
    sc.pp.log1p(adata); sc.tl.pca(adata); sc.pp.neighbors(adata)
    return adata


def bin_timepoints(adata: ad.AnnData, t_bins: np.ndarray) -> None:
    adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1


def maskdiag(A: np.ndarray) -> np.ndarray:
    return A * (1 - np.eye(A.shape[0]))


def plot_grns(grns: dict[str, np.ndarray],
              true_grn: np.ndarray | None = None,
              cmap="RdBu_r",
              clamp=20.0) -> plt.Figure:
    """
    Plot each GRN adjacency matrix with:
      • its own min/max (unless |val| > clamp, then we clamp)
      • 0 always white (via TwoSlopeNorm)
      • negative→blue, positive→red (RdBu_r)

    Args:
      grns:     dict of name→adjacency matrix
      true_grn: optional “True” matrix to append
      cmap:     diverging colormap
      clamp:    maximum absolute value allowed (per panel)
    """
    # collect names/mats
    names = list(grns)
    mats  = [maskdiag(grns[n]) for n in names]
    if true_grn is not None:
        names.append("True")
        mats.append(maskdiag(true_grn))

    # set up figure
    fig, axs = plt.subplots(1, len(mats), figsize=(4 * len(mats), 4), squeeze=False)

    for ax, name, mat in zip(axs[0], names, mats):
        # 1) panel raw min/max
        mn, mx = mat.min(), mat.max()
        # 2) clamp if too big, else use raw
        vmin = max(mn, -clamp)
        vmax = min(mx, +clamp)

        # 3) ensure vmin<0<vmax
        eps = max((vmax - vmin) * 1e-6, 1e-6)
        if vmin >= 0:
            vmin = -eps
        if vmax <= 0:
            vmax = eps

        # 4) normalize with center at zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        im = ax.imshow(mat, cmap=cmap, norm=norm)
        ax.set_title(name)
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig


backbone = (
    Path(args.path).name.split("-")[0]
    if args.curated else Path(args.path).name.split("-")[1]
)
print(f"[INFO] backbone = {backbone}")

paths   = discover_replicates(args.path)
adatas  = [load_adata(p) for p in paths]

# truth matrix ----------------------------------------------------
df_truth = pd.read_csv(Path(paths[0]).parent / "refNetwork.csv")
n_genes  = adatas[0].n_vars
true_matrix = pd.DataFrame(
    np.zeros((n_genes, n_genes), int),
    index=adatas[0].var.index, columns=adatas[0].var.index,
)
for _, row in df_truth.iterrows():
    tgt, src, sign = row[1], row[0], {"+": 1, "-": -1}[row[2]]
    true_matrix.loc[tgt, src] = sign

# pseudo-time bins ------------------------------------------------
t_bins = np.linspace(0, 1, args.T+1)[:-1]
for adata in adatas: bin_timepoints(adata, t_bins)

adata = (ad.concat(adatas, axis=0, join="inner", label="source",
                   keys=[p.name for p in paths], index_unique=None)
         if args.concat_all else adatas[0])


from dynGENIE3 import dynGENIE3

X = []
t = []
X.append(np.vstack([adata.X[adata.obs.t == t, :].mean(0) for t in np.sort(adata.obs.t.unique())]))
t.append(np.sort(adata.obs.t.unique()))
A_dyngenie3, _, _, _, _ =  dynGENIE3(X, t)
save_csv(pd.DataFrame(A_dyngenie3, index=adata.var.index, columns=adata.var.index),
         f"A_dyngenie3_T{args.T}", header=False, index=False)

import sklearn as sk
from sklearn import covariance, preprocessing
gl = sk.covariance.GraphicalLassoCV().fit(sk.preprocessing.StandardScaler().fit_transform(adata.X))
A_glasso = -gl.precision_
np.fill_diagonal(A_glasso, 0)
save_csv(pd.DataFrame(A_glasso, index=adata.var.index, columns=adata.var.index),
         "A_glasso", header=False, index=False)

# SINCERITIES -----------------------------------------------------
df_sinc = pd.DataFrame(adata.X, columns=adata.var.index, index=adata.obs.index)
df_sinc["t"] = adata.obs["t"]; df_sinc.sort_values("t").to_csv(
    "../../tools/SINCERITIES/X.csv", index=False)
os.system("Rscript ../../tools/SINCERITIES/MAIN.R")

edge_sign = {"repression": -1, "activation": 1, "no regulation": 0}
gene2idx  = {g: i for i, g in enumerate(adata.var.index)}
A_sinc    = np.zeros((adata.n_vars, adata.n_vars))
for row in pd.read_csv("../../tools/SINCERITIES/Results/GRNprediction.txt").itertuples():
    src, tgt = row.SourceGENES, row.TargetGENES
    if src not in gene2idx or tgt not in gene2idx: continue
    A_sinc[gene2idx[src], gene2idx[tgt]] = row.Interaction * \
        edge_sign.get(row.Edges.strip(), 0)

save_csv(pd.DataFrame(A_sinc, index=adata.var.index, columns=adata.var.index),
         "A_sincerities", header=False, index=False)

# copy SINCERITIES raw GRNprediction.txt into dataset_root --------
shutil.copy2("../../tools/SINCERITIES/Results/GRNprediction.txt",
             dataset_root / f"GRNprediction_{regime_tag}_{dataset_name}.txt")

# SCODE -----------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
expr_txt, time_txt = tmp / "sc_expr.txt", tmp / "sc_time.txt"
pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names).to_csv(
    expr_txt, sep="\t", index=False, header=False, float_format="%.6g")
pd.DataFrame({0: np.arange(adata.n_obs),
              1: adata.obs.t_sim.values}).to_csv(
    time_txt, sep="\t", index=False, header=False, float_format="%.6f")

# put SCODE outputs *inside* dataset_root/output_scode_<dataset>
scode_out = dataset_root / f"output_scode_{dataset_name}"
print(scode_out)
scode_out.mkdir(parents=True, exist_ok=True)

subprocess.run([
    "ruby", "run_R.rb",
    str(expr_txt), str(time_txt), str(scode_out),
    str(adata.n_vars), "8", str(adata.n_obs), "100", "10"
], check=True)

A_scode = pd.read_csv(scode_out / "meanA.csv")
save_csv(A_scode, "A_scode", header=False, index=False)

# ░░░  Diagnostics & metrics
# ----------------------------------------------------------------
methods = {
    "dynGENIE3":  A_dyngenie3,
    "GLASSO":     A_glasso,
    "SINCERITIES": A_sinc,
    "SCODE":      A_scode.values,
}

fig = plot_grns(methods, true_matrix.values)
fig.savefig(results_dir / f"comparison_T{args.T}_{regime_tag}_{dataset_name}_seed{args.seed}.png",
            dpi=300); plt.close(fig)

y_true = np.abs(maskdiag(true_matrix.values)).astype(int).flatten()

 # ---------------- PR + ROC curves ----------------
curves_pr  = {}
curves_roc = {}
for name, A in methods.items():
    y_score = np.abs(maskdiag(A)).flatten()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    aupr = auc(rec, prec)
    curves_pr[name] = (rec, prec, ap, aupr)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    curves_roc[name] = (fpr, tpr, roc_auc_score(y_true, y_score))

# ---- PR figure (unchanged) ----
fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
for name, (rec, prec, ap, aupr) in curves_pr.items():
    ax_pr.step(rec, prec, where="post",
               label=f"{name} (AP={ap:.2f}, AUPR={aupr:.2f})")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title(f"PR curves – {regime_tag.upper()} data"); ax_pr.legend()
ax_pr.grid(True)
fig_pr.tight_layout()
fig_pr.savefig(results_dir / f"pr_{regime_tag}_{dataset_name}_seed{args.seed}.pdf",
               dpi=300)
plt.close(fig_pr)

# ---- ROC figure ----
fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
for name, (fpr, tpr, auc) in curves_roc.items():
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
ax_roc.set_xlabel("False‑positive rate");  ax_roc.set_ylabel("True‑positive rate")
ax_roc.set_title(f"ROC curves – {regime_tag.upper()} data");  ax_roc.legend()
fig_roc.savefig(results_dir / f"roc_{regime_tag}_{dataset_name}_seed{args.seed}.pdf",
                dpi=300, bbox_inches="tight")
plt.close(fig_roc)

# ---------------- CSV ----------------
metric_rows = []
for m in methods:
    metric_rows.append({
        "dataset": dataset_name,
        "regime":  regime_tag,
        "seed":    args.seed,
        "method":  m,
        "AP":      curves_pr[m][2],
        "AUPR":    curves_pr[m][3],
        "AUC_ROC": curves_roc[m][2],
    })

pd.DataFrame(metric_rows).to_csv(
    results_dir / f"metrics_seed{args.seed}.csv", index=False)

print(f"[DONE] All outputs under  {dataset_root.relative_to(script_dir)}/")
