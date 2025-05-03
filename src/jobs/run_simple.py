import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy
import sys, os, itertools
import seaborn as sns

import glob
import argparse
import sys
from pathlib import Path
import tempfile
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--T", type=int, default = 5)
parser.add_argument("--curated", action = "store_true")
parser.add_argument(
    "--concat_all",
    action="store_true",
    help="If set, stack every AnnData in `adatas` (rows = cells) before GRN inference",
)

# sys.argv = ['infer_perturb.py', '/scratch/users/zys/Synthetic/dyn-TF/dyn-TF-1000-1',]
# sys.argv = ['infer_perturb.py', '/scratch/users/zys/Curated/HSC/HSC-1000-1', '--curated']
args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_tag = Path(args.path).name 
results_dir = Path(script_dir) / "results" / dataset_tag
results_dir.mkdir(parents=True, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def save_csv(df: pd.DataFrame, stem: str, **to_csv_kwargs):
    """
    Write <stem>_<dataset_tag>.csv inside results_dir.
    Extra kwargs are forwarded to DataFrame.to_csv().
    """
    out_path = results_dir / f"{stem}_{dataset_tag}.csv"
    df.to_csv(out_path, **to_csv_kwargs)
    return out_path


def discover_replicates(backbone_dir: str) -> list[Path]:
    """Return [WT_rep_path, *KO_rep_paths] for a dyn-TF backbone directory."""
    bb = Path(backbone_dir).resolve()
    backbone = bb.name                       # 'dyn-TF'
    # WT replicate(s) inside dyn-TF/
    wt_reps = sorted(bb.glob(f"{backbone}-*/ExpressionData.csv"))
    if not wt_reps:
        raise FileNotFoundError("No WT replicate found under", bb)
    wt_rep = wt_reps[0].parent               # drop the csv

    # KO replicates: dyn-TF_ko_*/dyn-TF_ko_*-*/ExpressionData.csv
    ko_reps = sorted(
        bb.parent.glob(f"{backbone}_ko_*/*/ExpressionData.csv")
    )
    ko_reps = [p.parent for p in ko_reps]    # strip csv

    return [wt_rep, *ko_reps]

# Load data
def load_adata(path):
    adata = ad.AnnData(pd.read_csv(os.path.join(path, "ExpressionData.csv"), index_col = 0).T)
    df_pt = pd.read_csv(os.path.join(path, "PseudoTime.csv"), index_col = 0)
    df_pt[np.isnan(df_pt)] = 0
    adata.obs["t_sim"] = np.max(df_pt.to_numpy(), -1)
    sc.pp.log1p(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    return adata

def bin_timepoints(adata):
    adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins)-1
ko_genes = []

# Get backbone
backbone = os.path.basename(args.path).split('-')[0] if args.curated else os.path.basename(args.path).split('-')[1]
print(f"backbone = {backbone}")

paths = discover_replicates(args.path)   # args.path now = backbone dir
names = [p.parent.name if p.name == "ExpressionData.csv" else p.name for p in paths]
adatas = [load_adata(p) for p in paths]

df = pd.read_csv(os.path.join(os.path.dirname(paths[0]), "refNetwork.csv"))
n_genes = adatas[0].n_vars
dim = n_genes
true_matrix = pd.DataFrame(
    np.zeros((n_genes, n_genes), int),
    index=adatas[0].var.index,
    columns=adatas[0].var.index,
)
for i in range(df.shape[0]):
    _i = df.iloc[i, 1]  # target gene
    _j = df.iloc[i, 0]  # source gene
    _v = {"+": 1, "-": -1}[df.iloc[i, 2]]
    true_matrix.loc[_i, _j] = _v

# Bin timepoints
t_bins = np.linspace(0, 1, args.T+1)[:-1]
for adata in adatas:
    bin_timepoints(adata)
# Get KOs
kos = []
for p in paths:
    try:
        kos.append(os.path.basename(p).split('_ko_')[1].split("-")[0])
    except:
        kos.append(None)


if args.concat_all:
    adata = ad.concat(adatas, axis=0, join="inner", label="source",
                      keys=names, index_unique=None)
    print(f"[INFO] Concatenated {len(adatas)} replicates → "
          f"{adata.n_obs} cells × {adata.n_vars} genes")
else:
    adata = adatas[0]          # original WT-only behaviour
    print(f"[INFO] Using WT only → {adata.n_obs} cells × {adata.n_vars} genes")



# dynGENIE3
X = []
t = []
X.append(np.vstack([adata.X[adata.obs.t == t, :].mean(0) for t in np.sort(adata.obs.t.unique())]))
t.append(np.sort(adata.obs.t.unique()))
from dynGENIE3 import dynGENIE3
A_dyngenie3, _, _, _, _ =  dynGENIE3(X, t)
pd.DataFrame(A_dyngenie3, index = adata.var.index, columns = adata.var.index).to_csv(os.path.join(results_dir, f"A_dyngenie3_T_{args.T}_{dataset_tag}.csv"),
                                                                                     header=False, index=False)

# GLASSO
import sklearn as sk
from sklearn import covariance, preprocessing
gl = sk.covariance.GraphicalLassoCV().fit(sk.preprocessing.StandardScaler().fit_transform(adata.X))
A_glasso = -gl.precision_
np.fill_diagonal(A_glasso, 0)
pd.DataFrame(A_glasso, index = adata.var.index, columns = adata.var.index).to_csv(os.path.join(results_dir, f"A_glasso_{dataset_tag}.csv"), 
                                                                                  header=False, index=False)

# SINCERITIES
df = pd.DataFrame(adata.X, columns = adata.var.index, index = adata.obs.index)
df["t"] = adata.obs["t"]
df =df.sort_values(by = "t")
df.to_csv("../../tools/SINCERITIES/X.csv", index = False)
cmd = 'Rscript ../../tools/SINCERITIES/MAIN.R'
print(f"Ran SINCERITIES, return code = {os.system(cmd)}")

edge_sign_map = {
    "repression": -1,
    "activation": 1,
    "no regulation": 0
}

gene_to_idx = {g: idx for idx, g in enumerate(adata.var.index)}

A_sincerities = np.zeros((adata.shape[1], adata.shape[1]))

for row in pd.read_csv("../../tools/SINCERITIES/Results/GRNprediction.txt").itertuples():
    src, tgt = row.SourceGENES, row.TargetGENES

    # Skip any genes not found in the AnnData
    if src not in gene_to_idx or tgt not in gene_to_idx:
        print(f"Warning: {src}->{tgt} not in adata.var.index; skipping")
        continue

    i = gene_to_idx[src]
    j = gene_to_idx[tgt]

    v    = row.Interaction
    sign = edge_sign_map.get(row.Edges.strip(), 0)
    A_sincerities[i, j] = v * sign


# SCODE
tmp = Path(tempfile.mkdtemp())
expr_txt = tmp / "sc_expr.txt"
time_txt = tmp / "sc_time.txt"

expr_df = pd.DataFrame(
    adata.X.T,
    index=adata.var_names,
    columns=adata.obs_names
)

expr_df.to_csv(expr_txt, sep="\t", index=False, header=False, float_format="%.6g")

pt = adata.obs["t_sim"].values
pt_df = pd.DataFrame({0: np.arange(len(pt)), 1: pt})
pt_df.to_csv(time_txt, sep="\t", index=False, header=False, float_format="%.6f")


cmd = [
    "ruby",
    "run_R.rb",
    str(expr_txt),
    str(time_txt),
    f"output_scode_{Path(args.path).name}",
    str(adata.n_vars),
    str(8),
    str(adata.n_obs),
    str(100),
    str(10)
]
subprocess.run(cmd, check=True)

A_scode = pd.read_csv(f"./output_scode_{Path(args.path).name}/meanA.csv")


def maskdiag(A):
    """Zero out the diagonal entries (or remove them) from matrix A."""
    return A * (1 - np.eye(A.shape[0]))

def plot_grns(grns: dict[str, np.ndarray],
              true_grn: np.ndarray | None = None,
              cmap="RdBu_r"):
    """
    grns: dict mapping method name → estimated adjacency matrix
    true_grn: optional ground truth adjacency (will be plotted last)
    """
    names = list(grns.keys())
    mats  = [maskdiag(grns[name]) for name in names]
    print(mats)
    

    if true_grn is not None:
        names.append("True")
        mats.append(maskdiag(true_grn))

    n = len(mats)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)

    for ax, name, mat in zip(axs[0], names, mats):
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title(name)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

        # Individual colorbar for each
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        

    return fig

# --- Example usage after you compute all your As ---
methods = {
    "dynGENIE3": A_dyngenie3,
    "GLASSO":      A_glasso,
    "SINCERITIES": A_sincerities,
    "SCODE":        A_scode.values,
}

regime_tag = "full" if args.concat_all else "wt"

fig = plot_grns(methods, true_matrix.values)
dataset_name = Path(args.path).name
outfile = f"comparison_{regime_tag}_T_{args.T}_{dataset_name}.png"
fig.savefig(os.path.join(results_dir, outfile), dpi=300)

# ────────────────────────────────────────────────────────────────
#  AUPR curves  +  Average-Precision (AP) scores
# ────────────────────────────────────────────────────────────────
y_true = np.abs(np.sign(maskdiag(true_matrix.values)).astype(int).flatten())
curves   = {}          # method -> (recall, precision, AP)

for name, A in methods.items():
    y_score = np.abs(maskdiag(A)).flatten()
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    curves[name] = (rec, prec, ap)
    print(f"[AUPR] {regime_tag.upper():4} | {name:12}  AP = {ap:.4f}")


fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
for name, (rec, prec, ap) in curves.items():
    ax_pr.step(rec, prec, where="post", label=f"{name} (AP={ap:.2f})")
ax_pr.set_xlabel("Recall");  ax_pr.set_ylabel("Precision")
ax_pr.set_title(f"PR curves – {regime_tag.upper()} data")
ax_pr.legend()
fig_pr.savefig(os.path.join(results_dir, f"pr_{regime_tag}.png"),
               dpi=300, bbox_inches="tight")
plt.close(fig_pr)


ap_csv = Path(results_dir) / "metrics.csv"
row_df = pd.DataFrame(
    [{"regime": regime_tag, "method": m, "AP": curves[m][2]} for m in curves]
)
if ap_csv.exists():
    row_df = pd.concat([pd.read_csv(ap_csv), row_df], ignore_index=True)
row_df.to_csv(ap_csv, index=False)

ap_data = pd.read_csv(ap_csv)
if set(ap_data.regime) == {"wt", "full"}:
    methods_sorted = ap_data.method.unique()
    xs = np.arange(len(methods_sorted));  bar_w = 0.35
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    for i, reg in enumerate(["wt", "full"]):
        vals = ap_data[ap_data.regime == reg].set_index("method").loc[methods_sorted]
        ax_bar.bar(xs + i*bar_w, vals.AP, width=bar_w, label=reg.upper())
    ax_bar.set_xticks(xs + bar_w/2)
    ax_bar.set_xticklabels(methods_sorted, rotation=45, ha="right")
    ax_bar.set_ylabel("Average Precision (AUPR)")
    ax_bar.set_title("WT vs FULL – Average Precision")
    ax_bar.legend()
    fig_bar.tight_layout()
    fig_bar.savefig(os.path.join(results_dir, "ap_barplot.png"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig_bar)


