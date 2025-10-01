#!/usr/bin/env python3
# heatmap_panels.py -----------------------------------------------------------
"""
Create side‑by‑side heat‑maps for WT and FULL conditions.

Folder layout expected
----------------------
results/
└── dyn-SW/
    ├── A_true_dyn-SW.csv
    ├── A_sincerities_full_dyn-SW.csv
    ├── A_sincerities_wt_dyn-SW.csv
    ├── A_scode_full_dyn-SW.csv
    └── ... etc.

Each file name encodes:
  A_<method>_[T<bins>]_<(wt|full)>?_dyn-<DATASET>.csv
"""

from __future__ import annotations
import os, re, glob
from pathlib import Path
from typing  import Dict, List, Tuple

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib        as mpl
from matplotlib.ticker import FormatStrFormatter

# ──────────────────────────────────────────────────────────────────────────────
# 1. small helpers
# ──────────────────────────────────────────────────────────────────────────────
def maskdiag(A: np.ndarray) -> np.ndarray:
    """Zero the diagonal – makes weights clearer in the heat‑map."""
    out = A.copy()
    for i in range(min(out.shape)):
        out[i, i] = 0
    return out


MODEL_MAP = {
    "glasso":      "Glasso",
    "w_v":         "StructureFlow",
    "scode":       "SCODE",
    "rf":          "RF",
    "sincerities": "Sincerities",
    "dyngenie3":   "dynGENIE3",
}

def extract_model_and_dataset(fname: str) -> Tuple[str, str, str, str | None]:
    """
    Decode *one* CSV filename.

    Returns
    -------
    title   : str   pretty panel title           "SCODE (SW)"
    out_stub: str   filename stub w/o extension  "SCODE_FULL_SW"
    dataset : str   dataset code                 "SW"
    cond    : str|None  "WT" | "FULL" | None
    """
    base  = Path(fname).stem
    parts = base.split("_")

    # ground‑truth: A_true_dyn-XYZ.csv
    if parts[:2] == ["A", "true"]:
        ds = Path(fname).parent.name
        ds = ds.split("-", 1)[1].upper() if ds.startswith("dyn-") else ds.upper()
        title   = f"True Graph ({ds})"
        out     = f"A_true_{ds}"
        return title, out, ds, None

    # dataset code = last token (strip leading dyn-)
    last    = parts[-1].lower()
    dataset = last.split("-", 1)[1].upper() if last.startswith("dyn-") else last.upper()

    # optional WT / FULL flag
    cond = None
    if len(parts) >= 3 and parts[-2].lower() in ("wt", "full"):
        cond           = parts[-2].upper()
        method_tokens  = parts[1:-2]
    else:
        method_tokens  = parts[1:-1]

    # drop T<number> tokens
    method_tokens = [tok for tok in method_tokens if not re.fullmatch(r"[tT]\d+", tok)]
    method_key    = "_".join(tok.lower() for tok in method_tokens)
    method_label  = MODEL_MAP.get(method_key, method_key.replace("_", " ").title())

    title   = f"{method_label} ({dataset})"
    out     = f"{method_label}_{cond}_{dataset}" if cond else f"{method_label}_{dataset}"
    return title, out, dataset, cond


# canonical panel order – anything not listed is pushed to the end
PANEL_ORDER = [
    "True Graph", "Sincerities", "SCODE",
    "dynGENIE3", "Glasso", "RF", "StructureFlow",
]
def panel_key(title: str) -> int:           # order helper
    label = title.split(" (")[0]
    try:
        return PANEL_ORDER.index(label)
    except ValueError:
        return len(PANEL_ORDER) + 1

def generate_heatmaps(
    source: str,
    output_base_dir: str = "heatmaps",
    cmap: str = "Reds",
    panel_size: float = 6.0,          # inches per heat‑map
    # vmax_cap: float  = 4.0,           # clip colour‑bar if matrices vary a lot
    normalise: str | None = "minimax"
) -> None:

    out_root = Path(output_base_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 2 a. gather every csv
    if Path(source).is_dir():
        csv_paths = glob.glob(os.path.join(source, "A*.csv"))
    elif source.lower().endswith(".csv"):
        csv_paths = [source]
    else:
        raise ValueError(f"{source!r} is neither a directory nor a CSV file")

    groups: Dict[Tuple[str, str], List[str]] = {}
    for p in csv_paths:
        title, _, ds, cond = extract_model_and_dataset(p)
        cond_key = cond or "TRUE"         
        groups.setdefault((ds, cond_key), []).append(p)

    for (ds, cond), lst in list(groups.items()):
        if cond == "TRUE":
            for target_cond in ("WT", "FULL"):
                groups.setdefault((ds, target_cond), []).extend(lst)

    for (dataset, cond), paths in groups.items():
        if cond == "TRUE":                 
            continue

        # ---- read matrices --------------------------------------------------
        mats, titles, vmax_list = [], [], []
        for csv_file in paths:
            df   = pd.read_csv(csv_file, header=None)
            data = np.abs(maskdiag(df.values))
            if normalise == "minmax":
                denom = np.nanmax(data)
                data   = data / denom if denom and denom > 0 else data
            elif normalise == "zscore":
                mu, sigma = np.nanmean(data), np.nanstd(data)
                sigma     = sigma if sigma > 0 else 1
                data       = (data - mu) / sigma
            
            mats.append(data)
            vmax_list.append(np.nanmax(data))

            title, *_ = extract_model_and_dataset(csv_file)[:1]
            titles.append(title)

        # sort into canonical order
        order      = np.argsort([panel_key(t) for t in titles])
        mats       = [mats[i]   for i in order]
        titles     = [titles[i] for i in order]
        vmax       = min(1, max(vmax_list))

        # ---- figure & axes --------------------------------------------------
        n_panels   = len(mats)
        fig_w      = panel_size * n_panels + 1.2
        fig        = plt.figure(figsize=(fig_w, panel_size))
        gs         = mpl.gridspec.GridSpec(1, n_panels + 1,
                                           width_ratios=[1]*n_panels + [0.05],
                                           wspace=0.25)

        for i, (mat, ttl) in enumerate(zip(mats, titles)):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
            ax.invert_yaxis()
            ax.set_title(ttl, fontsize=35, pad=10)
            ax.tick_params(axis="both", labelsize=14)

            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_xticklabels(np.arange(1, mat.shape[1] + 1))
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_yticklabels(np.arange(1, mat.shape[0] + 1))

        # shared colour‑bar
        cax = fig.add_subplot(gs[0, -1])
        mpl.colorbar.ColorbarBase(
            cax, cmap=cmap,
            norm=mpl.colors.Normalize(vmin=0, vmax=vmax),
            orientation="vertical", ticks=[]
        )
        cax.set_ylabel("Interaction strength", fontsize=35)

        ds_dir   = out_root / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        out_pdf  = ds_dir / f"{dataset}_{cond}_HEATMAPS.pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {out_pdf}")

if __name__ == "__main__":
    generate_heatmaps(
        source="./results-curated/HSC/", 
        output_base_dir="heatmaps",
        cmap="Reds"
    )
