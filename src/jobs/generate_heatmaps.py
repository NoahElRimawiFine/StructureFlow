# discover_items.py
from __future__ import annotations
from pathlib import Path
import re
import tempfile
import math
import shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import fitz
from PIL import Image

# ---------- CONFIG ----------
RESULTS_CURATED = Path("results-curated")
OUT_DIR         = RESULTS_CURATED / "panels"  # where final PDFs go
N_COLS = 11               # tweakable
CELL_IN = 5.0            # per-panel width/height in inches
CMAP = "Reds"


# The desired ordered sequence (PDFs appear after 'sincerities')
DISPLAY_ORDER = [
    "A_true",
    "glasso",
    "scode",
    "dyngenie3",
    "ngmsf2m",
    "sincerities",
    "PDF:OTVelo-Corr",
    "PDF:OTVelo-Granger",
    "tigon",
    "rf",
    "structureflow",
]

# ---------- CORE HELPERS ----------

def find_csvs_for_system_type(base_dir: Path, system: str, typ: str) -> Dict[str, Path]:
    """
    Return a dict {model_key -> file_path} for all CSVs for a given system and type.
    Includes 'A_true' which applies to BOTH types.
    
    Expected CSV layout:
      results-curated/{system}/{typ}/A_{model}_{typ}_{system}.csv
      results-curated/{system}/A_true_{system}.csv
    """
    assert typ in {"full", "wt"}, "typ must be 'full' or 'wt'"

    sys_dir = base_dir / system
    typ_dir = sys_dir / typ

    out: Dict[str, Path] = {}

    # A_true lives at the system root and applies to both types
    true_path = sys_dir / f"A_true.csv"
    if true_path.exists():
        out["A_true"] = true_path

    # Model CSVs live under {system}/{typ}/
    if typ_dir.exists():
        # Pattern: A_{model}_{typ}_{system}.csv
        rx = re.compile(rf"^A_(?P<model>.+?)_{typ}_{re.escape(system)}\.csv$", re.IGNORECASE)
        for p in typ_dir.glob("A_*.csv"):
            m = rx.match(p.name)
            if m:
                model = m.group("model").lower()
                out[model] = p

    return out


def find_pdfs_for_system_type(base_dir: Path, system: str, typ: str) -> Dict[str, Path]:
    """
    Return a dict of the two PDFs for a given system and type.
    
    PDFs live directly in results-curated/ and follow:
      OTVelo-Corr_{system}_{all|wt}.pdf
      OTVelo-Granger_{system}_{all|wt}.pdf
    
    Note: 'full' -> suffix 'all', 'wt' -> suffix 'wt'
    """
    assert typ in {"full", "wt"}, "typ must be 'full' or 'wt'"

    suffix = "all" if typ == "full" else "wt"
    root = base_dir

    pdfs = {}
    corr = root / f"OTVelo-Corr_{system}_{suffix}.pdf"
    granger = root / f"OTVelo-Granger_{system}_{suffix}.pdf"

    if corr.exists():
        pdfs["PDF:OTVelo-Corr"] = corr
    if granger.exists():
        pdfs["PDF:OTVelo-Granger"] = granger

    return pdfs

def discover_systems(base_dir: Path) -> List[str]:
    """
    Systems are subdirectories of results-curated/ (e.g., dyn-BF, dyn-TF, HSC, etc.).
    Ignores files (like the PDFs).
    """
    systems = []
    for p in base_dir.iterdir():
        if p.is_dir():
            systems.append(p.name)
    systems.sort()
    return systems


def list_all_ordered(base_dir: Path) -> List[Tuple[str, str, List[Tuple[str, Optional[Path]]]]]:
    """
    For each system and for typ in ['full', 'wt'], assemble the ordered list of items
    according to DISPLAY_ORDER. Returns a structure:
      [
        (system, typ, [(label, path_or_None), ...]),
        ...
      ]
    Missing items get path=None (so you can see gaps).
    """
    output = []
    systems = discover_systems(base_dir)
    for system in systems:
        for typ in ("full", "wt"):
            csvs = find_csvs_for_system_type(base_dir, system, typ)
            pdfs = find_pdfs_for_system_type(base_dir, system, typ)

            # Merge into one lookup table
            lookup = {}
            lookup.update(csvs)
            lookup.update(pdfs)

            # Build the ordered list
            ordered_items: List[Tuple[str, Optional[Path]]] = []
            for key in DISPLAY_ORDER:
                path = lookup.get(key)
                ordered_items.append((key, path))

            output.append((system, typ, ordered_items))
    return output


def _render_pdf_first_page_as_rgba(
    pdf_path: Path,
    dpi: int = 200,
    trim_whitespace: bool = True,
    white_threshold: int = 250,
    pad: int = 8,
    crop_box_frac: tuple[float, float, float, float] | None = None,  # (l, t, r, b) in [0,1]
) -> np.ndarray:
    import fitz, numpy as np
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(0)
        rect = page.rect
        # Optional crop BEFORE rendering
        if crop_box_frac is not None:
            l, t, r, b = crop_box_frac
            # fractions from each edge
            clip = fitz.Rect(
                rect.x0 + l * rect.width,
                rect.y0 + t * rect.height,
                rect.x1 - r * rect.width,
                rect.y1 - b * rect.height,
            )
        else:
            clip = None

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=True, clip=clip)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)

    if not trim_whitespace:
        return img

    # --- auto-trim (as you already had) ---
    rgb = img[..., :3]; alpha = img[..., 3]
    is_white = (rgb >= white_threshold).all(axis=-1)
    is_bg = is_white | (alpha == 0)
    if not np.any(~is_bg):  # fully blank
        return img
    rows = np.where((~is_bg).any(axis=1))[0]
    cols = np.where((~is_bg).any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]; c0, c1 = cols[0], cols[-1]
    r0 = max(0, r0 - pad); r1 = min(img.shape[0]-1, r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(img.shape[1]-1, c1 + pad)
    return img[r0:r1+1, c0:c1+1, :]


def _load_csv_matrix(csv_path: Path) -> np.ndarray:
    return pd.read_csv(csv_path, header=None).values.astype(float)

def _minmax_for_csvs(items):
    vals = []
    for label, path in items:
        if path and path.suffix.lower() == ".csv":
            try:
                a = _load_csv_matrix(path)
                vals.append((np.nanmin(a), np.nanmax(a)))
            except Exception as e:
                print(f"[warn] failed to read {path}: {e}")
    if not vals:
        return (0.0, 1.0)
    vmin = min(v[0] for v in vals)
    vmax = max(v[1] for v in vals)
    if vmin == vmax:
        eps = 1e-6
        return (vmin - eps, vmax + eps)
    return (float(vmin), float(vmax))

def _title(label: str, system: str) -> str:
    pretty = {
        "A_true": "Ground Truth",
        "glasso": "Glasso",
        "scode": "SCODE",
        "dyngenie3": "DynGENIE3",
        "ngmsf2m": r"$\mathrm{NGM{-}[SF]^{2}M}$",
        "sincerities": "SINCERITIES",
        "tigon": "TIGON",
        "rf": "RF",
        "structureflow": "StructureFlow",
        "PDF:OTVelo-Corr": "OTVelo-Corr",
        "PDF:OTVelo-Granger": "OTVelo-Granger",
    }

    base = pretty.get(label, label)

    # only attach the system for model panels (not PDFs or A_true)
    if label not in ("PDF:OTVelo-Corr", "PDF:OTVelo-Granger"):
        # remove the "dyn-" prefix if present
        sys_short = system.replace("dyn-", "").upper()
        base = f"{base} ({sys_short})"

    return base

def make_one_grid(system: str, typ: str, out_dir: Path = OUT_DIR):
    # Grab ordered items for just this (system, typ)
    items = next(((s, t, its)
                  for (s, t, its) in list_all_ordered(RESULTS_CURATED)
                  if s == system and t == typ), (None, None, []))[2]
    items = [(lbl, p) for (lbl, p) in items if p is not None]
    if not items:
        print(f"[info] Nothing for {system} ({typ})")
        return

    # Compute layout
    n = len(items)
    ncols = N_COLS
    nrows = math.ceil(n / ncols)

    # Shared heatmap scaling across *all CSVs* in this grid
    vmin, vmax = _minmax_for_csvs(items)

    # Figure size
    fig_w = CELL_IN * ncols + 20
    fig_h = CELL_IN * nrows + 1.5 * CELL_IN  
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw=dict(
            left=0.01,    # <- shrink figure margins
            right=0.95,   # <- leaves room for the colorbar column
            bottom=0.01,
            top=0.99,
            wspace=0.2,  # <- horizontal gap between panels
            hspace=0.02,  # <- vertical gap between panels
        ),
    )
    axes = axes.ravel()

    heatmap_handles = []

    for i, (label, path) in enumerate(items):
        ax = axes[i]
        if path.suffix.lower() == ".csv":
            arr = _load_csv_matrix(path)
            arr = np.abs(arr)
            d = min(arr.shape[0], arr.shape[1])
            idx = np.arange(d)
            arr[idx, idx] = 0.0
            if any(key in str(path).lower() for key in ("ngmsf2m", "scode")):
                im = ax.imshow(arr, vmin=0, vmax=arr.max(), cmap=CMAP)
            else:
                im = ax.imshow(arr, vmin=0, vmax=1, cmap=CMAP)
            ax.invert_yaxis()
            heatmap_handles.append(im)
            ax.tick_params(labelsize=9, length=2)
        elif path.suffix.lower() == ".pdf":
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox

            # 1) render first page (already cropped/trimmed if you want)
            img = _render_pdf_first_page_as_rgba(
                path, dpi=500, trim_whitespace=True, pad=6,
                # crop_box_frac=(0.0, 0.12, 0.0, 0.0),  # optional: chop top 12% off
            )

            # 2) draw the PDF image
            oi = OffsetImage(img, zoom=0.25)
            ab = AnnotationBbox(
                oi, (0.5, 0.5),
                xycoords='axes fraction',
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=1,                    # put the image *behind* overlays
            )
            ax.add_artist(ab)

            # 3) overlay a white band at the top to hide the built-in title
            #    (height=0.12 is ~top 12% of the axis; tweak to fit your PDFs)
            white_band = Rectangle(
                (0.0, 0.72), 1.0, 0.3,      # (left, bottom), width, height in axes coords
                transform=ax.transAxes,
                fc="white", ec="none",
                zorder=11                 
            )
            ax.add_patch(white_band)

            # 4) add your custom "hand-written" title on top of the band
            sys_short = system.replace("dyn-", "").upper()
            pdf_name = "OTVelo-Corr" if "Corr" in label else "OTVelo-Granger"
            ax.text(
                0.54, 0.77, f"{pdf_name} ({sys_short})",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=40,  
                zorder=11
            )

            # 5) clean the panel chrome
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False) 
        else:
            ax.text(0.5, 0.5, f"Unsupported: {path.suffix}", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])

        if path.suffix.lower() != ".pdf":
            ax.set_title(_title(label, system), fontsize=40, pad=6)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Shared colorbar if we had any heatmaps
    if heatmap_handles:
        cax = fig.add_axes([0.97, 0.25, 0.010, 0.51])  # [left, bottom, width, height] in figure coords
        cbar = fig.colorbar(heatmap_handles[0], cax=cax)
        cbar.ax.set_ylabel("Interaction Strength", rotation=90, va="center", fontsize=40, labelpad=21)
        cbar.set_ticks([])

    # fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{system}_{typ}_grid.pdf"
    with PdfPages(out_path) as pp:
        pp.savefig(fig, dpi=500)

    plt.close(fig)
    print(f"[ok] wrote {out_path}")

def make_all_grids():
    for system in discover_systems(RESULTS_CURATED):
        for typ in ("full", "wt"):
            make_one_grid(system, typ)

if __name__ == "__main__":
    make_all_grids()


# def print_verification(base_dir: Path) -> None:
#     """
#     Pretty-print a verification report showing which files were found (✓) or missing (✗)
#     in the specified order for each (system, type).
#     """
#     all_found = list_all_ordered(base_dir)

#     for system, typ, items in all_found:
#         print(f"\n=== System: {system} | Type: {typ} ===")
#         for label, path in items:
#             mark = "✓" if path and path.exists() else "✗"
#             print(f"  {mark} {label:20s} -> {str(path) if path else 'MISSING'}")


