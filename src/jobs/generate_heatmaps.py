import os
import re
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fitz 

# =======================
# CONFIG: edit these
# =======================
BASE_DIR = "./src/jobs/"
PDF_BASE = BASE_DIR
PDF_PATHS = [
    "results-curated/*.pdf"
]
OUTPUT_BASENAME = "combined_grid"  # will produce combined_grid.pdf and .png
TITLE_MAP_FROM_PATH = True  # derive titles from file paths
MODELS = None
MODEL_FILTER = None

CSV_RX = re.compile(
    r"results-curated[/\\](?P<system>[^/\\]+)[/\\](?P<kind>full|wt)[/\\]"
    r"A_(?P<model>[^/\\]+?)(?:_T5)?_full_(?P<sysfile>[^/\\]+)\.csv$",
    re.IGNORECASE,
)

PDF_RX = re.compile(
    r"(?P<model>[^/\\_]+)_(?P<system>[^/\\_]+)_(?P<kind>wt|all)\.pdf$",
    re.IGNORECASE,
)

def find_csvs(base_dir):
    paths = glob(os.path.join(base_dir, "results-curated", "**", "*.csv"), recursive=True)
    assets = {}
    for p in paths:
        m = CSV_RX.search(p.replace("\\", "/"))
        if not m:
            continue
        system = m.group("system")
        kind = m.group("kind").lower()          # 'full' or 'wt'
        model = m.group("model")
        if MODEL_FILTER and model != MODEL_FILTER:
            continue
        d = assets.setdefault(system, {
            "csv": {"wt": {}, "full": {}},
            "pdf": {"wt": {}, "all": {}},
            "models": set(),
        })
        d["csv"][kind][model] = p               # <-- keep per-model
        d["models"].add(model)
    return assets

def find_pdfs(pdf_root, assets):
    paths = glob(os.path.join(pdf_root, "**", "*.pdf"), recursive=True)
    for p in paths:
        bn = os.path.basename(p)
        m = PDF_RX.match(bn)
        if not m:
            continue
        model = m.group("model")
        system = m.group("system")
        kind = m.group("kind").lower()          # 'wt' or 'all'
        if MODEL_FILTER and model != MODEL_FILTER:
            continue
        d = assets.setdefault(system, {
            "csv": {"wt": {}, "full": {}},
            "pdf": {"wt": {}, "all": {}},
            "models": set(),
        })
        d["pdf"][kind][model] = p               # <-- keep per-model
        d["models"].add(model)
    return assets

def render_pdf_first_page(pdf_path, dpi=250):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=True)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return img

def plot_csv_heatmap(ax, csv_path, title):
    try:
        df = pd.read_csv(csv_path)
        num = df.select_dtypes(include="number")
        num = np.abs(num).T
        im = ax.imshow(num.values, cmap="Reds", vmin=0, vmax=1, aspect="auto", interpolation="nearest")

        # minimal chrome
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.margins(0)

        # compact title
        ax.set_title(title, fontsize=8, pad=2)
    except Exception as e:
        ax.text(0.5, 0.5, f"CSV error:\n{e}", ha="center", va="center", fontsize=8)
        ax.set_axis_off()

def plot_pdf(ax, pdf_path, title):
    try:
        img = render_pdf_first_page(pdf_path, dpi=300)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if title:  # you call with title=None in your row function
            ax.set_title(title, fontsize=8, pad=2)
    except Exception as e:
        ax.text(0.5, 0.5, f"PDF error:\n{e}", ha="center", va="center", fontsize=8)
        ax.set_axis_off()

def plot_strength_bar(ax, label="Strength"):
    h = 800
    grad = np.linspace(0.2, 1.0, h).reshape(h, 1)  # light -> dark red
    ax.imshow(grad, aspect="auto", cmap="Reds", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=10, pad=4)

def make_grid_for_kind(assets, kind_csv: str, kind_pdf: str, out_suffix: str):
    # systems that have at least a CSV or PDF for this kind
    systems = sorted([
        s for s, d in assets.items()
        if d["csv"].get(kind_csv) or d["pdf"].get(kind_pdf)
    ])
    if not systems:
        print(f"No systems found for kind_csv='{kind_csv}', kind_pdf='{kind_pdf}'. Skipping.")
        return

    n = len(systems)
    fig = plt.figure(figsize=(14, 2.8 * n), dpi=150)
    # 3 columns: CSV (kind) | PDF (kind) | Strength
    gs = GridSpec(nrows=n, ncols=3, width_ratios=[1, 1.2, 0.25], wspace=0.15, hspace=0.32)

    for i, sys in enumerate(systems):
        d = assets[sys]
        model = d.get("model", "")

        csv_path = d["csv"].get(kind_csv)
        pdf_path = d["pdf"].get(kind_pdf)

        # Column 0: CSV (kind)
        ax0 = fig.add_subplot(gs[i, 0])
        if csv_path:
            title = f"{model}"
            plot_csv_heatmap(ax0, csv_path, title)
        else:
            ax0.text(0.5, 0.5, f"CSV {kind_csv}\nmissing", ha="center", va="center", fontsize=9)
            ax0.set_axis_off()

        # Column 1: PDF (kind)
        ax1 = fig.add_subplot(gs[i, 1])
        if pdf_path:
            title = f"{sys}"
            plot_pdf(ax1, pdf_path, title)
        else:
            ax1.text(0.5, 0.5, f"PDF {kind_pdf}\nmissing", ha="center", va="center", fontsize=9)
            ax1.set_axis_off()

    # Strength bar spanning all rows
    ax_bar = fig.add_subplot(gs[:, 2])
    # Label the bar with the out_suffix so each figure is self-explanatory
    label = "Strength (WT)" if out_suffix == "wt" else "Strength (FULL/ALL)"
    plot_strength_bar(ax_bar, label=label)

    out_pdf = f"{OUTPUT_BASENAME}_{out_suffix}.pdf"
    out_png = f"{OUTPUT_BASENAME}_{out_suffix}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Wrote: {out_pdf} and {out_png}")

def make_single_system_row(system_name: str, d: dict, kind_csv: str, kind_pdf: str, out_suffix: str):
    models = sorted(d.get("models", []))
    if not models:
        print(f"Skipping {system_name} ({out_suffix}): no models found.")
        return

    # Fixed 10 columns (pad with blanks if fewer)
    max_cols = 10
    ncols = min(len(models), max_cols)
    models = models[:max_cols]  # truncate if >10

    # Adjust figure width dynamically so 10 columns fits nicely on one page
    fig_width = 2.2 * ncols if ncols < 10 else 30
    fig = plt.figure(figsize=(fig_width, 10), dpi=200)
    gs = GridSpec(nrows=2, ncols=ncols, wspace=0.05, hspace=0.0)

    for j, model in enumerate(models):
        ax = fig.add_subplot(gs[0, j])

        pdf_path = d["pdf"].get(kind_pdf, {}).get(model)
        csv_path = d["csv"].get(kind_csv, {}).get(model)

        if pdf_path:
            plot_pdf(ax, pdf_path, title=None)
            ax.set_title(model, fontsize=8, pad=2)
        elif csv_path:
            plot_csv_heatmap(ax, csv_path, model)
            ax.set_title(model, fontsize=8, pad=2)
        else:
            ax.set_axis_off()

    # Add padding for fewer than 10 models
    if len(models) < max_cols:
        for j in range(len(models), max_cols):
            ax = fig.add_subplot(gs[0, j])
            ax.set_axis_off()

    safe_sys = system_name.replace("/", "-")
    out_pdf = f"{OUTPUT_BASENAME}_{safe_sys}_{out_suffix}.pdf"
    out_png = f"{OUTPUT_BASENAME}_{safe_sys}_{out_suffix}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Wrote: {out_pdf} and {out_png}")


def main():
    # Collect assets once
    assets = find_csvs(BASE_DIR)
    assets = find_pdfs(PDF_BASE, assets)

    systems = sorted([s for s, d in assets.items() if d.get("models")])
    if not systems:
        raise SystemExit("No systems found. Check BASE_DIR/PDF_BASE and naming patterns.")

    for sys in systems:
        d = assets[sys]

        # WT output for this system
        if d["csv"].get("wt") or d["pdf"].get("wt"):
            make_single_system_row(system_name=sys, d=d, kind_csv="wt",   kind_pdf="wt",  out_suffix="wt")
        else:
            print(f"⚠️ {sys}: no WT assets. Skipping.")

        # FULL/ALL output for this system
        if d["csv"].get("full") or d["pdf"].get("all"):
            make_single_system_row(system_name=sys, d=d, kind_csv="full", kind_pdf="all", out_suffix="full")
        else:
            print(f"⚠️ {sys}: no FULL/ALL assets. Skipping.")

if __name__ == "__main__":
    main()