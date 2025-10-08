import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
import anndata as ad
import scanpy as sc
import networkx as nx


def load_adata(path, log_transform=True):
    expr_df = pd.read_csv(os.path.join(path, "ExpressionData.csv"), index_col=0)
    adata = ad.AnnData(expr_df.T)

    df_pt = pd.read_csv(os.path.join(path, "PseudoTime.csv"), index_col=0)

    df_pt_aligned = df_pt.loc[adata.obs_names, "PseudoTime"]
    adata.obs["t_sim"] = df_pt_aligned.values

    if log_transform:
        sc.pp.log1p(adata)
    return adata


def load_reference_network(path):
    df = pd.read_csv(os.path.join(path, "refNetwork.csv"))
    return df


def create_network_graph(ref_network_df, ax, title=""):
    G = nx.DiGraph()

    for _, row in ref_network_df.iterrows():
        source = row["Gene1"]
        target = row["Gene2"]
        edge_type = row["Type"]

        G.add_node(source)
        G.add_node(target)
        G.add_edge(source, target, type=edge_type)

    if title == "LL":
        nodes = sorted(G.nodes(), key=lambda x: int(x[1:]))
        pos = {}
        n = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n - np.pi / 2
            pos[node] = (np.cos(angle), np.sin(angle))
    else:
        pos = nx.circular_layout(G)

    node_color = "#4A90E2"
    node_size = 800
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_color, node_size=node_size, alpha=1.0
    )

    node_radius = np.sqrt(node_size) / 150.0

    for edge in G.edges(data=True):
        source, target, data = edge

        if source == target:
            continue

        source_pos = np.array(pos[source])
        target_pos = np.array(pos[target])

        direction = target_pos - source_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            continue

        direction_normalized = direction / distance

        start_pos = source_pos + direction_normalized * node_radius * 0.5
        end_pos = target_pos - direction_normalized * node_radius * 0.5

        arrow = FancyArrowPatch(
            start_pos,
            end_pos,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            color="black",
            linewidth=1.5,
            mutation_scale=30,
            alpha=0.6,
            zorder=1,
        )
        ax.add_patch(arrow)

    ax.set_title(title, fontsize=28, fontweight="bold", pad=10)
    ax.axis("off")
    ax.set_xlim(
        [min(x for x, y in pos.values()) - 0.15, max(x for x, y in pos.values()) + 0.15]
    )
    ax.set_ylim(
        [min(y for x, y in pos.values()) - 0.15, max(y for x, y in pos.values()) + 0.15]
    )


def create_trajectory_pca(adata, ax, title="", T=5):
    t_bins = np.linspace(0, 1, T + 1)[:-1]
    adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(adata.X)

    times = adata.obs["t"].values

    scatter = ax.scatter(
        pca_data[:, 0], pca_data[:, 1], c=times, cmap="viridis", s=30, alpha=0.6
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    return scatter


def get_system_name(path):
    basename = os.path.basename(path)

    if basename.startswith("dyn-"):
        name = basename[4:]
    elif basename.startswith("HSC"):
        if basename == "HSC":
            name = "HSC"
        else:
            name = "HSC"
    else:
        name = basename

    return name


def main():
    data_path = "data"

    synthetic_systems = ["dyn-TF", "dyn-LL", "dyn-BF", "dyn-CY", "dyn-SW"]

    all_systems = []
    for sys in synthetic_systems:
        pattern = os.path.join(data_path, "Synthetic", sys, f"{sys}*-1")
        matches = glob.glob(pattern)
        if matches:
            all_systems.append(("Synthetic", sys, os.path.dirname(matches[0])))

    n_systems = len(all_systems)

    fig = plt.figure(figsize=(5 * n_systems + 1, 10))

    gs = fig.add_gridspec(2, n_systems, hspace=0.3, wspace=0.3, left=0.08)

    row_labels = ["Static network", "Trajectory"]
    for row_idx, label in enumerate(row_labels):
        fig.text(
            0.02,
            0.75 - row_idx * 0.48,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=28,
            fontweight="bold",
        )

    for col_idx, (dataset_type, system_name, system_path) in enumerate(all_systems):
        print(f"Processing {system_name} ({dataset_type})...")

        ref_network = load_reference_network(system_path)

        adata = load_adata(system_path, log_transform=True)

        ax_network = fig.add_subplot(gs[0, col_idx])
        create_network_graph(
            ref_network, ax_network, title=get_system_name(system_name)
        )

        ax_pca = fig.add_subplot(gs[1, col_idx])
        scatter = create_trajectory_pca(adata, ax_pca, title="", T=5)

    plt.savefig("systems_overview.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("systems_overview.png", dpi=300, bbox_inches="tight")
    print("\nFigure saved as 'systems_overview.pdf' and 'systems_overview.png'")
    plt.close()


if __name__ == "__main__":
    main()
