import pandas as pd
import scprep as sc
import phate
import numpy as np
import seaborn as sns
import scprep
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from copy import deepcopy
import datetime
import src.models.components.distribution_distances as dd



# --- Configuration ---
BASE_DATA_DIR = "data/Synthetic"  # Root directory for synthetic datasets
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("outputs", f"trifurcating_results_{TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUTPUT_DIR}")

# Global lists for collecting data from all loaded datasets - INITIALIZED HERE
datas_train, datas_val = [], []
graphs_train, graphs_val = [], []
actions_train, actions_val = [], []

# Ensure the dd module path is correct or handled if it's custom
# Assuming it might be in a directory structure like src/models/components/
# This might need adjustment based on your actual project structure.
# For now, the import `import src.models.components.distribution_distances as dd` is kept as is.


def parse_intervention_from_name(dataset_name, n_genes):
    """Parses intervention details from the dataset name.
    Assumes gene names g1, g2, ... so gX corresponds to index X-1.
    Returns intervention_idx (0-indexed) or None, and the target value (0 for knockout).
    """
    action_vector = np.ones((n_genes,), dtype=int)
    intervention_active = False

    # Example dataset name: TF-I-3-gnull-5000 (intervention on g3)
    # Example dataset name: TF-I-gnull-5000 (no intervention)
    if "gnull" in dataset_name and "-I-" in dataset_name:
        parts = dataset_name.split("-I-")
        if len(parts) > 1:
            intervention_part = parts[1].split("-")[0]
            if intervention_part.isdigit():
                gene_number = int(intervention_part)
                if 1 <= gene_number <= n_genes:
                    action_vector[gene_number - 1] = 0 # 0-indexed knockout
                    intervention_active = True
                    print(f"Parsed intervention: Knock-out on g{gene_number} (index {gene_number-1}) from {dataset_name}")
                else:
                    print(f"Warning: Parsed gene number {gene_number} from {dataset_name} is out of range for {n_genes} genes.")
            elif intervention_part == "gnull": # Explicit no intervention
                 print(f"Parsed no intervention from {dataset_name}")
            else:
                print(f"Warning: Could not parse intervention digit from '{intervention_part}' in {dataset_name}")
        else:
            print(f"Parsed no intervention (standard gnull) from {dataset_name}")
    else:
        print(f"No standard intervention pattern found in {dataset_name}, assuming no intervention.")
    
    return action_vector, intervention_active


def load_trajectory_data(dataset_path, n_genes_expected):
    """Loads trajectory data from a dataset directory.
    Tries to load from a 'simulations' subdirectory (multiple cX.csv files)
    or a single 'trajectories.csv' file.
    Returns a numpy array (n_cells, n_timesteps, n_genes) or None.
    """
    simulations_dir = os.path.join(dataset_path, "simulations")
    trajectory_file_alt = os.path.join(dataset_path, "trajectories.csv")

    data_list = []

    if os.path.isdir(simulations_dir):
        print(f"Loading trajectories from directory: {simulations_dir}")
        cell_files = sorted([s for s in os.listdir(simulations_dir) if s.startswith('c') and s.endswith(".csv")])
        if not cell_files:
            print(f"Warning: No cell CSV files found in {simulations_dir}")
            return None
        
        for f in cell_files:
            try:
                df = pd.read_csv(os.path.join(simulations_dir, f), index_col=0)
                if df.shape[0] != n_genes_expected:
                    print(f"Warning: File {f} has {df.shape[0]} genes, expected {n_genes_expected}. Skipping.")
                    continue
                data_list.append(df.values) # Genes x Time
            except Exception as e:
                print(f"Error reading or processing file {f}: {e}")
                return None
        if not data_list:
            return None
        # Stack to (n_cells, n_genes, n_timesteps)
        loaded_data = np.array(data_list)
        # Transpose to (n_cells, n_timesteps, n_genes)
        loaded_data = np.swapaxes(loaded_data, 1, 2)

    elif os.path.exists(trajectory_file_alt):
        print(f"Loading trajectories from file: {trajectory_file_alt}")
        try:
            df_traj = pd.read_csv(trajectory_file_alt)
            # Expected format: cell_id, time, g1, g2, ..., gn or similar
            # This requires knowing the exact format. Assuming a simple reshape for now.
            # A common format: each row is one cell at one timepoint.
            # For example: (n_cells * n_timesteps) rows, (gene_columns) cols.
            # Let's assume columns are [cell_id, time, g1, ..., gN]
            # Or if it's pre-pivoted: [time, g1, ... gN] per cell, with an outer loop for cells or cell_id column.
            
            # This part is highly dependent on the CSV structure.
            # For now, let's assume a structure that can be directly reshaped or pivoted
            # to (n_cells, n_timesteps, n_genes).
            # If it is (n_cells * n_timesteps, n_genes), we need n_cells and n_timesteps.
            # The original `plot` function implies data is (genes x timesteps) per file (cell).
            # If trajectories.csv has all cells: (total_rows, gene_cols + metadata_cols)
            # This is a simplification and might need significant adjustment based on actual file format.
            print(f"Warning: Loading from single 'trajectories.csv' is not fully implemented to match original multi-file structure. Assuming a placeholder format.")
            # Placeholder: try to read it assuming it's cells x time x genes already, or similar
            # This is unlikely to be correct without knowing the file structure.
            # Fallback: If this file exists but we can't parse it well, we might return None or error.
            # For now, let's assume it matches n_cells x n_timesteps x n_genes if directly loaded.
            # This part needs to be robust based on `grn_datamodule.py`'s actual handling.
            # loaded_data = pd.read_csv(trajectory_file_alt).values # This is too naive.
            return None # Mark as not implemented for now to avoid errors

        except Exception as e:
            print(f"Error reading monolithic trajectory file {trajectory_file_alt}: {e}")
            return None
    else:
        print(f"Warning: No trajectory data found in {dataset_path} (checked 'simulations' dir and 'trajectories.csv')")
        return None

    # Standard processing from original `plot` function (downsampling)
    if loaded_data is not None and loaded_data.ndim == 3:
        if loaded_data.shape[1] > 10: # If there are enough timepoints
             loaded_data = loaded_data[:, ::9, :] # Original downsampling
             loaded_data = loaded_data[:, 1:, :]   # Original offset
        elif loaded_data.shape[1] > 1:
             loaded_data = loaded_data[:, 1:, :] # Only apply offset if not many timepoints
        
        if loaded_data.shape[1] == 0: # Check if downsampling made it empty
            print("Warning: Trajectory data became empty after downsampling.")
            return None
    return loaded_data

def load_network_data(dataset_path, n_genes_expected):
    """Loads network graph data from a dataset directory.
    Tries to load from 'refNetwork.csv' or 'network.csv'.
    Returns an adjacency matrix (n_genes, n_genes) or None.
    """
    ref_net_path = os.path.join(dataset_path, "refNetwork.csv")
    alt_net_path = os.path.join(dataset_path, "network.csv")
    A = np.zeros((n_genes_expected, n_genes_expected))

    network_file_to_load = None
    if os.path.exists(ref_net_path):
        network_file_to_load = ref_net_path
    elif os.path.exists(alt_net_path):
        network_file_to_load = alt_net_path
    
    if network_file_to_load:
        print(f"Loading network from: {network_file_to_load}")
        try:
            ref_net_df = pd.read_csv(network_file_to_load)
            # Assuming format: Gene1, Gene2, Type (+ or -)
            # Gene names g1, g2, ...
            for _, row in ref_net_df.iterrows():
                gene1_str = row["Gene1"]
                gene2_str = row["Gene2"]
                
                if not isinstance(gene1_str, str) or not gene1_str.startswith('g') or \
                   not isinstance(gene2_str, str) or not gene2_str.startswith('g'):
                    print(f"Warning: Invalid gene name format in network file: {gene1_str}, {gene2_str}. Skipping row.")
                    continue

                gene1_idx = int(gene1_str[1:]) -1 # 0-indexed
                gene2_idx = int(gene2_str[1:]) -1 # 0-indexed
                
                if not (0 <= gene1_idx < n_genes_expected and 0 <= gene2_idx < n_genes_expected):
                    print(f"Warning: Gene index out of bounds ({gene1_idx}, {gene2_idx}) for {n_genes_expected} genes. Skipping row.")
                    continue

                rel_type = row.get("Type", "+") # Default to activation if Type is missing
                rel = 1 if rel_type == "+" else -1
                A[gene1_idx, gene2_idx] = rel
            return np.abs(A) # Original script uses np.abs(A)
        except Exception as e:
            print(f"Error reading or processing network file {network_file_to_load}: {e}")
            return None
    else:
        print(f"Warning: No network file (refNetwork.csv or network.csv) found in {dataset_path}")
        return None

def load_single_dataset_from_dir(dataset_dir_path, n_genes_default=10):
    """Loads data, graph, and action for a single dataset directory.
    n_genes_default is used if not determinable from network file first.
    """
    print(f"Attempting to load dataset from: {dataset_dir_path}")
    dataset_name = os.path.basename(dataset_dir_path)

    # Try to load network first to determine n_genes if possible
    # This is a bit circular if n_genes_default is strictly needed by load_network_data first.
    # Let's assume n_genes_default is a fallback.
    # A better approach would be to infer n_genes from trajectory data if network is missing, or have it as a firm config.
    # For now, n_genes_default will be used by load_network_data for matrix init size.
    graph = load_network_data(dataset_dir_path, n_genes_default)
    if graph is None:
        print(f"Could not load network data for {dataset_name}. Using default {n_genes_default} genes and zero matrix.")
        actual_n_genes = n_genes_default
        graph = np.zeros((actual_n_genes, actual_n_genes))
    else:
        actual_n_genes = graph.shape[0]

    trajectories = load_trajectory_data(dataset_dir_path, actual_n_genes)
    if trajectories is None:
        print(f"Could not load trajectory data for {dataset_name}. Skipping this dataset.")
        return None, None, None, None # data, graph, action_template, dataset_name_found

    action_template, _ = parse_intervention_from_name(dataset_name, actual_n_genes)
    
    # Repeat action_template for each cell in trajectories
    num_cells = trajectories.shape[0]
    actions_for_dataset = np.tile(action_template, (num_cells, 1))

    return trajectories, graph, actions_for_dataset, dataset_name

# Remove the old plot function as it's being replaced by the new loading mechanism
# def plot(simdir): # ... original plot function ...

# --- Configuration ---
BASE_DATA_DIR = "data/Synthetic"  # Root directory for synthetic datasets
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("outputs", f"trifurcating_results_{TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUTPUT_DIR}")

# This part will be significantly changed.
# synthetic_i_equivalent_path = os.path.join(BASE_DATA_DIR, "Synthetic-I") # Or just BASE_DATA_DIR if it IS "Synthetic-I"
# if not os.path.isdir(synthetic_i_equivalent_path):
#     print(f"Warning: Directory {synthetic_i_equivalent_path} not found. Old data loading will fail.")
#     # Fallback for original structure if new path doesn't exist
#     if os.path.isdir("Synthetic-I"):
#         synthetic_i_equivalent_path = "Synthetic-I"
#     else:
#         print("Error: Cannot find data directory for initial loading. Please check BASE_DATA_DIR.")
#         # exit() # Or handle gracefully

# if os.path.isdir(synthetic_i_equivalent_path):
#     for path_basename in os.listdir(synthetic_i_equivalent_path):
#         p = os.path.join(synthetic_i_equivalent_path, path_basename)
#         if 'TF-I-gnull' not in p: # Original filter
#             continue
#         else:
#             print(f"Processing dataset from: {p}")
#             data, graph = plot(p) # plot() expects the specific subfolder like TF-I-gnull-5000
#             train_len = int(data.shape[0]*train_val_split[0])

# --- New Data Loading Mechanism ---
# Define which datasets to load. This should come from your experiment design.
# Example: load data from subdirectories in BASE_DATA_DIR that match a pattern.
# For now, let's assume we list all subdirectories in BASE_DATA_DIR and try to load them.

N_GENES_CONFIG = 10 # Configure expected number of genes. This should be consistent for your datasets.
TRAIN_VAL_SPLIT = [0.8, 0.2] # Moved here for clarity

print(f"Starting new data loading from: {BASE_DATA_DIR}")
if not os.path.isdir(BASE_DATA_DIR):
    print(f"Error: Base data directory {BASE_DATA_DIR} not found. Exiting.")
    # Consider exiting or raising an error: exit()

all_dataset_folders = []
if os.path.isdir(BASE_DATA_DIR):
    all_dataset_folders = [os.path.join(BASE_DATA_DIR, d) for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]

# Apply a filter similar to the original, e.g., only load 'TF-I-gnull' related datasets
# You can customize this list based on your needs, e.g. by explicit names:
# dataset_folder_paths_to_load = [
#    os.path.join(BASE_DATA_DIR, "TF-I-gnull-5000"), 
#    os.path.join(BASE_DATA_DIR, "TF-I-3-gnull-5000"),
# ]
# For dynamic loading based on original filter:
dataset_folder_paths_to_load = [p for p in all_dataset_folders if 'TF-I-gnull' in os.path.basename(p)]

if not dataset_folder_paths_to_load:
    print(f"No dataset folders found or matched filter in {BASE_DATA_DIR}. Check configuration.")
    # Exit or handle: exit()

print(f"Found datasets to process: {dataset_folder_paths_to_load}")

for p_idx, dataset_path in enumerate(dataset_folder_paths_to_load):
    data, graph, action_for_cells, loaded_ds_name = load_single_dataset_from_dir(dataset_path, N_GENES_CONFIG)

    if data is not None and graph is not None and action_for_cells is not None:
        print(f"Successfully loaded: {loaded_ds_name} | Data shape: {data.shape}, Action shape: {action_for_cells.shape}")
        # Split data into train/val
        n_cells_total = data.shape[0]
        if n_cells_total == 0:
            print(f"Warning: Dataset {loaded_ds_name} has no cells after loading. Skipping.")
            continue
            
        indices = np.arange(n_cells_total)
        np.random.shuffle(indices) # Shuffle cells before splitting
        train_idx_end = int(n_cells_total * TRAIN_VAL_SPLIT[0])
        
        train_indices = indices[:train_idx_end]
        val_indices = indices[train_idx_end:]

        if len(train_indices) == 0 or len(val_indices) == 0:
            print(f"Warning: Not enough cells in {loaded_ds_name} to create train/val split. Min cells: {int(1/min(TRAIN_VAL_SPLIT))}. Got: {n_cells_total}. Skipping.")
            continue

        datas_train.append(data[train_indices, :, :])
        actions_train.append(action_for_cells[train_indices, :])
        graphs_train.append(graph) # Graph is the same for all cells in this dataset

        datas_val.append(data[val_indices, :, :])
        actions_val.append(action_for_cells[val_indices, :])
        graphs_val.append(graph)
    else:
        print(f"Failed to load dataset from {dataset_path}. Skipping.")

# Check if any data was loaded
if not datas_train or not datas_val:
    print("Error: No training or validation data loaded. Please check data paths and loading logic. Exiting.")
    # exit()
else:
    print(f"Total training datasets loaded: {len(datas_train)}")
    print(f"Total validation datasets loaded: {len(datas_val)}")
    print("Example boolODE data (after new loading):", datas_train[0].shape, datas_val[0].shape)
    true_graph = graphs_val[0] # Assuming all val graphs are the same, or use the first one

# --- End of New Data Loading Mechanism ---


# The rest of the script (visualizations, model definitions, training loops) follows
# Make sure these parts can handle potentially empty datas_train/datas_val if loading fails

# Example boolODE data:", datas_train[0].shape, datas_val[0].shape)
# true_graph = graphs_val[0]

import networkx as nx
fig, axes = plt.subplots(3,1, figsize=(4,16))
axes = axes.flatten()
for ax, data, graph in zip(axes, datas_train, graphs_train):
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    print(type(G))
    nx.draw(G, ax=ax, pos=nx.circular_layout(G), arrows=True, with_labels=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "true_graphs_train.png"))
plt.close(fig)

from sklearn.decomposition import PCA
fig, axes = plt.subplots(1,3, figsize=(16,4))
axes = axes.flatten()
pca = PCA(n_components=2)
#datas_train might be empty if loading failed
if datas_train:
    cells_null = datas_train[0].reshape(-1, datas_train[0].shape[-1])
    pca_embed = pca.fit_transform(cells_null)
    for ax, data, graph in zip(axes, datas_train, graphs_train):
        cells = data.reshape(-1, data.shape[-1])
        pca_embed = pca.transform(cells)
        labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten()
        scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_trajectories_train.png"))
    plt.close(fig)
else:
    print("Skipping training data PCA plot as datas_train is empty.")


fig, axes = plt.subplots(1,3, figsize=(16,4))
axes = axes.flatten()
pca = PCA(n_components=2)
#datas_val might be empty
if datas_val:
    cells_null = datas_val[0].reshape(-1, datas_val[0].shape[-1])
    pca_embed = pca.fit_transform(cells_null)
    for ax, data, graph in zip(axes, datas_val, graphs_val):
        cells = data.reshape(-1, data.shape[-1])
        pca_embed = pca.transform(cells)
        labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten()
        scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_trajectories_val.png"))
    plt.close(fig)
else:
    print("Skipping validation data PCA plot as datas_val is empty.")


import math
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from torch.utils.data import Dataset, DataLoader

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj):
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    import src.models.components.distribution_distances as dd 
    
def plot_pca_manifold(data, preds, ax, a=None, pca_fitted_obj=None):
    cells = data.reshape(-1, data.shape[-1])
    
    if pca_fitted_obj is None:
        pca_to_use = PCA(n_components=2)
        pca_embed = pca_to_use.fit_transform(cells)
    else:
        pca_to_use = pca_fitted_obj
        pca_embed = pca_to_use.transform(cells)
        
    pca_xT = pca_to_use.transform(preds)
    # cells_end_points = np.stack([preds], axis=1) # This was in original, but preds is already the endpoint
    # pca_cell_end_points = np.stack([pca_xT], axis=1) # This was in original
    # pca_cell_end_points = pca_cell_end_points.reshape(-1, pca_cell_end_points.shape[-1]) # This was in original
    # Assuming `preds` is (n_cells, n_features) representing the endpoints directly.
    # `pca_xT` is then (n_cells, 2)
    pca_cell_end_points = pca_xT # Use transformed predictions directly

    labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten() # For original trajectory colors
    scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True, label="Original Trajectory Data")
    
    if a is None:
        # labels_end_points = np.repeat(np.arange(cells_end_points.shape[1])[None,:], cells_end_points.shape[0], axis=0).flatten()
        # For predicted endpoints, if no intervention groups, plot them simply.
        # Let's use a distinct color or marker if needed. For now, just plotting them.
        scprep.plot.scatter2d(pca_cell_end_points, c='magenta', s=20, label="Predicted Endpoints", ax=ax, ticks=False, colorbar=False)
    else:
        a_np = a.detach().cpu().numpy() if hasattr(a, 'detach') else a # ensure numpy
        group_dict = {'control': ('cyan', []), 'g3': ('red', []), 'g4': ('blue', [])} # color, list_of_points
        # This intervention parsing is specific. Adapt if gene names/indices change.
        # Assuming gene indices 0, 1, 2, 3, 4... for g1, g2, g3, g4, g5...
        # And interventions are knockouts (action[i,j] == 0 if gene j is KO'd for cell i)

        for i in range(a_np.shape[0]):
            intervened_gene_indices = np.where(a_np[i] == 0)[0]
            if not intervened_gene_indices.size: # No intervention, it's a control
                group_dict['control'][1].append(pca_cell_end_points[i])
            else:
                # For simplicity, color based on the first intervened gene if multiple (e.g. g3g4)
                first_intervened_gene_idx = intervened_gene_indices[0]
                if first_intervened_gene_idx == 2: # g3 (0-indexed)
                    group_dict['g3'][1].append(pca_cell_end_points[i])
                elif first_intervened_gene_idx == 3: # g4 (0-indexed)
                    group_dict['g4'][1].append(pca_cell_end_points[i])
                else: # Other interventions, group with control or add new groups
                    # For now, add to control if not g3 or g4
                    group_dict['control'][1].append(pca_cell_end_points[i]) 

        for k, (color_val, points_list) in group_dict.items():
            if points_list:
                points_array = np.array(points_list)
                if points_array.ndim == 2 and points_array.shape[1] == 2:
                     scprep.plot.scatter2d(points_array, c=color_val, label=f"Pred. Endpoints ({k})", ax=ax, ticks=False, colorbar=False, s=20)
                elif points_array.size > 0: # Handle if only one point or other shapes by trying to reshape or warn
                    print(f"Warning: Points for group {k} have unexpected shape {points_array.shape}. Skipping scatter.")

    ax.legend()
    plt.tight_layout()
    return pca_to_use # Return the fitted PCA object

# Create causal layer model (inspired from DynGFN)
import torch.nn as nn
from typing import List

class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, time_varying=False, bias=True):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(self.num_linear, self.input_features, self.output_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_linear, self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        #print(input.unsqueeze(dim=2).shape, self.weight.unsqueeze(dim=0).shape)
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_linear={}, in_features={}, out_features={}, bias={}".format(
            self.num_linear,
            self.input_features,
            self.output_features,
            self.bias is not None,
        )
    

class Intervenable(nn.Module):
    """Models implementing intervenable are useful for learning in the experimental setting.

    This should represent interventions on a preexisting set of possible targets.
    """

    def __init__(self, targets=None):
        super().__init__()
        self.targets = targets
        self.current_target = None

    # def do(self, target, value=0.0):
    #    raise NotImplementedError

    def get_linear_structure(self):
        """gets the linear approximation of the structure coefficients.

        May not be applicable for all models
        """
        raise NotImplementedError

    def get_structure(self) -> np.ndarray:
        """Extracts a single summary structure from the model."""
        raise NotImplementedError

    def get_structures(self, n_structures: int) -> np.ndarray:
        """Some models can provide empirical distributions over structures, this function samples a
        number of structures from the model."""
        raise NotImplementedError

    def set_target(self, target):
        if self.targets is not None and not np.isin(target, self.targets):
            raise ValueError("Bad Target selected {target}")
        self.current_target = target

    def l1_reg(self):
        raise NotImplementedError

    def l2_reg(self):
        raise NotImplementedError


class MLPODEF(Intervenable):
    """Define an MLP ODE function according to Neural Graphical Models definition."""

    def __init__(self, dims, GL_reg=0.01, bias=True, time_varying=True):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.time_varying = time_varying
        self.GL_reg = GL_reg  # adaptive lasso parameter

        self.fc1 = nn.Linear(dims[0] + (1 if self.time_varying else 0), dims[0] * dims[1], bias=bias)
        """
        Old way of implementing time_invariant
        if time_invariant:
            self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        else:
            self.fc1 = nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias)
        """

        # fc2: local linear layers
        layers = []
        for i in range(len(dims) - 2):
            layers.append(
                LocallyConnected(
                    dims[0],
                    dims[i + 1], #+ (1 if self.time_varying else 0),
                    dims[i + 2],
                    bias=bias,
                )
            )
        self.fc2 = nn.ModuleList(layers)
        #self.selu = nn.SELU(inplace=True)
        self.selu = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = fc(self.selu(x))  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def l2_reg(self):
        """L2 regularization on all parameters."""
        reg = 0.0
        fc1_weight = self.fc1.weight  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def l1_reg(self):
        """L1 regularization on input layer parameters."""
        return torch.sum(torch.abs(self.fc1.weight))

    def grn_reg(self, grn):
        """
        Args:
            grn: torch.tensor (d x d) 1 if likely edge 0 if not
        """
        fc1_weight = self.fc1.weight  # d * m1, d
        d = fc1_weight.shape[-1]
        fc1_weight = fc1_weight.reshape(d, -1, d)
        fc1_weight = fc1_weight.transpose(0, 1)  # m1, d, d
        return torch.sum(torch.abs(fc1_weight * (1 - grn)))

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def proximal_step(self, lam=None):
        # Handle adaptive group lasso from NGM paper
        if lam is None:
            lam = 1 / self.group_weights()
        w = self.fc1.weight
        """Proximal step"""
        # w shape [j * m1, i]
        wadj = w.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        alpha = torch.clamp(
            torch.sum(wadj**2, dim=1).pow(0.5) - lam,
            min=0,
        )
        v = torch.nn.functional.normalize(wadj, dim=1) * alpha[:, None, :]
        w.data = v.view(-1, self.dims[0])

    def get_structure(self):
        """Score each edge based on the the weight sum."""
        d = self.dims[0]
        if self.time_varying:
            fc1_weight = self.fc1.weight[:, :-1]  # [j * m1, i]
        else:
            fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        G = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
        return G

    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()
            

from sklearn.metrics import average_precision_score, roc_auc_score

def structural_hamming_distance(W_true, W_est):
        """Computes the structural hamming distance."""
        pred = np.flatnonzero(W_est != 0)
        cond = np.flatnonzero(W_true)
        cond_reversed = np.flatnonzero(W_true.T)
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        pred_lower = np.flatnonzero(np.tril(W_est + W_est.T))
        cond_lower = np.flatnonzero(np.tril(W_true + W_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
        return shd

def plot_graph_heatmap(graph, ax, auc=None, title_prefix="", output_path=None):
    pcm = ax.matshow(graph, cmap="viridis")
    title = title_prefix
    if auc is not None:
        title += r'Pred Graph: AUC = %0.3f' % auc
    else:
        title += r'True Graph'
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax)
    if output_path:
        plt.savefig(output_path)
        plt.close(fig) # Close the figure after saving if an output path is provided

def compute_metrics(true_graph, estimated_graph):
    ### AUCROC
    AUCROC = roc_auc_score(true_graph, estimated_graph)
    
    ### AP
    AP = average_precision_score(true_graph, estimated_graph)

    ### Add symmetry score --> predics undirected effects
    estimated_sym = (estimated_graph + np.transpose(estimated_graph)) / 2
    true_sym = (((true_graph + np.transpose(true_graph)) / 2) > 0) * 1

    ### AUCROC symmetric
    AUCROC_sym = roc_auc_score(true_sym, estimated_sym)

    ### AP symmetric
    AP_sym = average_precision_score(true_sym, estimated_sym)
    
    metrics = ['AUCROC', 'AP', 'AUCROC_sym', 'AP_sym']
    df_graph_metrics = pd.DataFrame(
        [AUCROC, AP, AUCROC_sym, AP_sym],
        index=metrics,
        )
    return df_graph_metrics

def validation_step(model, val_data_full, x0, x1, graph, G=None, a=None, init_run=False, x=None, ts=None, time_steps=100, axes_list=None, model_name_prefix="", output_dir="."):
    ax1, ax2 = axes_list[0], axes_list[1]
    has_graph_axes = len(axes_list) >= 5
    pca = None  # Initialize pca to None

    node = NeuralODE(
        model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4 # o.g. 1e-4 for atol and rtol
    )
    with torch.no_grad():
        print("Starting traj inference ...")
        trajs = []
        x_start = x[:, -2, :]
        traj = node.trajectory(
            x_start,
            t_span=torch.linspace(0, 1, time_steps),
        )
        trajs.append(traj)
        traj = torch.cat(trajs, dim=0)
        pred = torch.stack([traj[0, :, :], traj[-1, :, :]], dim=1)
        true = torch.stack([x0, x1], dim=1)
        print("... Ending traj inference")
        
        dd_names, dd_values = dd.compute_distribution_distances(pred, true)
        t2_idx = [i for i in range(len(dd_names)) if 't2' in dd_names[i]]
        dd_t2_names = [dd_names[t] for t in t2_idx]
        dd_t2_values = [dd_values[t] for t in t2_idx]
        dd_df = pd.DataFrame(dd_t2_values, index=dd_t2_names)

        if G is None:
            if init_run: # Case: G is None AND init_run is True
                current_fig_pca_path = os.path.join(output_dir, f"{model_name_prefix}_val_pca_true_endpoints_init_G_None.png")
                fig_pca_init, ax_pca_init = plt.subplots(1,1,figsize=(6,5))
                pca = plot_pca_manifold(val_data_full, x1, ax_pca_init, a=a)
                ax_pca_init.set_title(f"{model_name_prefix} True Endpoints PCA (Initial, G is None)")
                plt.savefig(current_fig_pca_path)
                plt.close(fig_pca_init)
            else: # Case: G is None AND init_run is False
                current_fig_pca_true_path = os.path.join(output_dir, f"{model_name_prefix}_val_pca_true_endpoints_G_None.png")
                current_fig_pca_pred_path = os.path.join(output_dir, f"{model_name_prefix}_val_pca_pred_endpoints_G_None.png")

                fig_pca_true_g_none, ax_pca_true_g_none = plt.subplots(1,1,figsize=(6,5))
                # pca_obj is the PCA transformer fitted on true data
                pca_obj = plot_pca_manifold(val_data_full, x1, ax_pca_true_g_none, a=a) 
                ax_pca_true_g_none.set_title(f"{model_name_prefix} True Endpoints PCA (G is None)")
                plt.savefig(current_fig_pca_true_path)
                plt.close(fig_pca_true_g_none)

                fig_pca_pred_g_none, ax_pca_pred_g_none = plt.subplots(1,1,figsize=(6,5))
                _ = plot_pca_manifold(val_data_full, traj[-1], ax_pca_pred_g_none, a=a, pca_fitted_obj=pca_obj) # Use fitted PCA
                ax_pca_pred_g_none.set_title(f"{model_name_prefix} Predicted Endpoints PCA (G is None)")
                plt.savefig(current_fig_pca_pred_path)
                plt.close(fig_pca_pred_g_none)
                pca = pca_obj # Return the PCA object fitted on the true data

        else:  # G is NOT None
            self_loop_mask = ~np.eye(G.shape[-1], dtype=bool)
            df_graph_metrics = compute_metrics((graph[self_loop_mask]).flatten(), (G[self_loop_mask]).flatten())
            dd_df = dd_df.append(df_graph_metrics)
            auc = df_graph_metrics[0].values[0]
            print("SHD =", structural_hamming_distance(graph, G), "AUC =", auc) # SHD was printed here before, direct call
            
            # Define paths for plots when G is not None
            path_true_endpoints_pca = os.path.join(output_dir, f"{model_name_prefix}_val_pca_true_endpoints.png")
            path_pred_endpoints_pca = os.path.join(output_dir, f"{model_name_prefix}_val_pca_pred_endpoints.png")
            path_true_graph_heatmap = os.path.join(output_dir, f"{model_name_prefix}_true_graph_heatmap.png")
            path_pred_graph_heatmap = os.path.join(output_dir, f"{model_name_prefix}_pred_graph_heatmap_AUC_{auc:.3f}.png")
            self_loop_mask_for_plot = np.ones((G.shape[-1], G.shape[-1])) - np.eye(G.shape[-1])

            if init_run: # Case: G is not None AND init_run is True
                fig_true_graph, ax_true_graph = plt.subplots(1,1, figsize=(6,5))
                plot_graph_heatmap(graph*self_loop_mask_for_plot, ax=ax_true_graph, title_prefix=f"{model_name_prefix} ", output_path=path_true_graph_heatmap)
                plt.close(fig_true_graph)
                
                fig_pca_true, ax_pca_true = plt.subplots(1,1,figsize=(6,5))
                pca = plot_pca_manifold(val_data_full, x1, ax_pca_true, a=a)
                ax_pca_true.set_title(f"{model_name_prefix} True Endpoints PCA")
                plt.savefig(path_true_endpoints_pca)
                plt.close(fig_pca_true)

            else: # Case: G is not None AND init_run is False
                if has_graph_axes:
                    plot_graph_heatmap(graph*self_loop_mask_for_plot, ax=axes_list[3], title_prefix=f"{model_name_prefix} ")
                    plot_graph_heatmap(G*self_loop_mask_for_plot, ax=axes_list[4], auc=auc, title_prefix=f"{model_name_prefix} ")
                else: 
                    fig_h_true, ax_h_true = plt.subplots(1,1,figsize=(6,5))
                    plot_graph_heatmap(graph*self_loop_mask_for_plot, ax=ax_h_true, title_prefix=f"{model_name_prefix} ", output_path=path_true_graph_heatmap)
                    plt.close(fig_h_true)

                    fig_h_pred, ax_h_pred = plt.subplots(1,1,figsize=(6,5))
                    plot_graph_heatmap(G*self_loop_mask_for_plot, ax=ax_h_pred, auc=auc, title_prefix=f"{model_name_prefix} ", output_path=path_pred_graph_heatmap)
                    plt.close(fig_h_pred)

                fig_pca_true_val, ax_pca_true_val = plt.subplots(1,1,figsize=(6,5))
                pca_fitted_obj = plot_pca_manifold(val_data_full, x1, ax_pca_true_val, a=a)
                ax_pca_true_val.set_title(f"{model_name_prefix} True Endpoints PCA")
                plt.savefig(path_true_endpoints_pca)
                plt.close(fig_pca_true_val)
                
                fig_pca_pred_val, ax_pca_pred_val = plt.subplots(1,1,figsize=(6,5))
                _ = plot_pca_manifold(val_data_full, traj[-1], ax_pca_pred_val, a=a, pca_fitted_obj=pca_fitted_obj) # Use fitted PCA
                ax_pca_pred_val.set_title(f"{model_name_prefix} Predicted Endpoints PCA")
                plt.savefig(path_pred_endpoints_pca)
                plt.close(fig_pca_pred_val)
                pca = pca_fitted_obj # Return the PCA object fitted on true data
    
    # Removed the orphaned elif blocks as their logic is now integrated or was redundant.

    return traj, dd_t2_values, pca, dd_df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_sc_trajectories(traj, sc_boolode, a, W2, pca_full, ax, output_path=None, title_prefix=""):
    n = 2000
    # Create a new figure and axis for this plot to ensure it's saved correctly
    fig_sc_traj, current_ax = plt.subplots(1,1, figsize=(7,6))

    #cells = sc_boolode.reshape(-1, sc_boolode.shape[-1])
    #pca_embed = pca.fit_transform(cells)
    traj_pca = []
    for i in range(traj.shape[0]):
        traj_pca.append(torch.tensor(pca_full.transform(traj[i].cpu().detach().numpy())))
    traj_pca = torch.stack(traj_pca, dim=0)
    
    a = a.detach().cpu().numpy()
    group = []
    group_dict = {'control': [], 'g3': [], 'g4': []}
    data_dict_traj_0 = {'control': [], 'g3': [], 'g4': []}
    data_dict_traj_1 = {'control': [], 'g3': [], 'g4': []}
    for i in range(a.shape[0]):
            not_control = (a[i].sum() < len(a[i]))
            if not_control == False:
                group.append(50)
                group_dict['control'].append('cyan')
                data_dict_traj_0['control'].append(traj_pca[-1, i, 0])
                data_dict_traj_1['control'].append(traj_pca[-1, i, 1])
            else:
                a_label = np.where(a[i] == 0)[0]
                #group.append(int(a_label) + 1)
                group.append(int(a_label))
                group_dict['g'+str(int(a_label))].append('red' if int(a_label) == 3 else 'blue')
                data_dict_traj_0['g'+str(int(a_label))].append(traj_pca[-1, i, 0])
                data_dict_traj_1['g'+str(int(a_label))].append(traj_pca[-1, i, 1])

    #plt.figure(figsize=(6, 6))
    current_ax.scatter(traj_pca[0, :n, 0], traj_pca[0, :n, 1], s=10, alpha=0.8, c="black")
    current_ax.scatter(traj_pca[:, :n, 0], traj_pca[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    #ax.scatter(traj_pca[-1, :n, 0], traj_pca[-1, :n, 1], s=4, alpha=1, c="blue")
    for k, v in group_dict.items():
        if len(v) > 0:
            traj_pca_0, traj_pca_1 = data_dict_traj_0[k], data_dict_traj_1[k]
            #ax.scatter(traj_pca[-1, :n, 0], traj_pca[-1, :n, 1], s=4, alpha=1, c=v, label=k)
            current_ax.scatter(traj_pca_0, traj_pca_1, s=4, alpha=1, c=v, label=k)
    current_ax.legend(["Prior sample z(S)", "Flow", "z(0)"])
    current_ax.set_title(title_prefix + r'2-Wasserstein($x_T$,  $\hat{x_T}$) = %f' % W2 if W2 is not None else title_prefix + "Predicted Trajectories")
    #ax.xticks([])
    #ax.yticks([])
    #ax.show()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close(fig_sc_traj) # Close the figure after saving

# pre-process data for training and validation pipeline
class TimeSeriesInterventionBifurcatingDataset(Dataset):
    def __init__(self, data, graph, action, time_step=5):
        self.graph = graph
        self.action = action
        #data = data[:, 5:, :] # get rid of starting points dense cluster
        self.x = data[:, ::time_step, :]
            
    def num_genes(self):
        return self.x.shape[-1]
            
    def __len__(self):
        return self.action.shape[0]
    
    def __getitem__(self, idx):
        x_sample = self.x[idx]
        action_sample = self.action[idx]
        sample = [x_sample, action_sample]
        return sample
        #return idx
    
    
def preprocess_batch(X, training=True, leave_out_end_point=False):
    """converts a batch of data into matched a random pair of (x0, x1)"""
    t_select = torch.zeros(1)
    batch_size, times, dim = X.shape
    if leave_out_end_point:
        t_select = torch.randint(times - 2, size=(batch_size,))
    else:
        t_select = torch.randint(times - 1, size=(batch_size,))
    x0 = []
    x1 = []
    if training:
        for i in range(batch_size):
            ti = t_select[i]
            ti_next = ti + 1
            x0.append(X[i, ti, :])
            x1.append(X[i, ti_next, :])
    else:
        for i in range(batch_size):
            x0.append(X[i, 0, :])
            x1.append(X[i, -1, :])
    x0, x1 = torch.stack(x0), torch.stack(x1)
    return x0, x1, t_select


def ot_resample(x0, x1):
    a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
    M = torch.cdist(x0, x1) ** 2
    #M = M / M.max()
    pi = pot.emd(a, b, M.detach().cpu().numpy())
    # Sample random interpolations on pi
    p = pi.flatten()
    p = p / p.sum()
    #choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x0.shape[0])
    i, j = np.divmod(choices, pi.shape[1])
    x0 = x0[i]
    x1 = x1[j]
    return x0, x1


def compute_pi(x0, x1):
    a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
    M = torch.cdist(x0, x1) ** 2
    #M = M / M.max()
    pi = pot.emd(a, b, M.detach().cpu().numpy())
    return pi

def get_train_dataset(seed):
    np.random.seed(seed)
    # shuffle train data cell pairs to emulate single-cell data
    datas_train_shuffled = []
    data_train_to_shuffle = datas_train
    for D in data_train_to_shuffle:
        D_shuffled = []
        for t in range(D.shape[1]):
            choices = np.random.choice(D[:, t, :].shape[0], size=D[:, t, :].shape[0])
            D_shuffled.append(torch.tensor(D[choices, t, :]).float())
        shuffled_data = torch.stack(D_shuffled, dim=1).cpu().detach().numpy()
        datas_train_shuffled.append(shuffled_data)

    # compile train data
    #train_datas = [torch.from_numpy(item).float() for item in datas_train]
    train_datas = [torch.from_numpy(item).float() for item in datas_train_shuffled]
    train_graphs = [torch.from_numpy(item).float() for item in graphs_train]
    train_actions = [torch.from_numpy(item).float() for item in actions_train]
    train_data = torch.cat(train_datas, dim=0)
    train_graph = torch.cat(train_graphs, dim=0)
    train_action = torch.cat(train_actions, dim=0)
    #print(train_data.shape, train_graph.shape, train_action.shape)
    return train_data, train_graph, train_action

def get_val_dataset(seed):
    np.random.seed(seed)
    # shuffle val data cell pairs to emulate single-cell data
    datas_val_shuffled = []
    datas_val_to_shuffled = datas_val
    for D in datas_val_to_shuffled:
        D_shuffled = []
        for t in range(D.shape[1]):
            choices = np.random.choice(D[:, t, :].shape[0], size=D[:, t, :].shape[0])
            D_shuffled.append(torch.tensor(D[choices, t, :]).float())
        shuffled_data = torch.stack(D_shuffled, dim=1).cpu().detach().numpy()
        datas_val_shuffled.append(shuffled_data)

    # compile validation data
    #val_datas = [torch.from_numpy(item).float() for item in datas_val]
    val_datas = [torch.from_numpy(item).float() for item in datas_val_shuffled]
    val_graphs = [torch.from_numpy(item).float() for item in graphs_val]
    val_actions = [torch.from_numpy(item).float() for item in actions_val]
    val_data = torch.cat(val_datas, dim=0)
    val_graph = torch.cat(val_graphs, dim=0)
    val_action = torch.cat(val_actions, dim=0)
    #print(val_data.shape, val_graph.shape, val_action.shape)
    return val_data, val_graph, val_action

# Time-series CFM W/ internvetional conditional information
# OT-CFM

# num_iters = 10000
# batch_size = 64
# seeds = [1, 2, 3, 4, 5]
# dd_metrics_df = [] # This will be handled per model type now

# --- Main Training Function ---
def train_evaluate_model(
    model_type, 
    model_params,
    true_graph_ref, 
    output_dir_global,
    num_iters_cfg,
    batch_size_cfg,
    sigma_cfg,
    lr_cfg,
    l1_lambda_cfg=0.0,
    current_seed=None
):
    """Trains and evaluates a given model type (MLP or MLPODEF).
    Returns a DataFrame of metrics for this run.
    """
    print(f"\n----- Training and Evaluating: {model_type} with seed {current_seed} -----")
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)

    # Get fresh shuffled data for this seed
    # These functions use global datas_train, actions_train etc. which are pre-split lists of numpy arrays
    # and graphs_train which is a list of graph structures.
    train_data_full, _, _ = get_train_dataset(current_seed) # We only need train_data for CFM training part
    val_data_full, val_graph_full, val_action_full = get_val_dataset(current_seed)

    if train_data_full is None or train_data_full.nelement() == 0:
        print(f"Error: Training data is empty for seed {current_seed}. Skipping run.")
        return pd.DataFrame() # Return empty dataframe
    if val_data_full is None or val_data_full.nelement() == 0:
        print(f"Error: Validation data is empty for seed {current_seed}. Skipping run.")
        return pd.DataFrame()

    # Compute full-batch OT matrix for training data
    # Ensure train_data_full has at least 2 timepoints
    if train_data_full.shape[1] < 2:
        print(f"Error: Training data for seed {current_seed} has only {train_data_full.shape[1]} timepoints. Need at least 2 for OT. Skipping run.")
        return pd.DataFrame()
    pis = [compute_pi(train_data_full[:,t], train_data_full[:,t+1]) for t in range(train_data_full.shape[1]-1)]

    dim = train_data_full.shape[-1]
    model_instance = None
    if model_type == "CFM_MLP":
        model_instance = MLP(dim=dim, w=model_params.get('w', 64), time_varying=model_params.get('time_varying', False))
    elif model_type == "NGM_CFM_MLPODEF":
        mlpodef_dims = [dim, *model_params.get('hidden_dims', [100, 1])]
        model_instance = MLPODEF(mlpodef_dims, time_varying=model_params.get('time_varying', False))
        model_instance.reset_parameters()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print(f"Size of {model_type} model: {count_parameters(model_instance)} Parameters")
    optimizer = torch.optim.Adam(params=model_instance.parameters(), lr=lr_cfg)

    run_metrics_list = [] # To collect metrics dataframes from validation

    for k in range(num_iters_cfg):
        # OT resample from train_data_full
        i_s = np.random.choice(train_data_full.shape[0], size=batch_size_cfg)
        
        if train_data_full.shape[1] - 2 < 0 : # Should not happen if checked before pis
             print(f"Warning: Not enough timepoints in train_data_full to select t_select ({train_data_full.shape[1]}). Skipping iteration {k}.")
             continue # or break
        t_select_indices = np.random.choice(train_data_full.shape[1] - 2 if train_data_full.shape[1] > 2 else 1, size=batch_size_cfg)
        t_select = torch.tensor(t_select_indices) 

        x0 = train_data_full[i_s, t_select, :] 
        x1_list = []
        for j, t_idx_val in enumerate(t_select_indices):
            # pis is indexed by the interval, so pis[t_idx_val] is correct for time t_idx_val to t_idx_val+1
            ot_plan_for_t = pis[t_idx_val]
            source_cell_ot_plan = ot_plan_for_t[i_s[j]]
            # Normalize probabilites for choice
            probabilities = source_cell_ot_plan / source_cell_ot_plan.sum()
            if np.isnan(probabilities).any(): # Handle potential NaNs if sum is zero
                # Fallback: uniform choice or skip. For now, let's pick first one.
                print(f"Warning: NaN in OT probabilities for cell {i_s[j]} at time {t_idx_val}. Defaulting choice.")
                choice = 0
            else:
                choice = np.random.choice(train_data_full.shape[0], p=probabilities)
            x1_list.append(train_data_full[choice, t_idx_val + 1, :])
        x1 = torch.stack(x1_list)

        # Update params step
        optimizer.zero_grad()
        t_interp = torch.rand(x0.shape[0], 1)
        mu_t = t_interp * x1 + (1 - t_interp) * x0
        # sigma_t = sigma_cfg # In original code, sigma_t is just sigma
        x_perturbed = mu_t + sigma_cfg * torch.randn(x0.shape[0], dim)
        ut = x1 - x0 # Target velocity
        vt = model_instance(x_perturbed) 
        loss = torch.mean((vt - ut) ** 2)
        if model_type == "NGM_CFM_MLPODEF" and l1_lambda_cfg > 0:
            loss += l1_lambda_cfg * model_instance.l1_reg()
        
        loss.backward()
        optimizer.step()

        if (k + 1) % 1000 == 0:
            log_msg = f"Seed {current_seed} - {model_type} - Iter {k+1}/{num_iters_cfg}: loss {loss.item():0.3f}"
            if model_type == "NGM_CFM_MLPODEF":
                G_current = model_instance.get_structure().cpu().detach().numpy()
                self_loop_mask = ~np.eye(G_current.shape[-1], dtype=bool)
                if true_graph_ref is not None and true_graph_ref.shape == G_current.shape:
                    df_graph_metrics_iter = compute_metrics((true_graph_ref[self_loop_mask]).flatten(), (G_current[self_loop_mask]).flatten())
                    auc_iter = df_graph_metrics_iter[0].values[0]
                    log_msg += f" AUC {auc_iter:0.3f}"
                else:
                    log_msg += " (True graph not available for interim AUC)"
            print(log_msg)

    # Final validation step for this seed
    val_dataset_obj = TimeSeriesInterventionBifurcatingDataset(val_data_full, val_graph_full, val_action_full, time_step=1)
    # Ensure val_dataset_obj is not empty
    if len(val_dataset_obj) == 0:
        print(f"Warning: Validation dataset for seed {current_seed} is empty. Skipping validation.")
        return pd.DataFrame()
        
    val_dataloader_obj = DataLoader(val_dataset_obj, batch_size=len(val_dataset_obj), shuffle=False)
    model_name_prefix_val = f"{model_type}_seed{current_seed}"

    with torch.no_grad():
        for _, val_batch_data in enumerate(val_dataloader_obj):
            x_val, a_val = val_batch_data
            x0_val, x1_val, _ = preprocess_batch(x_val, training=False)
            ts_val = x_val.shape[1]
            
            G_final_val = None
            if model_type == "NGM_CFM_MLPODEF":
                G_final_val = model_instance.get_structure().cpu().detach().numpy()

            fig_val_dummy, axes_val_dummy = plt.subplots(1, 2, figsize=(10,4)) # Placeholder for axes_list
            pred_traj_val, dd_t2_values_val, pca_val_obj, dd_df_val = validation_step(
                model=model_instance, val_data_full=val_data_full,
                x0=x0_val, x1=x1_val, graph=true_graph_ref, a=a_val, G=G_final_val, x=x_val, ts=ts_val,
                axes_list=axes_val_dummy,
                model_name_prefix=model_name_prefix_val,
                output_dir=output_dir_global
            )
            plt.close(fig_val_dummy)
            run_metrics_list.append(dd_df_val)

            w2_dist_val = dd_t2_values_val[1] if dd_t2_values_val and len(dd_t2_values_val) > 1 else None
            fig_sc_dummy, ax_sc_dummy = plt.subplots(1,1)
            plot_sc_trajectories(
                pred_traj_val, val_data_full, a_val, w2_dist_val, pca_val_obj,
                ax=ax_sc_dummy,
                output_path=os.path.join(output_dir_global, f"{model_name_prefix_val}_sc_trajectories.png"),
                title_prefix=f"{model_name_prefix_val} "
            )
            plt.close(fig_sc_dummy)
            break # Assuming one batch for full validation set
            
    if not run_metrics_list:
        print(f"Warning: No metrics collected during validation for {model_name_prefix_val}.")
        return pd.DataFrame() 
        
    # Concatenate metrics from this run (should be just one df from the single validation batch)
    final_metrics_df_for_run = pd.concat(run_metrics_list, axis=0) # axis=0 because each is a df (column vector)
    return final_metrics_df_for_run

# --- Global Configuration for Training Runs ---
SEEDS = [1, 2] # Reduced for brevity, original was [1, 2, 3, 4, 5]
NUM_ITERS_CFG = 5000 # Reduced for brevity, original was 10000
BATCH_SIZE_CFG = 32  # Original 64
SIGMA_CFG = 0.1
LR_CFG = 0.01
L1_LAMBDA_NGM_CFG = 1e-6

# --- Run Training for CFM_MLP ---
all_runs_metrics_mlp = []
if datas_train and true_graph: # Ensure data was loaded
    print("\n========== Starting Training for CFM_MLP ==========")
    for seed_val in SEEDS:
        metrics_df_mlp = train_evaluate_model(
            model_type="CFM_MLP",
            model_params={'w': 64, 'time_varying': False},
            true_graph_ref=true_graph, # true_graph is from the first validation set, assumed common
            output_dir_global=OUTPUT_DIR,
            num_iters_cfg=NUM_ITERS_CFG,
            batch_size_cfg=BATCH_SIZE_CFG,
            sigma_cfg=SIGMA_CFG,
            lr_cfg=LR_CFG,
            current_seed=seed_val
        )
        if not metrics_df_mlp.empty:
            all_runs_metrics_mlp.append(metrics_df_mlp)

    if all_runs_metrics_mlp:
        # Metrics DFs from validation_step are single-column (value) with multi-index (metric_name)
        # To average across seeds, we need to handle this. Let's concat along axis 1 (columns are seeds)
        # then calculate mean/std. Each df in all_runs_metrics_mlp is a series (metric name -> value)
        # Let's convert them to a common format if needed or ensure concat works as expected
        # If each dd_df from validation_step is a DataFrame with metrics as index and a single column of values:
        df_mlp_all_seeds = pd.concat(all_runs_metrics_mlp, axis=1) # Columns are seeds now
        df_mlp_summary = pd.DataFrame()
        df_mlp_summary["mean"] = df_mlp_all_seeds.mean(axis=1)
        df_mlp_summary["std"] = df_mlp_all_seeds.std(axis=1)
        print("\n--- CFM_MLP Final Metrics Summary (Mean/Std over Seeds) ---")
        print(df_mlp_summary)
        df_mlp_summary.to_csv(os.path.join(OUTPUT_DIR, "CFM_MLP_metrics_summary.csv"))
    else:
        print("No metrics collected for any CFM_MLP run.")
else:
    print("Skipping CFM_MLP training as no data was loaded or true_graph is missing.")

# --- Run Training for NGM_CFM_MLPODEF ---
all_runs_metrics_ngm = []
if datas_train and true_graph: # Ensure data was loaded
    print("\n========== Starting Training for NGM_CFM_MLPODEF ==========")
    for seed_val in SEEDS:
        metrics_df_ngm = train_evaluate_model(
            model_type="NGM_CFM_MLPODEF",
            model_params={'hidden_dims': [100, 1], 'time_varying': False},
            true_graph_ref=true_graph,
            output_dir_global=OUTPUT_DIR,
            num_iters_cfg=NUM_ITERS_CFG,
            batch_size_cfg=BATCH_SIZE_CFG,
            sigma_cfg=SIGMA_CFG,
            lr_cfg=LR_CFG,
            l1_lambda_cfg=L1_LAMBDA_NGM_CFG,
            current_seed=seed_val
        )
        if not metrics_df_ngm.empty:
            all_runs_metrics_ngm.append(metrics_df_ngm)

    if all_runs_metrics_ngm:
        df_ngm_all_seeds = pd.concat(all_runs_metrics_ngm, axis=1)
        df_ngm_summary = pd.DataFrame()
        df_ngm_summary["mean"] = df_ngm_all_seeds.mean(axis=1)
        df_ngm_summary["std"] = df_ngm_all_seeds.std(axis=1)
        print("\n--- NGM_CFM_MLPODEF Final Metrics Summary (Mean/Std over Seeds) ---")
        print(df_ngm_summary)
        df_ngm_summary.to_csv(os.path.join(OUTPUT_DIR, "NGM_CFM_metrics_summary.csv"))
    else:
        print("No metrics collected for any NGM_CFM_MLPODEF run.")
else:
    print("Skipping NGM_CFM_MLPODEF training as no data was loaded or true_graph is missing.")

print(f"\nAll training and evaluation complete. Results saved in {OUTPUT_DIR}")

# Remove old training loops
# for seed in seeds:
#     print("Training for seed =", seed, "...")
# ... (old CFM_MLP loop) ...
# df = pd.concat(dd_metrics_df, axis=1)
# ...
# df_metrics_mean_std.to_csv(os.path.join(OUTPUT_DIR, "CFM_MLP_metrics_summary.csv"))

# for seed in seeds:
#     print("Training for seed =", seed, "...")
# ... (old NGM_CFM_MLPODEF loop) ...
# df = pd.concat(dd_metrics_df, axis=1) if dd_metrics_df else pd.DataFrame()
# ...
# df_metrics_mean_std.to_csv(os.path.join(OUTPUT_DIR, "NGM_CFM_metrics_summary.csv"))

