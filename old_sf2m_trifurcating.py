import os
import sys

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
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torch.nn as nn
import torchdyn
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from torch.utils.data import Dataset, DataLoader

import src.models.components.distribution_distances as dd

def load_synthetic_data(data_dir, network_type=None, knockout=None):
    """
    Load synthetic data from the given structure.
    
    Args:
        data_dir: Root directory of the synthetic data
        network_type: Type of network (e.g., 'BF', 'TF', 'SW', etc.)
        knockout: Gene knockout identifier (e.g., 'g2', 'g3', etc.)
    
    Returns:
        data: Expression data in the shape (cells, timepoints, genes)
        graph: Network adjacency matrix
        gene_names: List of gene names
    """
    # Build the directory path based on parameters
    if knockout:
        dir_path = os.path.join(data_dir, f"dyn-{network_type}_ko_{knockout}")
    else:
        dir_path = os.path.join(data_dir, f"dyn-{network_type}")
    # Find the specific dataset path (might have numeric suffix)
    if os.path.exists(dir_path):
        dataset_dirs = []
        # Look for directories like dyn-BF-1000-1 
        for item in os.listdir(dir_path):
            full_path = os.path.join(dir_path, item)
            if os.path.isdir(full_path) and f"dyn-{network_type}" in item:
                dataset_dirs.append(full_path)
        
        # If we found specific dataset directories, use the first one
        if dataset_dirs:
            data_path = dataset_dirs[0]
        else:
            data_path = dir_path
    else:
        raise FileNotFoundError(f"Directory {dir_path} not found")
    
    # Load expression data
    expr_path = os.path.join(data_path, "ExpressionData.csv")
    expr_data = pd.read_csv(expr_path, index_col=0)
    
    # Load reference network
    ref_path = os.path.join(data_path, "refNetwork.csv")
    ref_net = pd.read_csv(ref_path)
    
    # Gene names are the index of expression data
    gene_names = expr_data.index.tolist()
    
    # Extract number of genes and cells
    n_genes = len(gene_names)
    n_cells = len(expr_data.columns)
    
    # Load pseudotime data to order cells
    pseudo_path = os.path.join(data_path, "PseudoTime.csv")
    if os.path.exists(pseudo_path):
        pseudo_time = pd.read_csv(pseudo_path, index_col=0)
        # Get the maximum pseudo-time value across any trajectory for each cell
        t_values = np.max(pseudo_time.values, axis=1)
        
        # Order cells by pseudo-time
        cell_order = np.argsort(t_values)
        sorted_expr = expr_data.iloc[:, cell_order]
        
        # Bin cells into a discrete number of timepoints
        n_timepoints = 5  # Number of timepoints to create
        bin_edges = np.linspace(t_values.min(), t_values.max(), n_timepoints+1)
        binned_cells = [[] for _ in range(n_timepoints)]
        
        for i, t in enumerate(t_values):
            bin_index = np.digitize(t, bin_edges) - 1
            # Cap at the last bin
            bin_index = min(bin_index, n_timepoints-1)
            binned_cells[bin_index].append(i)
        
        # Create data matrix (cells per timepoint, timepoints, genes)
        # First, find how many cells per timepoint to allocate
        cells_per_timepoint = max(len(bin) for bin in binned_cells if bin)
        if cells_per_timepoint == 0:
            cells_per_timepoint = 100  # Default if no cells in a bin
        
        # Initialize data array
        data = np.zeros((cells_per_timepoint, n_timepoints, n_genes))
        
        # Fill in data array
        for t, cell_indices in enumerate(binned_cells):
            if not cell_indices:
                # If no cells in this bin, use cells from previous bin or if first bin, use the first cells
                if t > 0 and binned_cells[t-1]:
                    cell_indices = binned_cells[t-1]
                else:
                    cell_indices = list(range(min(cells_per_timepoint, n_cells)))
            
            # Make sure we don't exceed cells_per_timepoint
            if len(cell_indices) > cells_per_timepoint:
                cell_indices = np.random.choice(cell_indices, cells_per_timepoint, replace=False)
            elif len(cell_indices) < cells_per_timepoint:
                # Pad with repeated cells if needed
                cell_indices = np.random.choice(cell_indices, cells_per_timepoint, replace=True)
            
            # Fill in the data for this timepoint
            for i, cell_idx in enumerate(cell_indices):
                data[i, t, :] = expr_data.iloc[:, cell_idx].values
    
    else:
        # If no pseudotime, treat as a single timepoint
        # Reshape to (cells, 1, genes)
        data = expr_data.T.values.reshape(n_cells, 1, n_genes)
    
    # Create adjacency matrix from reference network
    A = np.zeros((n_genes, n_genes))
    for _, row in ref_net.iterrows():
        # Extract gene names from the reference network
        source_gene = row[0]  # Gene1 
        target_gene = row[1]  # Gene2
        reg_type = row[2]  # Regulation type (+ or -)
        
        # Convert gene names to indices in our matrix
        source_idx = gene_names.index(source_gene) if source_gene in gene_names else -1
        target_idx = gene_names.index(target_gene) if target_gene in gene_names else -1
        
        if source_idx >= 0 and target_idx >= 0:
            rel = 1 if reg_type == "+" else -1
            A[target_idx, source_idx] = rel  # Note: target is regulated by source
    
    # Load cluster IDs if available (not used in the main flow but could be useful)
    cluster_info = None
    try:
        cluster_path = os.path.join(data_path, "ClusterIds.csv")
        if os.path.exists(cluster_path):
            cluster_info = pd.read_csv(cluster_path, index_col=0)
    except:
        pass
    
    return data, np.abs(A), gene_names  # Return absolute values for graph structure

# Load datasets from Synthetic directory
data_root = "data/Synthetic"
network_types = ['TF']  # Only use TF network
datas_train, datas_val = [], []
graphs_train, graphs_val = [], []
actions_train, actions_val = [], []
train_val_split = [0.8, 0.2]
gene_names_list = []  # Store the gene names for each network type

# First, load base networks without knockouts
for net_type in network_types:
    try:
        print(f"Loading base network {net_type}")
        data, graph, gene_names = load_synthetic_data(data_root, network_type=net_type)
        gene_names_list.append(gene_names)
        
        # If data has only one timepoint, expand it (handled in the dataset class)
        if data.shape[1] == 1:
            print(f"Network {net_type} has only one timepoint, will be expanded in the dataset.")
        
        train_len = int(data.shape[0]*train_val_split[0])
        val_len = data.shape[0] - train_len
        
        # No intervention for base networks
        action = np.ones((data.shape[0], data.shape[-1]))
        
        # Split into train/val
        datas_train.append(data[:train_len])
        datas_val.append(data[train_len:])
        graphs_train.append(graph)
        graphs_val.append(graph)
        actions_train.append(action[:train_len])
        actions_val.append(action[train_len:])
        
    except Exception as e:
        print(f"Error loading {net_type}: {e}")
        continue

# Next, load networks with knockouts
knockouts_to_try = []
# Only try knockouts g2 through g8 as specified
for i in range(2, 9):  # Try gene knockouts g2 through g8
    knockouts_to_try.append(f"g{i}")

# Skip trying network-specific gene names - we'll just use g2-g8 format
# for gene_names in gene_names_list:
#     if gene_names:
#         for gene in gene_names:
#             if gene not in knockouts_to_try:
#                 knockouts_to_try.append(gene)

for net_type in network_types:
    for knockout in knockouts_to_try:
        try:
            print(f"Loading knockout network {net_type} with {knockout}")
            data, graph, gene_names = load_synthetic_data(data_root, network_type=net_type, knockout=knockout)
            
            train_len = int(data.shape[0]*train_val_split[0])
            val_len = data.shape[0] - train_len
            
            # Create action matrix with knocked out gene set to 0
            action = np.ones((data.shape[0], data.shape[-1]))
            
            # Find the index of the knocked out gene
            if knockout.startswith('g') and knockout[1:].isdigit():
                # If knockout is in the format g1, g2, etc.
                ko_idx = int(knockout[1:]) - 1  # Convert g1 to index 0
                if ko_idx < data.shape[-1]:
                    action[:, ko_idx] = 0
            else:
                # If knockout is an actual gene name
                try:
                    ko_idx = gene_names.index(knockout)
                    action[:, ko_idx] = 0
                except ValueError:
                    print(f"Warning: Gene {knockout} not found in gene names list.")
                    continue
            
            # Split into train/val
            datas_train.append(data[:train_len])
            datas_val.append(data[train_len:])
            graphs_train.append(graph)
            graphs_val.append(graph)
            actions_train.append(action[:train_len])
            actions_val.append(action[train_len:])
            
        except Exception as e:
            # Skip if this network/knockout doesn't exist
            continue

print("Data loaded successfully")
if len(datas_train) > 0:
    print(f"Example data shape: {datas_train[0].shape}")
    true_graph = graphs_val[0]  # Save the first graph as the true graph for evaluation
else:
    print("WARNING: No data was loaded. Check that data/Synthetic directory exists and contains valid data.")
    true_graph = np.array([])  # Empty graph as fallback

import networkx as nx
fig, axes = plt.subplots(min(3, len(graphs_train)), 1, figsize=(4,16))
if len(graphs_train) == 1:
    axes = [axes]  # Make sure axes is iterable
else:
    axes = axes.flatten()

for i, (ax, data, graph) in enumerate(zip(axes, datas_train[:len(axes)], graphs_train[:len(axes)])):
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    nx.draw(G, ax=ax, pos=nx.circular_layout(G), arrows=True, with_labels=True)
plt.tight_layout()

from sklearn.decomposition import PCA
fig, axes = plt.subplots(1, min(3, len(datas_train)), figsize=(16,4))
if len(datas_train) == 1:
    axes = [axes]  # Make sure axes is iterable
else:
    axes = axes.flatten()

pca = PCA(n_components=2)
cells_null = datas_train[0].reshape(-1, datas_train[0].shape[-1])
pca_embed = pca.fit_transform(cells_null)
for i, (ax, data, graph) in enumerate(zip(axes, datas_train[:len(axes)], graphs_train[:len(axes)])):
    cells = data.reshape(-1, data.shape[-1])
    pca_embed = pca.transform(cells)
    labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten()
    scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True)
plt.tight_layout()

fig, axes = plt.subplots(1, min(3, len(datas_val)), figsize=(16,4))
if len(datas_val) == 1:
    axes = [axes]  # Make sure axes is iterable
else:
    axes = axes.flatten()

pca = PCA(n_components=2)
cells_null = datas_val[0].reshape(-1, datas_val[0].shape[-1])
pca_embed = pca.fit_transform(cells_null)
for i, (ax, data, graph) in enumerate(zip(axes, datas_val[:len(axes)], graphs_val[:len(axes)])):
    cells = data.reshape(-1, data.shape[-1])
    pca_embed = pca.transform(cells)
    labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten()
    scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True)
plt.tight_layout()

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        input_dim = dim + (1 if time_varying else 0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        # Ensure input is 2D [batch_size, features]
        if x.dim() == 3:
            x = x.squeeze(1)
        # At this point x should be [batch_size, features]
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
        # Ensure t has the right shape for broadcasting
        if t.dim() == 0:
            t = t.view(1)
        # Ensure x is 2D [batch_size, features]
        if x.dim() == 3:
            x = x.squeeze(1)
        # Ensure t matches batch size
        if t.dim() == 1:
            t = t.expand(x.shape[0])
        # Only concatenate time if model is time_varying
        if self.model.time_varying:
            x = torch.cat([x, t.unsqueeze(-1)], dim=1)
        return self.model(x)


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
    
def plot_pca_manifold(data, preds, ax, a=None):
    cells = data.reshape(-1, data.shape[-1])
    pca = PCA(n_components=2)
    pca_embed = pca.fit_transform(cells)
    pca_xT = pca.transform(preds)
    cells_end_points = np.stack([preds], axis=1)
    pca_cell_end_points = np.stack([pca_xT], axis=1)
    pca_cell_end_points = pca_cell_end_points.reshape(-1, pca_cell_end_points.shape[-1])
    labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten()
    scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True)
    if a is None:
        labels_end_points = np.repeat(np.arange(cells_end_points.shape[1])[None,:], cells_end_points.shape[0], axis=0).flatten()
        scprep.plot.scatter2d(pca_cell_end_points, c=labels_end_points, ax=ax, ticks=False, colorbar=True)
    else:
        a = a.detach().cpu().numpy()
        group = []
        group_dict = {'control': [], 'g3': [], 'g4': []}
        data_dict = {'control': [], 'g3': [], 'g4': []}
        for i in range(a.shape[0]):
            not_control = (a[i].sum() < len(a[i]))
            if not_control == False:
                group.append(50)
                group_dict['control'].append('cyan')
                data_dict['control'].append(pca_cell_end_points[i])
            else:
                a_label = np.where(a[i] == 0)[0]
                #group.append(int(a_label) + 1)
                group.append(int(a_label))
                group_dict['g'+str(int(a_label))].append('red' if int(a_label) == 3 else 'blue')
                data_dict['g'+str(int(a_label))].append(pca_cell_end_points[i])
        for k, v in group_dict.items():
            if len(v) > 0:
                #scprep.plot.scatter2d(pca_cell_end_points, c=v, label=k, ax=ax, ticks=False, colorbar=True)
                scprep.plot.scatter2d(np.array(data_dict[k]), c=v, label=k, ax=ax, ticks=False, colorbar=True)
    plt.tight_layout()
    return pca

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

def plot_graph_heatmap(graph, ax, auc=None):
    pcm = ax.matshow(graph, cmap="viridis")
    if auc is not None:
        #title = title + ''
        ax.set_title(r'Pred Graph: AUC = %0.3f' % auc)
    else:
        ax.set_title(r'True Graph')
    fig.colorbar(pcm, ax=ax)
    

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

def compute_w2(x, y):
    """Compute 2-Wasserstein distance between two sets of points."""
    import ot
    # Compute cost matrix
    M = ot.dist(x, y)
    # Uniform weights
    a = np.ones((x.shape[0],)) / x.shape[0]
    b = np.ones((y.shape[0],)) / y.shape[0]
    # Compute Wasserstein distance
    return np.sqrt(ot.emd2(a, b, M))

def compute_mmd(x, y, sigma=1.0):
    """Compute Maximum Mean Discrepancy between two sets of points."""
    from sklearn.metrics.pairwise import rbf_kernel
    # Compute kernel matrices
    K_xx = rbf_kernel(x, x, gamma=1.0/(2*sigma**2))
    K_yy = rbf_kernel(y, y, gamma=1.0/(2*sigma**2))
    K_xy = rbf_kernel(x, y, gamma=1.0/(2*sigma**2))
    # Compute MMD
    return np.mean(K_xx) + np.mean(K_yy) - 2*np.mean(K_xy)

def compute_kl(x, y, bins=50):
    """Compute KL divergence between two sets of points using histogram approximation."""
    from scipy.stats import entropy
    # Compute histograms
    hist_x, _ = np.histogramdd(x, bins=bins)
    hist_y, _ = np.histogramdd(y, bins=bins)
    # Normalize
    hist_x = hist_x / np.sum(hist_x)
    hist_y = hist_y / np.sum(hist_y)
    # Add small constant to avoid division by zero
    eps = 1e-10
    hist_x = hist_x + eps
    hist_y = hist_y + eps
    # Compute KL divergence
    return entropy(hist_x.flatten(), hist_y.flatten())

def validation_step(model, val_data_full, x0, x1, graph, G=None, a=None, init_run=False, x=None, ts=None, time_steps=100, axes_list=None):
    """
    Run validation and visualization of model predictions
    """
    print("\n=== Starting Validation Step ===")
    print(f"Initial shapes - x0: {x0.shape}, x1: {x1.shape}")
    if x is not None:
        print(f"x shape: {x.shape}")
    
    if axes_list is None or len(axes_list) < 2:
        print("Error: Not enough axes for plotting")
        return None, None, None, None
    
    # Ensure tensors are on the right device
    device = next(model.parameters()).device
    
    # Make sure datasets are tensors
    if not torch.is_tensor(x0):
        x0 = torch.tensor(x0, dtype=torch.float32).to(device)
    
    if not torch.is_tensor(x1):
        x1 = torch.tensor(x1, dtype=torch.float32).to(device)
    
    # Check if data exists
    if x0.shape[0] == 0 or x1.shape[0] == 0:
        print("Error: Empty data in x0 or x1")
        return None, None, None, None
    
    print(f"Device: {device}")
    print(f"x0 device: {x0.device}, x1 device: {x1.device}")
    
    if a is not None:
        print(f"Action matrix shape: {a.shape}")
        print(f"Number of interventions: {torch.sum(1.0 - a)}")
    
    # Create integration time steps
    t = torch.linspace(0, 1, time_steps+1)
    print(f"Time steps shape: {t.shape}")
    
    # Run Neural ODE to get learned trajectories for all samples
    try:
        print("\nRunning ODE integration...")
        traj = odeint(
            torch_wrapper(model),
            x0,
            t,
            method="dopri5",
            atol=1e-4,
            rtol=1e-4
        )
        traj = traj.cpu().detach().numpy()
        print(f"Trajectory shape after ODE: {traj.shape}")
    except Exception as e:
        print(f"Error during ODE integration: {e}")
        return None, None, None, None
    
    # Initialize metrics
    dd_metrics = {}
    dd_t2_values = []
    
    try:
        # Get real and predicted endpoints
        real_endpoints = x1.cpu().detach().numpy()
        pred_endpoints = traj[-1]
        print(f"\nEndpoints shapes - real: {real_endpoints.shape}, predicted: {pred_endpoints.shape}")
        
        # Compute distribution distances
        W2_val = compute_w2(real_endpoints, pred_endpoints)
        print(f"W2 distance: {W2_val}")
        dd_t2_values.extend([W2_val, W2_val])
        dd_metrics["W2"] = W2_val
        
    except Exception as e:
        print(f"Error computing distribution distances: {e}")
        dd_t2_values = [np.nan, np.nan]
        dd_metrics = {"W2": np.nan, "MMD": np.nan, "KL": np.nan}
    
    # PCA visualization of trajectories
    pca = None
    if x is not None and torch.is_tensor(x) and x.shape[0] > 0:
        try:
            print("\nPreparing PCA visualization...")
            # Reshape for PCA
            cells = x.reshape(-1, x.shape[-1]).cpu().detach().numpy()
            print(f"Cells shape for PCA: {cells.shape}")
            
            # Create and fit PCA
            pca = PCA(n_components=2)
            pca_embed = pca.fit_transform(cells)
            print(f"PCA embedding shape: {pca_embed.shape}")
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            
            # Plot manifold with ground truth trajectories
            if axes_list[1] is not None:
                plot_pca_manifold(pca_embed, [], axes_list[1], a)
                axes_list[1].set_title("Ground Truth Trajectories (PCA)")
            
        except Exception as e:
            print(f"Error in PCA visualization: {e}")
            print(f"cells array content: {cells}")
            print("WARNING: PCA computation failed, but continuing with pca object")
            # Don't set pca to None here, as we want to keep the fitted object
    
    # Create metrics DataFrame
    df_metrics = pd.DataFrame(dd_metrics, index=["validation"]).T
    print("\n=== Validation Step Complete ===")
    print(f"Returning PCA object: {'Success' if pca is not None else 'None'}")
    
    return traj, dd_t2_values, pca, df_metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_sc_trajectories(traj, sc_boolode, a, W2, pca_full, ax):
    print("\n=== Starting plot_sc_trajectories ===")
    print(f"Input shapes - traj: {traj.shape}, sc_boolode: {sc_boolode.shape}")
    if pca_full is None:
        print("Error: pca_full is None, cannot proceed with visualization")
        return
        
    n = 2000
    traj_pca = []
    print("\nStarting PCA transformation of trajectories...")
    for i in range(traj.shape[0]):
        print(f"Processing trajectory step {i}, shape: {traj[i].shape}")
        try:
            transformed = pca_full.transform(traj[i].cpu().detach().numpy())
            print(f"Transformed shape: {transformed.shape}")
            traj_pca.append(torch.tensor(transformed))
        except Exception as e:
            print(f"Error transforming trajectory step {i}: {e}")
            return
            
    print("Stacking transformed trajectories...")
    traj_pca = torch.stack(traj_pca, dim=0)
    print(f"Final traj_pca shape: {traj_pca.shape}")
    
    print("\nProcessing action matrix...")
    a = a.detach().cpu().numpy()
    print(f"Action matrix shape: {a.shape}")
    
    group = []
    group_dict = {'control': [], 'g3': [], 'g4': []}
    data_dict_traj_0 = {'control': [], 'g3': [], 'g4': []}
    data_dict_traj_1 = {'control': [], 'g3': [], 'g4': []}
    
    print("\nProcessing groups...")
    for i in range(a.shape[0]):
        not_control = (a[i].sum() < len(a[i]))
        if not_control == False:
            group.append(50)
            group_dict['control'].append('cyan')
            data_dict_traj_0['control'].append(traj_pca[-1, i, 0])
            data_dict_traj_1['control'].append(traj_pca[-1, i, 1])
        else:
            a_label = np.where(a[i] == 0)[0]
            group.append(int(a_label))
            group_dict['g'+str(int(a_label))].append('red' if int(a_label) == 3 else 'blue')
            data_dict_traj_0['g'+str(int(a_label))].append(traj_pca[-1, i, 0])
            data_dict_traj_1['g'+str(int(a_label))].append(traj_pca[-1, i, 1])
    
    print("\nPlotting trajectories...")
    ax.scatter(traj_pca[0, :n, 0], traj_pca[0, :n, 1], s=10, alpha=0.8, c="black")
    ax.scatter(traj_pca[:, :n, 0], traj_pca[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    
    print("\nPlotting groups...")
    for k, v in group_dict.items():
        if len(v) > 0:
            print(f"Plotting group {k} with {len(v)} points")
            traj_pca_0, traj_pca_1 = data_dict_traj_0[k], data_dict_traj_1[k]
            ax.scatter(traj_pca_0, traj_pca_1, s=4, alpha=1, c=v, label=k)
    
    ax.legend(["Prior sample z(S)", "Flow", "z(0)"])
    ax.set_title(r'2-Wasserstein($x_T$,  $\hat{x_T}$) = %f' % W2)
    plt.tight_layout()
    print("=== plot_sc_trajectories complete ===\n")

# pre-process data for training and validation pipeline
class TimeSeriesInterventionBifurcatingDataset(Dataset):
    def __init__(self, data, graph, action, time_step=1, trajectory_length=5):
        """
        Dataset for single-cell gene expression data with gene knockout interventions.
        
        Args:
            data: Shape (cells, timesteps, genes)
            graph: Adjacency matrix of gene regulatory network
            action: Intervention matrix (1 for active genes, 0 for knocked out genes)
            time_step: Interval between timepoints to use
            trajectory_length: Number of timepoints to create for a trajectory if data has only one timepoint
        """
        self.graph = graph
        self.action = action
        self.trajectory_length = trajectory_length
        
        # For the synthetic data, we may have only one timepoint
        # but we'll replicate it to create a "trajectory"
        if data.shape[1] == 1:
            # Create a synthetic trajectory by replicating the data
            # and adding small noise to create a trajectory effect
            num_steps = trajectory_length
            noise_scale = 0.05
            
            # Create initial replicated data
            expanded_data = np.repeat(data, num_steps, axis=1)
            expanded_data = expanded_data.reshape(data.shape[0], num_steps, data.shape[2])
            
            # Add progressive noise to create a trajectory effect
            for t in range(1, num_steps):
                # Add small increasing perturbations to create trajectory
                noise = np.random.normal(0, noise_scale * t, expanded_data[:, t, :].shape)
                expanded_data[:, t, :] += noise
                
                # Ensure values stay positive for gene expression
                expanded_data[:, t, :] = np.maximum(expanded_data[:, t, :], 0)
            
            self.x = expanded_data[:, ::time_step, :]
        else:
            # If we already have multiple timepoints, just use them with the specified time step
            self.x = data[:, ::time_step, :]
        
        # Convert to tensor for pytorch operations
        if not torch.is_tensor(self.x):
            self.x = torch.from_numpy(self.x).float()
        if not torch.is_tensor(self.action):
            self.action = torch.from_numpy(self.action).float()
        
    def num_genes(self):
        return self.x.shape[-1]
            
    def __len__(self):
        return self.action.shape[0]
    
    def __getitem__(self, idx):
        x_sample = self.x[idx]
        action_sample = self.action[idx]
        sample = [x_sample, action_sample]
        return sample
    
    
def preprocess_batch(X, training=True, leave_out_end_point=False):
    """
    Converts a batch of data into matched pairs of (x0, x1) for trajectory modeling.
    
    Args:
        X: Tensor of shape (batch_size, times, dim) - gene expression over time
        training: If True, randomly select timepoints; if False, use first and last timepoints
        leave_out_end_point: If True, don't use the last timepoint as destination in training
    
    Returns:
        x0: Starting points
        x1: Destination points
        t_select: Timepoints selected for x0
    """
    # Ensure X is a tensor
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
        
    batch_size, times, dim = X.shape
    
    # Initialize tensor of selected timepoints
    t_select = torch.zeros(batch_size, dtype=torch.long)
    
    if training:
        # In training mode, randomly select timepoints
        if leave_out_end_point:
            # Don't use the last timepoint as destination
            if times <= 2:
                # If we have only 2 or fewer timepoints, just use first and second
                t_select = torch.zeros(batch_size, dtype=torch.long)
            else:
                t_select = torch.randint(times - 2, size=(batch_size,), dtype=torch.long)
        else:
            # Can use any timepoint except the last one as starting point
            if times <= 1:
                # If we have only 1 timepoint, duplicate it
                t_select = torch.zeros(batch_size, dtype=torch.long)
            else:
                t_select = torch.randint(times - 1, size=(batch_size,), dtype=torch.long)
    else:
        # In validation/test mode, use first timepoint as starting point for all samples
        t_select = torch.zeros(batch_size, dtype=torch.long)
    
    # Create lists to hold starting and destination points
    x0 = []
    x1 = []
    
    if training:
        # Generate pairs for training by selecting random timepoints
        for i in range(batch_size):
            ti = t_select[i]
            ti_next = ti + 1
            x0.append(X[i, ti, :])
            x1.append(X[i, ti_next, :])
    else:
        # For validation/test, use the first and last timepoints
        for i in range(batch_size):
            x0.append(X[i, 0, :])
            x1.append(X[i, -1, :])
    
    # Stack the lists into tensors
    x0 = torch.stack(x0)
    x1 = torch.stack(x1)
    
    return x0, x1, t_select


def ot_resample(x0, x1):
    """
    Optimal transport-based resampling between beginning and ending points
    """
    a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
    M = torch.cdist(x0, x1) ** 2
    pi = pot.emd(a, b, M.detach().cpu().numpy())
    # Sample random interpolations on pi
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x0.shape[0])
    i, j = np.divmod(choices, pi.shape[1])
    x0 = x0[i]
    x1 = x1[j]
    return x0, x1


def compute_pi(x0, x1):
    """Compute optimal transport plan between two distributions"""
    a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
    M = torch.cdist(x0, x1) ** 2
    pi = pot.emd(a, b, M.detach().cpu().numpy())
    return pi

def get_train_dataset(seed):
    """
    Prepare training dataset with optional shuffling
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        train_data: Training data tensor
        train_graph: Training graph tensor
        train_action: Training action tensor with intervention information
    """
    np.random.seed(seed)
    
    # Check if we have data to work with
    if len(datas_train) == 0:
        print("No training data available.")
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # shuffle train data cell pairs to emulate single-cell data
    datas_train_shuffled = []
    
    for D in datas_train:
        # Convert to numpy if tensor
        if torch.is_tensor(D):
            D = D.cpu().detach().numpy()
            
        D_shuffled = []
        for t in range(D.shape[1]):
            choices = np.random.choice(D.shape[0], size=D.shape[0])
            D_shuffled.append(D[choices, t, :])
        
        # Stack the shuffled timepoints back together
        shuffled_data = np.stack(D_shuffled, axis=1)
        datas_train_shuffled.append(shuffled_data)

    # Compile train data - ensure all are tensors
    train_datas = []
    train_graphs = []
    train_actions = []
    
    for item in datas_train_shuffled:
        if not torch.is_tensor(item):
            train_datas.append(torch.tensor(item, dtype=torch.float32))
        else:
            train_datas.append(item)
            
    for item in graphs_train:
        if not torch.is_tensor(item):
            train_graphs.append(torch.tensor(item, dtype=torch.float32))
        else:
            train_graphs.append(item)
            
    for item in actions_train:
        if not torch.is_tensor(item):
            train_actions.append(torch.tensor(item, dtype=torch.float32))
        else:
            train_actions.append(item)
    
    # Concatenate the data
    if len(train_datas) > 0:
        train_data = torch.cat(train_datas, dim=0)
        
        # For the graph, we just need one copy per sample
        # So we repeat the graph for each sample from the same dataset
        counts = [d.shape[0] for d in train_datas]
        expanded_graphs = []
        
        for g, count in zip(train_graphs, counts):
            # For each graph, repeat it for the number of samples
            expanded_graphs.extend([g] * count)
            
        train_graph = torch.stack(expanded_graphs, dim=0)
        train_action = torch.cat(train_actions, dim=0)
        
        return train_data, train_graph, train_action
    else:
        # Return empty tensors if no data is available
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

def get_val_dataset(seed):
    """
    Prepare validation dataset with optional shuffling
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        val_data: Validation data tensor
        val_graph: Validation graph tensor
        val_action: Validation action tensor with intervention information
    """
    np.random.seed(seed)
    
    # Check if we have data to work with
    if len(datas_val) == 0:
        print("No validation data available.")
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # shuffle val data cell pairs to emulate single-cell data
    datas_val_shuffled = []
    
    for D in datas_val:
        # Convert to numpy if tensor
        if torch.is_tensor(D):
            D = D.cpu().detach().numpy()
            
        D_shuffled = []
        for t in range(D.shape[1]):
            choices = np.random.choice(D.shape[0], size=D.shape[0])
            D_shuffled.append(D[choices, t, :])
        
        # Stack the shuffled timepoints back together
        shuffled_data = np.stack(D_shuffled, axis=1)
        datas_val_shuffled.append(shuffled_data)

    # Compile validation data - ensure all are tensors
    val_datas = []
    val_graphs = []
    val_actions = []
    
    for item in datas_val_shuffled:
        if not torch.is_tensor(item):
            val_datas.append(torch.tensor(item, dtype=torch.float32))
        else:
            val_datas.append(item)
            
    for item in graphs_val:
        if not torch.is_tensor(item):
            val_graphs.append(torch.tensor(item, dtype=torch.float32))
        else:
            val_graphs.append(item)
            
    for item in actions_val:
        if not torch.is_tensor(item):
            val_actions.append(torch.tensor(item, dtype=torch.float32))
        else:
            val_actions.append(item)
    
    # Concatenate the data
    if len(val_datas) > 0:
        val_data = torch.cat(val_datas, dim=0)
        
        # For the graph, we just need one copy per sample
        # So we repeat the graph for each sample from the same dataset
        counts = [d.shape[0] for d in val_datas]
        expanded_graphs = []
        
        for g, count in zip(val_graphs, counts):
            # For each graph, repeat it for the number of samples
            expanded_graphs.extend([g] * count)
            
        val_graph = torch.stack(expanded_graphs, dim=0)
        val_action = torch.cat(val_actions, dim=0)
        
        return val_data, val_graph, val_action
    else:
        # Return empty tensors if no data is available
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

# Time-series CFM W/ interventional conditional information
# OT-CFM

num_iters = 10000
batch_size = 64

seeds = [1, 2, 3, 4, 5]
dd_metrics_df = []

for seed in seeds:
    print("Training for seed =", seed, "...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, train_graph, train_action = get_train_dataset(seed)
    
    # Skip this seed if no data was loaded
    if train_data.shape[0] == 0:
        print(f"No data available for seed {seed}, skipping...")
        continue

    # compute full-batch OT matrix
    pis = [compute_pi(train_data[:,t], train_data[:,t+1]) for t in range(train_data.shape[1]-1)]

    fig, axes = plt.subplots(1,5, figsize=(16,4))
    axes = axes.flatten()

    sigma = 0.1
    dim = train_data.shape[-1]
    model = MLP(dim=dim, w=64, time_varying=False)
    print("Size of model", count_parameters(model), "Parameters")
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.01,
        )

    for k in range(num_iters):
            
        # OT resample
        i_s = np.random.choice(train_data.shape[0], size=batch_size, )
        t_select = torch.tensor(np.random.choice(train_data.shape[1] - 2, size=batch_size, )) # minus 2 to leave out end point
        x0 = train_data[i_s, t_select, :]
        x1 = []
        for j,t in enumerate(t_select):
            choice = np.random.choice(train_data.shape[0], p=pis[t][i_s[j]] / pis[t][i_s[j]].sum())
            x1.append(train_data[choice, t+1, :])
        x1 = torch.stack(x1)

        # update params step
        optimizer.zero_grad()
        t = torch.rand(x0.shape[0], 1)
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = sigma
        t = t + t_select[:, None].to(t)
        x = mu_t + sigma_t * torch.randn(x0.shape[0], dim)
        ut = x1 - x0
        vt = model(x) 
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()

        if (k + 1) % 1000 == 0:
            print(f"{k+1}: loss {loss.item():0.3f}")
    
    
    # run final validation step 
    val_data, val_graph, val_action = get_val_dataset(seed)
    
    # Skip validation if no validation data
    if val_data.shape[0] == 0:
        print(f"No validation data available for seed {seed}, skipping validation...")
        continue
        
    val_dataset = TimeSeriesInterventionBifurcatingDataset(val_data, val_graph, val_action, time_step=1)
    val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=False)
    with torch.no_grad():
        for val_idx, val_batch in enumerate(val_dataloader):
            x, a = val_batch
            x0, x1, t_select = preprocess_batch(x, training=False)
            ts = x.shape[1]
            pred_traj, dd_t2_values, pca, dd_df = validation_step(model=model, val_data_full=val_data, x0=x0, x1=x1, graph=true_graph, a=a, x=x, ts=ts, axes_list=axes)
            if pred_traj is not None:  # Only plot if we had valid data
                dd_metrics_df.append(dd_df)
                plot_sc_trajectories(pred_traj, val_data, a, dd_t2_values[1], pca, axes[2])

if dd_metrics_df:
    df = pd.concat(dd_metrics_df, axis=1)
    print(df)
    df_metrics_mean_std = pd.DataFrame()
    df_metrics_mean_std["mean"] = df.mean(axis=1)
    df_metrics_mean_std["std"] = df.std(axis=1)
    print(df_metrics_mean_std)
else:
    print("No metrics to display. Make sure data is being loaded correctly.")

# Time-series SCM-CFM W/ interventional conditional information
# NGM-OT-CFM

num_iters = 10000
batch_size = 64

seeds = [1, 2, 3, 4, 5]
dd_metrics_df = []

for seed in seeds:
    print("Training for seed =", seed, "...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, train_graph, train_action = get_train_dataset(seed)
    
    # Skip this seed if no data was loaded
    if train_data.shape[0] == 0:
        print(f"No data available for seed {seed}, skipping...")
        continue

    # compute full-batch OT matrix
    pis = [compute_pi(train_data[:,t], train_data[:,t+1]) for t in range(train_data.shape[1]-1)]
    
    fig, axes = plt.subplots(1,5, figsize=(16,4))
    axes = axes.flatten()

    sigma = 0.1
    dim = train_data.shape[-1]
    dims = [100, 1]
    dims = [dim, *dims]
    model = MLPODEF(dims, time_varying=False)
    model.reset_parameters()
    print("Size of model", count_parameters(model), "Parameters")
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.01,
        )

    for k in range(num_iters):
            
        # OT resample
        i_s = np.random.choice(train_data.shape[0], size=batch_size, )
        t_select = torch.tensor(np.random.choice(train_data.shape[1] - 2, size=batch_size, )) # minus 2 to leave out end point
        x0 = train_data[i_s, t_select, :]
        x1 = []
        for j,t in enumerate(t_select):
            choice = np.random.choice(train_data.shape[0], p=pis[t][i_s[j]] / pis[t][i_s[j]].sum())
            x1.append(train_data[choice, t+1, :])
        x1 = torch.stack(x1)

        # update params step
        optimizer.zero_grad()
        t = torch.rand(x0.shape[0], 1)
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = sigma
        t = t + t_select[:, None].to(t)
        x = mu_t + sigma_t * torch.randn(x0.shape[0], dim)
        ut = x1 - x0
        vt = model(x) 
        loss = torch.mean((vt - ut) ** 2) + 1e-6*model.l1_reg()
        loss.backward()
        optimizer.step()

        if (k + 1) % 1000 == 0:
            G = model.get_structure().cpu().detach().numpy()  # [i, j]
            if true_graph.shape[0] > 0 and G.shape[0] > 0:  # Only compute metrics if graphs are valid
                self_loop_mask = ~np.eye(G.shape[-1], dtype=bool)
                try:
                    df_graph_metrics = compute_metrics((true_graph[self_loop_mask]).flatten(), (G[self_loop_mask]).flatten())
                    auc = df_graph_metrics[0].values[0]
                    print(f"{k+1}: loss {loss.item():0.3f} AUC {auc:0.3f}")
                except Exception as e:
                    print(f"{k+1}: loss {loss.item():0.3f} Error computing AUC: {e}")
            else:
                print(f"{k+1}: loss {loss.item():0.3f}")

    # run final validation step
    val_data, val_graph, val_action = get_val_dataset(seed)
    
    # Skip validation if no validation data
    if val_data.shape[0] == 0:
        print(f"No validation data available for seed {seed}, skipping validation...")
        continue
        
    val_dataset = TimeSeriesInterventionBifurcatingDataset(val_data, val_graph, val_action, time_step=1)
    val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=False)
    with torch.no_grad():
        for val_idx, val_batch in enumerate(val_dataloader):
            x, a = val_batch
            x0, x1, t_select = preprocess_batch(x, training=False)
            ts = x.shape[1]
            G = model.get_structure().cpu().detach().numpy()  # [i, j]
            pred_traj, dd_t2_values, pca, dd_df = validation_step(model=model, val_data_full=val_data, x0=x0, x1=x1, graph=true_graph, a=a, G=G, x=x, ts=ts, axes_list=axes)
            if pred_traj is not None:  # Only plot if we had valid data
                dd_metrics_df.append(dd_df)
                plot_sc_trajectories(pred_traj, val_data, a, dd_t2_values[1], pca, axes[2])
            

if dd_metrics_df:
    df = pd.concat(dd_metrics_df, axis=1)
    print(df)
    df_metrics_mean_std = pd.DataFrame()
    df_metrics_mean_std["mean"] = df.mean(axis=1)
    df_metrics_mean_std["std"] = df.std(axis=1)
    print(df_metrics_mean_std)
else:
    print("No metrics to display. Make sure data is being loaded correctly.")