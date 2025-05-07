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

def plot(simdir):
    
    simulation_dir = os.path.join(simdir, "simulations")
    n_cells = len([s for s in os.listdir(simulation_dir) if s.endswith(".csv")])
    dfs = [0] * n_cells
    for f in os.listdir(simulation_dir):
        if not f.endswith("csv"):
            continue
        i = int(f.split(".")[0][1:])
        dfs[i] = pd.read_csv(os.path.join(simulation_dir, f), index_col=0)
    ref_net = pd.read_csv(os.path.join(simdir, "refNetwork.csv"))
    n_genes = dfs[0].shape[0]
    genes = [f"g{i+1}" for i in range(n_genes)]

    data = []
    for df in dfs:
        data.append(df.values)
    data = np.array(data)

    #print(data.shape)
    data = np.swapaxes(data, 1, 2)
    
    # cells x time x genes
    data = data[:, ::9, :]
    data = data[:, 1:, :]


    A = np.zeros((n_genes, n_genes))
    for i, row in ref_net.iterrows():
        gene1 = int(row["Gene1"][1:]) # - 1 if gene names don't start at zero
        gene2 = int(row["Gene2"][1:]) # - 1 if gene names don't start at zero
        rel = 1 if row["Type"] == "+" else -1
        A[gene1, gene2] = rel

    return data, np.abs(A)

datas_train, datas_val = [], []
graphs_train, graphs_val = [], []
actions_train, actions_val = [], []
train_val_split = [0.8, 0.2]
for path in os.listdir("Synthetic-I"):
    p = os.path.join("Synthetic-I", path)
    if 'TF-I-gnull' not in p:
        continue
    else:
        print(p)
        data, graph = plot(p)
        train_len = int(data.shape[0]*train_val_split[0])
        val_len = int(data.shape[0]*train_val_split[1])
        print(data.shape, train_len, val_len)
        action = np.ones((data.shape[0], data.shape[-1]))
        if 'null' in p:
            print("No intervention")
            actions_train.append(action[:train_len])
            actions_val.append(action[train_len:])
        else:
            if 'I-3' not in p:
                a = int(list(filter(str.isdigit, p))[0])
            else:
                a = int(list(filter(str.isdigit, p))[1])
            print("Knock-out intervention on g%d" % a)
            action[:, a] = 0
            actions_train.append(action[:train_len])
            actions_val.append(action[train_len:])
        graphs_train.append(graph)
        graphs_val.append(graph)
        datas_train.append(data[:train_len])
        datas_val.append(data[train_len:])
print("Example boolODE data:", datas_train[0].shape, datas_val[0].shape)
true_graph = graphs_val[0]

import networkx as nx
fig, axes = plt.subplots(3,1, figsize=(4,16))
axes = axes.flatten()
for ax, data, graph in zip(axes, datas_train, graphs_train):
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    print(type(G))
    nx.draw(G, ax=ax, pos=nx.circular_layout(G), arrows=True, with_labels=True)
plt.tight_layout()

from sklearn.decomposition import PCA
fig, axes = plt.subplots(1,3, figsize=(16,4))
axes = axes.flatten()
pca = PCA(n_components=2)
cells_null = datas_train[0].reshape(-1, datas_train[0].shape[-1])
pca_embed = pca.fit_transform(cells_null)
for ax, data, graph in zip(axes, datas_train, graphs_train):
    cells = data.reshape(-1, data.shape[-1])
    pca_embed = pca.transform(cells)
    labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten()
    scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True)
plt.tight_layout()

fig, axes = plt.subplots(1,3, figsize=(16,4))
axes = axes.flatten()
pca = PCA(n_components=2)
cells_null = datas_val[0].reshape(-1, datas_val[0].shape[-1])
pca_embed = pca.fit_transform(cells_null)
for ax, data, graph in zip(axes, datas_val, graphs_val):
    cells = data.reshape(-1, data.shape[-1])
    pca_embed = pca.transform(cells)
    labels = np.repeat(np.arange(data.shape[1])[None,:], data.shape[0], axis=0).flatten()
    scprep.plot.scatter2d(pca_embed, c=labels, ax=ax, ticks=False, colorbar=True)
plt.tight_layout()

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
        return model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


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

def validation_step(model, val_data_full, x0, x1, graph, G=None, a=None, init_run=False, x=None, ts=None, time_steps=100, axes_list=None):
    ax1, ax2 = axes_list[0], axes_list[1]
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
            if init_run:
                pca = plot_pca_manifold(val_data_full, x1, ax1, a=a)
            else:
                pca = plot_pca_manifold(val_data_full, x1, axes_list[0], a=a)
                pca = plot_pca_manifold(val_data_full, traj[-1], axes_list[1], a=a)
        else:
            # get estimated graph
            shd = structural_hamming_distance(graph, G)
            #auc = roc_auc_score(graph.flatten(), G.flatten())
            #auc = compute_metrics(graph.flatten(), G.flatten())
            self_loop_mask_for_plot = np.ones((G.shape[-1], G.shape[-1])) - np.eye(G.shape[-1])
            #auc = compute_metrics((graph*self_loop_mask).flatten(), (G*self_loop_mask).flatten())
            self_loop_mask = ~np.eye(G.shape[-1], dtype=bool)
            df_graph_metrics = compute_metrics((graph[self_loop_mask]).flatten(), (G[self_loop_mask]).flatten())
            dd_df = dd_df.append(df_graph_metrics)
            auc = df_graph_metrics[0].values[0]
            #dd_df.loc["AUC"] = auc
            print("SHD =", shd, "AUC =", auc)
            if init_run:
                plot_graph_heatmap(graph*self_loop_mask_for_plot, ax=ax2)
                pca = plot_pca_manifold(val_data_full, x1, ax1, a=a)
            else:
                plot_graph_heatmap(graph*self_loop_mask_for_plot, ax=axes_list[3])
                plot_graph_heatmap(G*self_loop_mask_for_plot, ax=axes_list[4], auc=auc)
                pca = plot_pca_manifold(val_data_full, x1, axes_list[0], a=a)
                pca = plot_pca_manifold(val_data_full, traj[-1], axes_list[1], a=a)
    return traj, dd_t2_values, pca, dd_df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_sc_trajectories(traj, sc_boolode, a, W2, pca_full, ax):
    n = 2000
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
    ax.scatter(traj_pca[0, :n, 0], traj_pca[0, :n, 1], s=10, alpha=0.8, c="black")
    ax.scatter(traj_pca[:, :n, 0], traj_pca[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    #ax.scatter(traj_pca[-1, :n, 0], traj_pca[-1, :n, 1], s=4, alpha=1, c="blue")
    for k, v in group_dict.items():
        if len(v) > 0:
            traj_pca_0, traj_pca_1 = data_dict_traj_0[k], data_dict_traj_1[k]
            #ax.scatter(traj_pca[-1, :n, 0], traj_pca[-1, :n, 1], s=4, alpha=1, c=v, label=k)
            ax.scatter(traj_pca_0, traj_pca_1, s=4, alpha=1, c=v, label=k)
    ax.legend(["Prior sample z(S)", "Flow", "z(0)"])
    ax.set_title(r'2-Wasserstein($x_T$,  $\hat{x_T}$) = %f' % W2)
    #ax.xticks([])
    #ax.yticks([])
    #ax.show()
    plt.tight_layout()
    
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

num_iters = 10000
batch_size = 64

seeds = [1, 2, 3, 4, 5]
dd_metrics_df = []

for seed in seeds:
    print("Training for seed =", seed, "...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, _, _ = get_train_dataset(seed)

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
    val_dataset = TimeSeriesInterventionBifurcatingDataset(val_data, val_graph, val_action, time_step=1)
    val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=False)
    with torch.no_grad():
        for val_idx, val_batch in enumerate(val_dataloader):
            x, a = val_batch
            x0, x1, t_select = preprocess_batch(x, training=False)
            ts = x.shape[1]
            pred_traj, dd_t2_values, pca, dd_df = validation_step(model=model, val_data_full=val_data, x0=x0, x1=x1, graph=true_graph, a=a, x=x, ts=ts, axes_list=axes)
            dd_metrics_df.append(dd_df)
            plot_sc_trajectories(pred_traj, val_data, a, dd_t2_values[1], pca, axes[2])
            
        
df = pd.concat(dd_metrics_df, axis=1)
print(df)
df_metrics_mean_std = pd.DataFrame()
df_metrics_mean_std["mean"] = df.mean(axis=1)
df_metrics_mean_std["std"] = df.std(axis=1)
print(df_metrics_mean_std)

# Time-series SCM-CFM W/ internvetional conditional information
# NGM-OT-CFM

num_iters = 10000
batch_size = 64

seeds = [1, 2, 3, 4, 5]
#seeds = [2]
dd_metrics_df = []

for seed in seeds:
    print("Training for seed =", seed, "...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, _, _ = get_train_dataset(seed)

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
            self_loop_mask = ~np.eye(G.shape[-1], dtype=bool)
            df_graph_metrics = compute_metrics((true_graph[self_loop_mask]).flatten(), (G[self_loop_mask]).flatten())
            auc = df_graph_metrics[0].values[0]
            print(f"{k+1}: loss {loss.item():0.3f} AUC {auc:0.3f}")

    # run final validation step
    val_data, val_graph, val_action = get_val_dataset(seed)
    val_dataset = TimeSeriesInterventionBifurcatingDataset(val_data, val_graph, val_action, time_step=1)
    val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=False)
    with torch.no_grad():
        for val_idx, val_batch in enumerate(val_dataloader):
            x, a = val_batch
            x0, x1, t_select = preprocess_batch(x, training=False)
            ts = x.shape[1]
            G = model.get_structure().cpu().detach().numpy()  # [i, j]
            pred_traj, dd_t2_values, pca, dd_df = validation_step(model=model, val_data_full=val_data, x0=x0, x1=x1, graph=true_graph, a=a, G=G, x=x, ts=ts, axes_list=axes)
            dd_metrics_df.append(dd_df)
            plot_sc_trajectories(pred_traj, val_data, a, dd_t2_values[1], pca, axes[2])
            

df = pd.concat(dd_metrics_df, axis=1)
print(df)
df_metrics_mean_std = pd.DataFrame()
df_metrics_mean_std["mean"] = df.mean(axis=1)
df_metrics_mean_std["std"] = df.std(axis=1)
print(df_metrics_mean_std)

