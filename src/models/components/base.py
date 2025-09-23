"""Implements dynamics models that support interventions on a known and prespecified set of
targets."""

import functools
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from .graph_samplers import GraphLayer


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

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
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

    def forward(self, input_: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input_.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_linear={}, in_features={}, out_features={}, bias={}".format(
            self.num_linear, self.in_features, self.out_features, self.bias is not None
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


class NNODEF(nn.Module):
    def __init__(self, in_dim, hid_dim, time_invariant=True):
        super().__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim + 1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):

        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out


class MLPODEF(nn.Module):
    def __init__(self, dims, GL_reg=0.01, bias=True, time_invariant=True):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # adaptive lasso parameter

        if time_invariant:
            self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        else:
            self.fc1 = nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias)

        # fc2: local linear layers
        layers = []
        for layer in range(len(dims) - 2):
            layers.append(LocallyConnected(dims[0], dims[layer + 1], dims[layer + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        self.elu = nn.ELU(inplace=True)

        # Initialize a mask buffer for knockout. By default, no mask is applied
        self.register_buffer("ko_mask", torch.ones(self.fc1.weight.shape))

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]

        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        # we do a linear operation with the masked weights
        w = self.fc1.weight * self.ko_mask
        x = F.linear(x, w, self.fc1.bias)  # x: [n,1,d], w: [d*m1, d]

        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = fc(self.elu(x))  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        x = x.unsqueeze(dim=1)  # [n, 1, d]
        return x

    def l2_reg(self):
        """L2 regularization on all parameters."""
        reg = 0.0
        fc1_weight = self.fc1.weight  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def fc1_reg(self):
        """L1 regularization on input layer parameters."""
        return torch.sum(torch.abs(self.fc1.weight))

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def causal_graph(self, w_threshold=0.3):  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim."""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i] or [j * m1, i+1] if time-varying

        if not self.time_invariant:
            # Remove the time dimension (last column) before reshaping
            fc1_weight = fc1_weight[:, :-1]  # [j * m1, i]

        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        W = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        W[np.abs(W) < w_threshold] = 0
        return np.round(W, 2)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()


class MLPODEFKO(nn.Module):
    def __init__(self, dims, GL_reg=0.01, bias=True, time_invariant=True, knockout_masks=None, device=None):
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims = dims
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if time_invariant:
            self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        else:
            self.fc1 = nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias)
        
        layers = []
        for layer in range(len(dims) - 2):
            layers.append(LocallyConnected(dims[0], dims[layer + 1], dims[layer + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        self.elu = nn.GELU() #nn.ELU(inplace=True)
        self.knockout_masks = None
        
        if callable(knockout_masks):
            knockout_masks = knockout_masks()

        if knockout_masks is not None:
            self.knockout_masks = [
                torch.tensor(m, dtype=torch.float32) 
                if not isinstance(m, torch.Tensor) else m
                for m in knockout_masks
            ]
        
        self.to(self.device)

    def forward(self, t, x, dataset_idx=None, step=0):
        x = x.to(self.device)
        t = t.to(self.device)
        
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
            
        if dataset_idx is not None and self.knockout_masks is not None:
            mask = self.knockout_masks[dataset_idx].to(self.device)
            x = x.squeeze(1)
            d = self.dims[0]
            m = self.dims[1]
            
            if self.time_invariant:
                w_raw = self.fc1.weight
                w_reshaped = w_raw.view(d, m, d)
                masked_w = w_reshaped * mask.unsqueeze(1)
                x_out = torch.einsum("rhd,nd->nrh", masked_w, x)
            else:
                w_raw = self.fc1.weight
                w_vars = w_raw[:, :d]
                w_time = w_raw[:, d:]
                w_vars_reshaped = w_vars.view(d, m, d)
                masked_w_vars = w_vars_reshaped * mask.unsqueeze(1)
                x_vars = x[:, :d]
                x_time = x[:, d:]
                out_vars = torch.einsum("rhd,nd->nrh", masked_w_vars, x_vars)
                w_time_reshaped = w_time.view(d, m, 1)
                out_time = w_time_reshaped * x_time.unsqueeze(1)
                x_out = out_vars + out_time
                
            if self.fc1.bias is not None:
                bias = self.fc1.bias.view(d, m)
                x_out = x_out + bias.unsqueeze(0)
        else:
            x = self.fc1(x)
            x_out = x.view(-1, self.dims[0], self.dims[1])
            
        for fc in self.fc2:
            x_out = fc(self.elu(x_out))
        x_out = x_out.squeeze(dim=2)
        x_out = x_out.unsqueeze(dim=1)
        return x_out

    def l2_reg(self):
        """L2 regularization on all parameters."""
        reg = 0.0
        fc1_weight = self.fc1.weight  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def fc1_reg(self):
        """L1 regularization on input layer parameters."""
        return torch.sum(torch.abs(self.fc1.weight))

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def causal_graph(self, w_threshold=0.3):  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim."""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        W = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [j, i]
        W = W.cpu().detach().numpy()  
        W[np.abs(W) < w_threshold] = 0
        return np.round(W, 2)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()


### Experimental layer ###
class KOGraph(nn.Module):
    def __init__(
        self, 
        dims, 
        bias=True,
        time_invariant=True,
        knockout_masks=None, 
        w_init_std=2e-2,
        warmup_steps=1000,
        alpha=0.1,
    ):
        super().__init__()
        assert len(dims) >= 2 and dims[-1] == 1
        d, m1 = dims[0], dims[1]
        self.dims = dims
        self.time_invariant = time_invariant
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.w_init_std = w_init_std

        self.w = nn.Linear(d, d*m1, bias=bias)

        layers = []
        for layer in range(len(dims) - 2):
            layers.append(LocallyConnected(dims[0], dims[layer + 1], dims[layer + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        self.act = nn.GELU()

        self.knockout_masks = None
        if callable(knockout_masks): knockout_masks = knockout_masks()
        if knockout_masks is not None:
            self.knockout_masks = [m if isinstance(m, torch.Tensor)
                                   else torch.tensor(m, dtype=torch.float32)
                                   for m in knockout_masks]
            for k, M in enumerate(self.knockout_masks):
                self.register_buffer(f"KO_mask_{k}", M)

    def get_mask(self, dataset_idx):
        if dataset_idx is None or self.knockout_masks is None: return None
        return getattr(self, f"KO_mask_{dataset_idx}")
    
    def _temperature(self, step: int):
        if step <= self.warmup_steps:
            return self.alpha
        return self.alpha * (1.0 + math.log1p(step - self.warmup_steps))
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=0, std=self.w_init_std)
    
    def forward(self, t, x, dataset_idx=None, step=0):  # [n, 1, d] -> [n, 1, d]
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        xb = x.squeeze(1) 
        alpha_t = self._temperature(step)

        W = self.w
        M = self.get_mask(dataset_idx)
        G = torch.sigmoid(alpha_t * W)

        if M is not None:
            G = G * M.unsqueeze(0)

        out = torch.einsum('hbs,bs->bdh', G, xb)

        if self.w.bias:
            out = out + self.w.bias.unsqueeze(0)

        for fc in self.fc2:
            x = fc(self.act(out))  
        x = x.squeeze(dim=2)
        x = x.unsqueeze(dim=1)
        return x  # x.shape [batch, t, d]

    def l2_reg(self):
        """L2 regularization on input layer parameters."""
        return torch.sum(self.graph() ** 2)

    def l1_reg(self):
        """L1 regularization on input layer parameters."""
        G = self.graph()
        return torch.sum(torch.abs(G)) / G.shape[-1]
    
    def get_structure(self, eval_n_graphs=None, test_mode=None):
        """Score each edge based on the the weight sum."""
        if isinstance(self.graph, GraphLayer):
            return self.graph(eval_n_graphs)
            
        return self.graph(eval_n_graphs, test_mode)

