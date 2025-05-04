import copy

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
        time_varying=True,
        conditional=False,
        conditional_dim=0,  # dimension of the knockout or condition
        device=None
    ):
        super().__init__()
        self.time_varying = time_varying
        self.conditional = conditional
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_dim = d
        if self.time_varying:
            input_dim += 1
        if self.conditional:
            input_dim += conditional_dim

        hidden_sizes = copy.copy(hidden_sizes)
        hidden_sizes.insert(0, input_dim)  # first layer's input size
        hidden_sizes.append(d)  # final layer is dimension d

        layers = []
        for i in range(len(hidden_sizes) - 1):
            in_size = hidden_sizes[i]
            out_size = hidden_sizes[i + 1]
            layers.append(nn.Linear(in_size, out_size))
            # activation except for the last layer
            if i < len(hidden_sizes) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)
        self.to(self.device)

        # Weight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x, cond=None):
        device = x.device
        inputs = [x]
        if self.time_varying:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inputs.append(t.to(device))

        if self.conditional:
            if cond is None:
                raise ValueError("Conditional flag = True, but no 'cond' input provided.")
            Bx = x.shape[0]
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(Bx, -1)
            elif cond.shape[0] != Bx:
                raise ValueError(f"cond batch size ({cond.shape[0]}) != x batch size ({Bx}). ")
            inputs.append(cond.to(device))

        # cat along dim=1 => shape [batch_size, (d + time + cond_dim)]
        net_in = torch.cat(inputs, dim=1)
        return self.net(net_in)

class MLPFlow(nn.Module):
    def __init__(
        self, dims, GL_reg=0.01, bias=True, time_invariant=True, knockout_masks=None, device=None
    ):
        # dims: [number of variables, hidden_layer_1_dim, hidden_layer_2_dim, ..., output_dim=1]
        super(MLPFlow, self).__init__()
        self.dims = dims
        self.d = dims[0]  # Number of variables
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # For compatibility with MLPODEF1
        self.knockout_masks = None
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if knockout_masks is not None:
            self.knockout_masks = [
                torch.tensor(m, dtype=torch.float32, device=self.device) for m in knockout_masks
            ]

        # Input dimension includes time if not time_invariant
        input_dim = self.d
        if not time_invariant:
            input_dim += 1

        # Build MLP layers dynamically
        layers = []
        prev_dim = input_dim

        # Create all layers except the last one
        for hidden_dim in dims[1:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(nn.ELU())
            prev_dim = hidden_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, self.d, bias=bias))

        self.network = nn.Sequential(*layers)
        self.to(self.device)

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)

    def forward(self, t, x, dataset_idx=None):  # [n, 1, d] -> [n, 1, d]
        device = x.device
        # Reshape input if needed
        if x.dim() == 3:  # [n, 1, d]
            x = x.squeeze(1)  # [n, d]

        # Combine with time if not time_invariant
        if not self.time_invariant:
            if t.dim() == 3:
                t = t.squeeze(1)
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            x = torch.cat((x, t.to(device)), dim=-1)

        # Forward pass through MLP
        out = self.network(x)  # [n, d]

        # Reshape to match expected output format
        out = out.unsqueeze(1)  # [n, 1, d]
        return out
    
    def l2_reg(self):
        """L2 regularization on all parameters"""
        reg = 0.0
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                reg += torch.sum(layer.weight**2)
        return reg

    def fc1_reg(self):
        """
        L1 regularization on input layer parameters
        For standard MLP, we return 0 as this is specific to MLPODEF
        """
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)

    def causal_graph(self, w_threshold=0.3):
        """
        Create a causal graph representation from the weights
        For MLP, we'll use the first layer weights as a proxy for causal relationships
        """
        first_layer = next(
            layer for layer in self.network if isinstance(layer, nn.Linear)
        )
        W = first_layer.weight.detach().cpu().numpy()
        # Take the absolute values and reshape to [d, d]
        W = np.abs(W[: self.d, : self.d])
        W[np.abs(W) < w_threshold] = 0
        return np.round(W, 2)

    def reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()