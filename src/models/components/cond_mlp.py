import copy

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
    ):
        super().__init__()
        self.time_varying = time_varying
        self.conditional = conditional

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

        # Weight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x, cond=None):
        inputs = [x]
        if self.time_varying:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inputs.append(t)

        if self.conditional:
            if cond is None:
                raise ValueError("Conditional flag = True, but no 'cond' input provided.")
            Bx = x.shape[0]
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(Bx, -1)
            elif cond.shape[0] != Bx:
                raise ValueError(f"cond batch size ({cond.shape[0]}) != x batch size ({Bx}). ")
            inputs.append(cond)

        # cat along dim=1 => shape [batch_size, (d + time + cond_dim)]
        net_in = torch.cat(inputs, dim=1)
        return self.net(net_in)
