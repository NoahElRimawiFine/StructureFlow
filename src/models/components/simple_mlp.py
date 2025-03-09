import torch
from torch import nn, optim
import copy
import numpy as np
import sys
import math


class NoiseScaledMLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
        time_varying=True,
    ):
        super(NoiseScaledMLP, self).__init__()
        self.net = nn.Sequential()
        self.time_varying = time_varying
        assert len(hidden_sizes) > 0
        hidden_sizes = copy.copy(hidden_sizes)
        if time_varying:
            hidden_sizes.insert(0, d + 1)
        else:
            hidden_sizes.insert(0, d)
        hidden_sizes.append(d)
        for i in range(len(hidden_sizes) - 1):
            self.net.add_module(
                name=f"L{i}", module=nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )
            if i < len(hidden_sizes) - 2:
                self.net.add_module(name=f"A{i}", module=activation())
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x, s):
        if self.time_varying:
            return self.net(
                torch.hstack(
                    [
                        x,
                        t.expand(*x.shape[:-1], 1),
                    ]
                )
            ) / s.expand(*x.shape[:-1], 1)
        else:
            return self.net(x) / s.expand(*x.shape[:-1], 1)


class ScalarConditionalMLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
        time_varying=True,
    ):
        super(ScalarConditionalMLP, self).__init__()
        self.net = nn.Sequential()
        self.time_varying = time_varying
        assert len(hidden_sizes) > 0
        hidden_sizes = copy.copy(hidden_sizes)
        if time_varying:
            hidden_sizes.insert(0, d + 2)
        else:
            hidden_sizes.insert(0, d + 1)
        hidden_sizes.append(d)
        for i in range(len(hidden_sizes) - 1):
            self.net.add_module(
                name=f"L{i}", module=nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )
            if i < len(hidden_sizes) - 2:
                self.net.add_module(name=f"A{i}", module=activation())
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x, s):
        if self.time_varying:
            return self.net(
                torch.hstack(
                    [x, t.expand(*x.shape[:-1], 1), s.expand(*x.shape[:-1], 1)]
                )
            )
        else:
            return self.net(x)


class MLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
        time_varying=True,
    ):
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        self.time_varying = time_varying
        assert len(hidden_sizes) > 0
        hidden_sizes = copy.copy(hidden_sizes)
        if time_varying:
            hidden_sizes.insert(0, d + 1)
        else:
            hidden_sizes.insert(0, d)
        hidden_sizes.append(d)
        for i in range(len(hidden_sizes) - 1):
            self.net.add_module(
                name=f"L{i}", module=nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )
            if i < len(hidden_sizes) - 2:
                self.net.add_module(name=f"A{i}", module=activation())
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x):
        if self.time_varying:
            return self.net(torch.hstack([x, t.expand(*x.shape[:-1], 1)]))
        else:
            return self.net(x)