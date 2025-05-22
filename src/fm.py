import torch
from torch import nn, optim
import copy
import numpy as np
import sys
import math
import ot


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


def _get_coupling(l):
    T = l.u[:, None] * l.K * l.v[None, :]
    return T / T.sum()


class BridgeMatcher:
    def __init__(self):
        pass

    def sample_map(self, pi, batch_size, replace=True):
        p = pi.flatten()
        p = p / p.sum()
        choices = torch.multinomial(p, num_samples=batch_size, replacement=replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, pi, batch_size, replace=True):
        i, j = self.sample_map(pi, batch_size, replace=replace)
        return x0[i], x1[j]

    def sample_bridge_and_flow(self, x0, x1, ts, sigma):
        # Sample Brownian bridges between paired entries of [x0, x1] at times ts \in [0, 1].

        mean_ = (1 - ts) * x0 + ts * x1
        var_ = (sigma**2) * ts * (1 - ts)
        # sample x(tau) = mean + sqrt(var)*epsilon
        eps = torch.randn_like(x0)
        x_tau = mean_ + torch.sqrt(var_.clamp_min(1e-10)) * eps
        s_true = -(x_tau - mean_) / var_.clamp_min(1e-10)
        denom = 2 * ts * (1 - ts) + 1e-10
        u = ((1 - 2 * ts) / denom) * (x_tau - mean_) + (x1 - x0)
        return mean_, var_, x_tau, s_true, u


class EntropicOTFM:
    def __init__(self, x, t_idx, dt, sigma, T, dim, device):
        def entropic_ot_plan(x0_tensor, x1_tensor, eps_val):
            x0_np = x0_tensor.cpu().numpy()
            x1_np = x1_tensor.cpu().numpy()

            if x0_np.shape[0] == 0 or x1_np.shape[0] == 0:
                print(f"Warning (fm.py entropic_ot_plan): Empty tensor for Sinkhorn input. x0_np shape: {x0_np.shape}, x1_np shape: {x1_np.shape}. Returning empty plan.")
                return torch.empty((x0_np.shape[0], x1_np.shape[0]), device="cpu", dtype=torch.float32)

            C_np = ot.utils.euclidean_distances(x0_np, x1_np, squared=True) / 2.0
            C_np = np.ascontiguousarray(C_np, dtype=np.float64)

            large_float_replacement = float(np.finfo(np.float32).max / (x0_np.shape[0] * x1_np.shape[0] + 1e-9)) if (x0_np.shape[0] * x1_np.shape[0] > 0) else float(np.finfo(np.float32).max)
            
            C_np = np.nan_to_num(C_np, nan=large_float_replacement, 
                                 posinf=large_float_replacement, 
                                 neginf=0.0)
            
            min_val_clip = 0.0
            max_val_clip = float(np.finfo(np.float32).max)
            C_np = np.clip(C_np, min_val_clip, max_val_clip)

            p_np = np.full((x0_np.shape[0],), 1.0 / x0_np.shape[0], dtype=np.float64)
            q_np = np.full((x1_np.shape[0],), 1.0 / x1_np.shape[0], dtype=np.float64)
            
            plan_np = np.empty((x0_np.shape[0], x1_np.shape[0]), dtype=np.float64)
            try:
                if x0_np.shape[0] > 0 and x1_np.shape[0] > 0:
                    plan_np = ot.sinkhorn(
                        p_np, q_np, C_np, eps_val,
                        method="sinkhorn_stabilized",
                        numItermax=5000,
                        stopThr=1e-5,
                        warn=True
                    )
            except Exception as e:
                print(f"Sinkhorn algorithm failed in fm.py: {e}. Using uniform fallback plan.")
                if x0_np.shape[0] > 0 and x1_np.shape[0] > 0:
                    plan_np = np.outer(p_np, q_np) * x0_np.shape[0]
            
            return torch.tensor(plan_np, device="cpu", dtype=torch.float32)

        self.sigma = sigma
        self.bm = BridgeMatcher()
        self.x = x
        self.t_idx = t_idx
        self.dt = dt
        self.T = T
        self.dim = dim
        self.device = device
        self.Ts = []
        # construct EOT plans
        for i in range(self.T - 1):
            self.Ts.append(
                entropic_ot_plan(
                    self.x[self.t_idx == i, :],
                    self.x[self.t_idx == i + 1, :],
                    self.dt * self.sigma**2,
                )
            )

    def sample_bridges_flows(self, batch_size=64):
        _x = []
        _t = []
        _t_orig = []
        _s = []
        _u = []
        for i in range(self.T - 1):
            with torch.no_grad():
                x0, x1 = self.bm.sample_plan(
                    self.x[self.t_idx == i, :],
                    self.x[self.t_idx == i + 1, :],
                    self.Ts[i],
                    batch_size,
                )
            ts = torch.rand_like(x0[:, :1])
            _, _, x, s, u = self.bm.sample_bridge_and_flow(
                x0, x1, ts, (self.sigma**2 * self.dt) ** 0.5
            )
            _x.append(x)
            _s.append(s)
            _t.append((i + ts) * self.dt)
            _t_orig.append(ts)
            _u.append(u)
        return (
            torch.vstack(_x).to(self.device),
            torch.vstack(_s).to(self.device),
            torch.vstack(_u).to(self.device),
            torch.vstack(_t).to(self.device),
            torch.vstack(_t_orig).to(self.device),
        )
