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
        # Ensure non-negativity and finite values
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0) # Replace NaN/Inf with 0
        p = torch.clamp(p, min=0) 

        # Add a small epsilon to prevent sum=0 after clamping
        p = p + 1e-12 
        p_sum = p.sum()

        if p_sum <= 0 or not torch.isfinite(p_sum):
             print(f"Warning: Invalid sum encountered in sample_map: {p_sum}. Setting p to uniform.", file=sys.stderr)
             # Fallback to uniform distribution if sum is invalid
             num_elements = pi.numel()
             p = torch.ones_like(p) / num_elements
        else:
            p = p / p_sum # Normalize

        # Final check before multinomial
        if not torch.all(torch.isfinite(p)) or torch.any(p < 0):
            print(f"Warning: Invalid probabilities detected before multinomial: min={p.min()}, max={p.max()}, has_nan={torch.isnan(p).any()}. Falling back to uniform.", file=sys.stderr)
            num_elements = pi.numel()
            p = torch.ones_like(p) / num_elements
            
        # Ensure p is on CPU if multinomial requires it (often does)
        p_cpu = p.cpu() 
        choices = torch.multinomial(p_cpu, num_samples=batch_size, replacement=replace)
        return np.divmod(choices.numpy(), pi.shape[1]) # Use numpy divmod as choices is now numpy

    def sample_plan(self, x0, x1, pi, batch_size, replace=True):
        # Ensure pi is a tensor before passing to sample_map
        if not isinstance(pi, torch.Tensor):
            pi = torch.tensor(pi, dtype=torch.float32)
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
        def entropic_ot_plan(x0, x1, eps):

            x0_cpu = x0.cpu()
            x1_cpu = x1.cpu()

            C = ot.utils.euclidean_distances(x0_cpu, x1_cpu, squared=True) / 2
            
            # --- Add Cost Matrix Normalization ---
            # Check for invalid values before normalization
            if not torch.all(torch.isfinite(C)):
                print(f"Warning: Non-finite values detected in Cost Matrix C before normalization. Replacing with 0.", file=sys.stderr)
                C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

            median_C = torch.median(C)
            if median_C > 1e-6: # Avoid division by zero if C is mostly zero
                C = C / median_C
                # Optional: print scale info
                # print(f"Normalized C: median={torch.median(C):.2e}, max={C.max():.2e}, min={C.min():.2e}", file=sys.stderr)
            # else: print(f"Skipping C normalization as median is near zero ({median_C:.2e})", file=sys.stderr)
            # -------------------------------------

            p = torch.full((x0_cpu.shape[0],), 1 / x0_cpu.shape[0], device="cpu")
            q = torch.full((x1_cpu.shape[0],), 1 / x1_cpu.shape[0], device="cpu")
            # Use max to ensure minimum regularization, increase epsilon slightly
            # Adjust eps relative to the normalized C (median=1)
            # Keep the original eps calculation based on sigma/dt, but ensure a minimum floor
            current_eps = eps # This is dt * sigma**2 passed from __init__
            min_eps = 1e-4 # Increased minimum epsilon slightly
            final_eps = max(current_eps, min_eps) 
            # Optional: print eps value
            # print(f"Using eps = {final_eps:.2e}", file=sys.stderr)

            # Switch to stabilized Sinkhorn from ot.bregman
            # Note: ot.sinkhorn_stabilized expects tensors directly
            # Ensure inputs are float64 for stability if possible, POT often uses float64 internally
            p_64 = p.to(dtype=torch.float64)
            q_64 = q.to(dtype=torch.float64)
            C_64 = C.to(dtype=torch.float64)
            # Use ot.bregman.sinkhorn_stabilized
            plan = ot.bregman.sinkhorn_stabilized(p_64, q_64, C_64, final_eps, numItermax=5000) 

            # Check for NaNs/Infs in the returned plan
            plan_tensor = torch.tensor(plan, device="cpu", dtype=torch.float32) # Convert back to float32 if needed
            if not torch.all(torch.isfinite(plan_tensor)):
                print(f"Warning: Sinkhorn produced non-finite plan for eps={final_eps}.", file=sys.stderr)
                # Fallback to uniform plan if Sinkhorn fails completely
                plan_tensor = torch.ones_like(C) / C.numel()
            return plan_tensor

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
