import math
import random
from functools import partial
from typing import Optional

import numpy as np
import ot as pot
import torch


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
        means = (1 - ts) * x0 + ts * x1
        vars = (sigma**2) * ts * (1 - ts)
        x = means + torch.sqrt(vars.clamp_min(1e-4)) * torch.randn_like(x0)
        s = (-1 / vars.clamp_min(1e-4)) * (x - means)
        u = (1 - 2 * ts) / (2 * ts * (1 - ts) + 1e-4) * (x - means) + x1 - x0
        return means, vars, x, s, u


class EntropicOTFM:
    def __init__(self, x, t_idx, dt, sigma, T, dim, device):
        def entropic_ot_plan(x0, x1, eps):
            C = pot.utils.euclidean_distances(x0, x1, squared=True) / 2
            p, q = torch.full((x0.shape[0],), 1 / x0.shape[0]), torch.full(
                (x1.shape[0],), 1 / x1.shape[0]
            )
            return pot.sinkhorn(p, q, C, eps, method="sinkhorn", numItermax=5000)

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

    def sample_bridges_flows(self, batch_size=64, skip_time=None):
        _x = []
        _t = []
        _t_orig = []
        _s = []
        _u = []
        for i in range(self.T - 1):
            if skip_time is not None and (i == skip_time or i + 1 == skip_time):
                continue

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
            torch.vstack(_x),
            torch.vstack(_s),
            torch.vstack(_u),
            torch.vstack(_t),
            torch.vstack(_t_orig),
        )


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an squared L2 OT plan with
    different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost=False,
        **kwargs,
    ):
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.kwargs = kwargs

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        return p

    def sample_map(self, pi, batch_size):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return x0[i], x1[j]

    def sample_trajectory(self, X):
        # Assume X is [batch, times, dim]
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret
