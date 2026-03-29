import torch
from torch import nn, optim
import copy
import numpy as np
import sys
import math
import ot
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


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
        x = means + torch.sqrt(vars.clamp_min(1e-8)) * torch.randn_like(x0)
        s = (-1 / vars.clamp_min(1e-8)) * (x - means)
        u = (1 - 2 * ts) / (2 * ts * (1 - ts) + 1e-8) * (x - means) + x1 - x0
        return means, vars, x, s, u


class EntropicOTFM:
    def __init__(
        self,
        x,
        x_pca,
        t_idx,
        dt,
        sigma,
        T,
        dim,
        tau=1.0,
        alpha=0.4,
        model="eot",
        device="cpu",
        held_out_time=None,
        normalize_C=False,
        lamda=None,
        dt_values=None,
    ):
        def entropic_ot_plan(x0, x1, eps, normalize_C, lamda):
            C = pot.utils.euclidean_distances(x0, x1, squared=True) / 2
            C_mean = C.mean().item()
            p, q = torch.full(
                (x0.shape[0],), 1.0 / x0.shape[0], dtype=x0.dtype, device=x0.device
            ), torch.full(
                (x1.shape[0],), 1.0 / x1.shape[0], dtype=x1.dtype, device=x1.device
            )
            if lamda is None:
                return pot.bregman.sinkhorn(
                    p.double(), q.double(), C.double(), reg=eps, numItermax=10000
                ).float()
            else:
                return pot.unbalanced.sinkhorn_stabilized_unbalanced(
                    p.double(),
                    q.double(),
                    C.double(),
                    eps,
                    lamda * C_mean,
                    numItermax=10000,
                ).float()

        def plan_uot(x0, x1, eps, normalize_C, tau):
            C = pot.utils.euclidean_distances(x0, x1, squared=True) / 2

            if normalize_C:
                C = C / C.max()
            a = torch.full(
                (x0.shape[0],), 1.0 / x0.shape[0], dtype=x0.dtype, device=x0.device
            )
            b = torch.full(
                (x1.shape[0],), 1.0 / x1.shape[0], dtype=x1.dtype, device=x1.device
            )
            try:
                T = pot.unbalanced.sinkhorn_unbalanced(
                    a, b, C, reg=eps, reg_m=tau, numItermax=5000
                )
            except AttributeError:
                T = pot.unbalanced.sinkhorn_knopp_unbalanced(
                    a, b, C, reg=eps, reg_m=tau, numItermax=5000
                )
            return T

        def compute_graph_distances(
            data, n_neighbors=5, mode="distance", metric="correlation"
        ):
            """ """
            graph = kneighbors_graph(
                data,
                n_neighbors=n_neighbors,
                mode=mode,
                metric=metric,
                include_self=True,
            )
            shortestPath = dijkstra(
                csgraph=csr_matrix(graph), directed=False, return_predecessors=False
            )
            max_dist = np.nanmax(shortestPath[shortestPath != np.inf])
            shortestPath[shortestPath > max_dist] = max_dist

            return np.asarray(shortestPath)

        def plan_fgw(Xt, Xt1, Xpca_t, Xpca_t1, eps, alpha):
            from ot.gromov import entropic_fused_gromov_wasserstein

            # Feature cost M on PCA space (or raw x if no PCA provided)
            M = pot.dist(Xpca_t, Xpca_t1)  # same as ot.dist, returns pairwise Euclidean
            if M.max() > 0:
                M = M / M.max()

            k = self.neighbors
            if k is None:
                k = max(1, min(int(0.2 * min(Xt.shape[0], Xt1.shape[0])), 50))
            D1 = compute_graph_distances(Xt, k)
            D2 = compute_graph_distances(Xt1, k)

            # If degenerate (all zeros), fall back to features only
            a = alpha if (D1.max() > 0 and D2.max() > 0) else 0.0

            T, _log = entropic_fused_gromov_wasserstein(
                M, D1, D2, epsilon=eps, alpha=a, log=True
            )
            T = T / T.sum()
            T = torch.from_numpy(T).to(self.device, dtype=torch.float32)
            return T

        self.sigma = sigma
        self.lamda = lamda
        self.bm = BridgeMatcher()
        self.x = x
        self.t_idx = t_idx
        self.dt = dt
        self.T = T
        self.dim = dim
        self.device = device
        self.Ts = []
        self.held_out_time = held_out_time
        self.has_bridge_over_held_out = False
        self.normalize_C = normalize_C
        self.tau = tau
        self.model = model
        self.alpha = alpha
        self.x_pca = x_pca
        self.neighbors = None
        self.dt_values = dt_values if dt_values is not None else np.ones(T - 1) * dt

        # construct EOT plans
        for i in range(self.T - 1):
            dt_i = self.dt_values[i] if i < len(self.dt_values) else self.dt
            if self.held_out_time is not None and (
                i == self.held_out_time or i + 1 == self.held_out_time
            ):
                self.Ts.append(None)

                # Create a bridge over the held-out time if it's the first encounter
                if i == self.held_out_time and not self.has_bridge_over_held_out:
                    dt_bridge = (
                        self.dt_values[i - 1] + self.dt_values[i]
                        if i > 0 and i < len(self.dt_values)
                        else 2 * self.dt
                    )
                    self.bridge_over_held_out = entropic_ot_plan(
                        self.x[self.t_idx == i - 1, :],
                        self.x[self.t_idx == i + 1, :],
                        dt_bridge * self.sigma**2,
                        self.normalize_C,
                    )
                    self.has_bridge_over_held_out = True
            else:
                if model == "eot":
                    x0 = self.x[self.t_idx == i, :]
                    x1 = self.x[self.t_idx == i + 1, :]
                    print(x0.shape, x1.shape)
                    if x0.shape[0] == 0 or x1.shape[0] == 0:
                        raise ValueError(
                            f"No samples available for transition {i}->{i+1}: "
                            f"x0 count={x0.shape[0]}, x1 count={x1.shape[0]}. "
                            "Check how t_idx and T are defined for this dataset."
                        )
                    self.Ts.append(
                        entropic_ot_plan(
                            x0, x1, dt_i * self.sigma**2, self.normalize_C, self.lamda
                        )
                    )
                elif model == "uot":
                    x0 = self.x[self.t_idx == i, :]
                    x1 = self.x[self.t_idx == i + 1, :]
                    if x0.shape[0] == 0 or x1.shape[0] == 0:
                        raise ValueError(
                            f"No samples available for transition {i}->{i+1}: "
                            f"x0 count={x0.shape[0]}, x1 count={x1.shape[0]}. "
                            "Check how t_idx and T are defined for this dataset."
                        )
                    self.Ts.append(
                        plan_uot(
                            x0,
                            x1,
                            dt_i * self.sigma**2,
                            self.normalize_C,
                            self.tau,
                        )
                    )
                else:
                    x0 = self.x[self.t_idx == i, :]
                    x1 = self.x[self.t_idx == i + 1, :]
                    if x0.shape[0] == 0 or x1.shape[0] == 0:
                        raise ValueError(
                            f"No samples available for transition {i}->{i+1}: "
                            f"x0 count={x0.shape[0]}, x1 count={x1.shape[0]}. "
                            "Check how t_idx and T are defined for this dataset."
                        )
                    self.Ts.append(
                        plan_fgw(
                            x0,
                            x1,
                            self.x_pca[self.t_idx == i, :],
                            self.x_pca[self.t_idx == i + 1, :],
                            dt_i * self.sigma**2,
                            self.alpha,
                        )
                    )

        # Uncomment the following lines if you want to use the MM Sinkhorn method instead
        # eps = 2 * self.dt * self.sigma**2
        # self.Ts = build_mm_sinkhorn(
        #     self.x, self.t_idx, self.T, eps, device=self.device, max_iter=400)

    def sample_bridges_flows(self, batch_size=64, skip_time=None):
        _x = []
        _t = []
        _t_orig = []
        _s = []
        _u = []
        _dt = []
        i = 0
        while i < self.T - 1:
            dt_i = self.dt_values[i] if i < len(self.dt_values) else self.dt
            if skip_time is not None and (i == skip_time or i + 1 == skip_time):
                if i == skip_time and self.has_bridge_over_held_out:
                    dt_bridge = (
                        self.dt_values[i - 1] + self.dt_values[i]
                        if i > 0 and i < len(self.dt_values)
                        else 2 * self.dt
                    )
                    with torch.no_grad():
                        x0, x1 = self.bm.sample_plan(
                            self.x[self.t_idx == i - 1, :],
                            self.x[self.t_idx == i + 1, :],
                            self.bridge_over_held_out,
                            batch_size,
                        )
                    ts = torch.rand_like(x0[:, :1])
                    _, _, x, s, u = self.bm.sample_bridge_and_flow(
                        x0, x1, ts, (self.sigma**2 * dt_bridge) ** 0.5
                    )
                    _x.append(x)
                    _s.append(s)
                    _t.append((i - 1 + ts * 2) * self.dt)
                    _t_orig.append(ts)
                    _u.append(u)
                    _dt.append(torch.full_like(ts, dt_bridge))
                    i += 1
                else:
                    i += 1
            else:
                with torch.no_grad():
                    x0, x1 = self.bm.sample_plan(
                        self.x[self.t_idx == i, :],
                        self.x[self.t_idx == i + 1, :],
                        self.Ts[i],
                        batch_size,
                    )
                ts = torch.rand_like(x0[:, :1])
                _, _, x, s, u = self.bm.sample_bridge_and_flow(
                    x0, x1, ts, (self.sigma**2 * dt_i) ** 0.5
                )
                _x.append(x)
                _s.append(s)
                _t.append((i + ts) * self.dt)
                _t_orig.append(ts)
                _u.append(u)
                _dt.append(torch.full_like(ts, dt_i))
                i += 1
        return (
            torch.vstack(_x),
            torch.vstack(_s),
            torch.vstack(_u),
            torch.vstack(_t),
            torch.vstack(_t_orig),
            torch.vstack(_dt),
        )


def build_mm_sinkhorn(x, t_idx, T, eps, max_iter=400, device="cpu"):

    # split cloud per snapshot + alpha_i initialization
    clouds = [x[t_idx == i].to(device) for i in range(T)]
    weights = [torch.ones(c.shape[0], device=device) / c.shape[0] for c in clouds]
    total_N = sum(c.shape[0] for c in clouds)
    for i in range(T):
        weights[i] *= clouds[i].shape[0] / total_N
    # Initialize alphas
    alphas = [torch.zeros(c.shape[0], device=device) for c in clouds]
    kernels = [
        torch.exp(-torch.cdist(clouds[i], clouds[i + 1]) ** 2 / (2 * eps))
        for i in range(T - 1)
    ]

    # IPFP loop
    for _ in range(max_iter):
        max_err = 0.0
        for i in range(T):
            qi = torch.zeros_like(alphas[i])
            if i > 0:
                qi += (
                    kernels[i - 1]
                    * torch.exp(alphas[i - 1][:, None] + alphas[i][None, :])
                ).sum(0)
            if i < T - 1:
                qi += (
                    kernels[i] * torch.exp(alphas[i][:, None] + alphas[i + 1][None, :])
                ).sum(1)
            pi = weights[i]

            max_err = max(max_err, (qi - pi).abs().max())
            print(f"max marginal error at slice {i}: {max_err:.2e}")

            alphas[i] += eps * (torch.log(pi + 1e-12) - torch.log(qi + 1e-12))

    # build pairwise couplings that that share alpha_i
    plans = []
    for i in range(T - 1):
        log_pi = (
            alphas[i][:, None]
            + alphas[i + 1][None, :]
            - torch.cdist(clouds[i], clouds[i + 1]) ** 2 / (2 * eps)
        )
        Pi = torch.exp(log_pi)
        plans.append(Pi / Pi.sum())
    return plans


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
            self.ot_fn = partial(
                pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m
            )
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
