"""solver.py.

Implements ODE and SDE solvers for the model.

Joins the torchdyn and torchsde libraries.
"""

import math
from functools import partial
from math import prod

import ot as pot
import geomloss
from ot.backend import get_backend
import torch
import torch.nn as nn
import torchsde
from torchdiffeq import odeint
from torchdyn.core import NeuralODE


class TorchSDE(torch.nn.Module):
    def __init__(
        self,
        sigma,
        forward_sde_drift,
        backward_sde_drift,
        noise_type,
        sde_type,
        reverse=False,
    ):
        super().__init__()
        self.sigma = sigma
        self.forward_sde_drift = forward_sde_drift
        self.backward_sde_drift = backward_sde_drift
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.reverse = reverse

    def f(self, t, y):
        if self.reverse:
            return self.backward_sde_drift(1 - t, y)
        return self.forward_sde_drift(t, y)

    def g(self, t, y):
        return self.sigma(t) * torch.ones_like(y)

    def h(self, t, y):
        return torch.zeros_like(y)


class FlowSolver(torch.nn.Module):
    def __init__(
        self,
        vector_field,
        corr_field,
        dim,
        dataset_idx,
        cond_vector,
        augmentations=None,
        score_field=None,
        sigma=None,
        ode_solver="euler",
        sde_solver="euler",
        sde_noise_type="diagonal",
        sde_type="ito",
        dt=0.01,
        dt_min=1e-3,
        atol=1e-5,
        rtol=1e-5,
        **kwargs,
    ):
        """Initializes the solver.

        Merges Torchdyn with torchsde.
        Args:
            vector_field (torch.nn.Module): The vector field of the ODE.
            augmentations (torch.nn.Module): The augmentations of the ODE. Not used for SDE
            score_field (torch.nn.Module): The score field of the SDE. Score field is -g(t)^2 / 2 \nabla log p(x(t)).
            sigma (noise_schedule): The noise schedule of the SDE.
            reverse (bool): Whether to reverse the SDE no effect on ODE.
            ode_solver (str): The ODE solver to use.
            sde_solver (str): The SDE solver to use.
            sde_noise_type (str): The noise type of the SDE.
            dt (float): The fixed time step of the ODE solver.
            atol (float): The absolute tolerance of the ODE solver.
            rtol (float): The relative tolerance of the ODE solver.
        """
        super().__init__()
        self.net = vector_field
        self.corr_net = corr_field
        self.data_idx = dataset_idx
        self.cond_vector = cond_vector
        self.dim = dim
        self.augmentations = augmentations
        self.score_net = score_field
        self.separate_score = score_field is not None
        if sigma is None:
            self.sigma = lambda t: 0.1
        else:
            self.sigma = sigma
        self.ode_solver = ode_solver
        self.sde_solver = sde_solver
        self.sde_noise_type = sde_noise_type
        self.sde_type = sde_type
        self.dt = dt
        self.dt_min = dt_min
        self.atol = atol
        self.rtol = rtol
        self.nfe = 0
        self.kwargs = kwargs
        self.is_image = not isinstance(self.dim, int)
        if self.is_image:
            self.flat_dim = prod(dim)

    def forward_flow_and_score(self, t, x, only_flow=False):
        if self.is_image:
            x = x.reshape(-1, *self.dim)
        v_out = (
            self.net(t, x.unsqueeze(1), self.data_idx).squeeze(1) if self.net is not None else 0
        )
        c_out = self.corr_net(t, x) if self.corr_net is not None else 0
        s_out = None
        if self.score_net is not None and not only_flow:
            s_out = self.score_net(t, x, self.cond_vector)
        if only_flow:
            return v_out + c_out

        return v_out, c_out, s_out

    def forward_sde_drift(self, t, x):
        """Computes the forwards drift of the SDE."""
        self.nfe += 1
        vt, ct, st = self.forward_flow_and_score(t, x)
        drift = vt + ct
        if st is not None:
            alpha = self.sigma(t) ** 2 / 2
            drift = drift + alpha * st
        return drift

    def backward_sde_drift(self, t, x):
        """Computes the backwards drift of the SDE."""
        self.nfe += 1
        vt, ct, st = self.forward_flow_and_score(t, x)
        drift = -(vt + ct)
        if st is not None:
            alpha = self.sigma(t) ** 2 / 2
            drift = drift + alpha * st
        return drift

    def forward_ode_drift(self, t, x):
        """Computes the forwards drift of the ODE."""
        self.nfe += 1
        return self.forward_flow_and_score(t, x, only_flow=True)

    def backward_ode_drift(self, t, x):
        """Computes the backwards drift of the ODE."""
        self.nfe += 1
        return -self.forward_flow_and_score(t, x, only_flow=True)

    def ode_drift(self, reverse=False):
        return self.forward_ode_drift if not reverse else self.backward_ode_drift

    def sde_drift(self, reverse=False):
        return self.forward_sde_drift if not reverse else self.backward_sde_drift

    def flat_wrapper(self, func):
        if not isinstance(self.dim, int):

            def wrap(t, x):
                x = x.reshape(-1, self.dim)
                y = func(t, x)
                y = y.reshape(-1, self.flat_dim)

    def sdeint(self, x0, t_span, logqp=False, adaptive=False, reverse=False):
        self.nfe = 0
        sde = TorchSDE(
            self.sigma,
            self.forward_sde_drift,
            self.backward_sde_drift,
            self.sde_noise_type,
            self.sde_type,
            reverse,
        )
        if self.is_image:
            x0 = x0.reshape(-1, self.flat_dim)
        traj = torchsde.sdeint(
            sde,
            x0,
            t_span,
            method=self.sde_solver,
            dt=self.dt,
            rtol=self.rtol,
            atol=self.atol,
            logqp=logqp,
            adaptive=adaptive,
        )
        if self.is_image:
            traj = traj.reshape(traj.shape[0], traj.shape[1], *self.dim)
        return traj

    def odeint(self, x0, t_span):
        """Computes the ODE trajectory.

        Relies on the torchdyn library to compute the ODE trajectory and to handle reverse t_spans.
        """
        self.nfe = 0

        node = NeuralODE(
            self.forward_ode_drift,
            solver=self.ode_solver,
            atol=self.atol,
            rtol=self.rtol,
            return_t_eval=False,
        )
        return node(x0, t_span)

    def get_nfe(self):
        return self.nfe

    def reset_nfe(self):
        self.nfe = 0


class DSBMFlowSolver(FlowSolver):
    """Same as SF2M except interprets net as forward and score_net as backward SDE drifts."""

    def forward_flow_and_score(self, t, x, only_forward=False, only_backward=False):
        if self.is_image:
            x = x.reshape(-1, *self.dim)
        if only_forward:
            fvt = self.net(t, x)
            return fvt.reshape(-1, self.flat_dim) if self.is_image else fvt
        if only_backward:
            return self.score_net(t, x)
        if self.separate_score:
            fvt, bvt = self.net(t, x), self.score_net(t, x)
        else:
            fbvt = self.net(t, x)
            # if using a single network split the network in two along the [1] dimension
            # batch, *(dims)
            split_idx = fbvt.shape[1] // 2
            fvt, bvt = fbvt[..., :split_idx], fbvt[..., split_idx:]
        if self.is_image:
            fvt = fvt.reshape(-1, self.flat_dim)
            bvt = bvt.reshape(-1, self.flat_dim)
        return fvt, bvt

    def forward_sde_drift(self, t, x):
        """Computes the forwards drift of the SDE."""
        self.nfe += 1
        return self.forward_flow_and_score(t, x, only_forward=True)

    def backward_sde_drift(self, t, x):
        """Computes the backwards drift of the SDE."""
        self.nfe += 1
        return self.forward_flow_and_score(t, x, only_backward=True)

    def forward_ode_drift(self, t, x):
        """Computes the forwards drift of the ODE."""
        self.nfe += 1
        fvt, bvt = self.forward_flow_and_score(t, x)
        return (fvt - bvt) / 2

    def backward_ode_drift(self, t, x):
        """Computes the backwards drift of the ODE."""
        self.nfe += 1
        fvt, bvt = self.forward_flow_and_score(t, x)
        return -(fvt - bvt) / 2


class TrajectorySolver(nn.Module):
    def __init__(
        self,
        sigma=1.0,
        T=1.0,
        device="cpu",
    ):
        """
        Parameters:
            flow_model: The neural network representing the flow field.
            corr_model: The network representing the correction term.
            score_model: The network returning the score (gradient of log-density).
            sigma (float): Noise level (diffusion coefficient).
            T (float): Total time such that dt = 1 / T.
            device (str): Device to use ("cpu" or "cuda").
            use_sde (bool): Whether to simulate using an SDE or ODE.
            cond_vector: Optional conditional vector used by score_model.
            dataset_idx: Identifier (or index) for the dataset.
        """
        super().__init__()
        self.sigma = sigma
        self.T = T
        self.dt = 1.0 / T
        self.device = device

    def simulate(
        self,
        x0,
        start_time,
        end_time,
        flow_model,
        score_model,
        corr_model,
        n_times=400,
        cond_vector=None,
        dataset_idx=None,
        use_sde=False,
    ):
        """Simulate a trajectory starting from x0.

        Parameters:
            x0 (torch.Tensor): The initial state tensor.
            start_time (float): Starting time index (in discrete units).
            end_time (float): Ending time index (in discrete units).
            n_times (int): Number of time points to evaluate the trajectory.

        Returns:
            trajectory (torch.Tensor): The simulated trajectory (moved to CPU).
        """
        x0 = x0.to(self.device)
        t_start = start_time * self.dt
        t_end = end_time * self.dt
        ts = torch.linspace(t_start, t_end, n_times, device=self.device)

        if use_sde:
            # Define an inner class for the SDE dynamics.
            class FlowSDE(nn.Module):
                def __init__(
                    self, flow_model, corr_model, score_model, sigma, dataset_idx, cond_vector
                ):
                    super().__init__()
                    self.flow_model = flow_model
                    self.corr_model = corr_model
                    self.score_model = score_model
                    self.sigma = sigma
                    self.dataset_idx = dataset_idx
                    self.cond_vector = cond_vector
                    self.noise_type = "diagonal"
                    self.sde_type = "ito"

                def f(self, t, x):
                    t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
                    flow_out = self.flow_model(t_batch, x.unsqueeze(1), self.dataset_idx).squeeze(
                        1
                    )
                    corr_out = self.corr_model(t_batch.unsqueeze(1), x)
                    # Note: score_model is computed here but not used in the drift.
                    score_out = self.score_model(t_batch, x, self.cond_vector)
                    return flow_out + corr_out

                def g(self, t, x):
                    return self.sigma * torch.ones_like(x)

            sde = FlowSDE(
                self.flow_model,
                self.corr_model,
                self.score_model,
                self.sigma,
                self.dataset_idx,
                self.cond_vector,
            )
            with torch.no_grad():
                trajectory = torchsde.sdeint(sde, x0, ts, method="euler")
        else:

            def ode_func(t, x):
                t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
                flow_out = self.flow_model(t_batch, x.unsqueeze(1), self.dataset_idx).squeeze(1)
                corr_out = self.corr_model(t_batch.unsqueeze(1), x)
                score_out = self.score_model(t_batch, x, self.cond_vector)
                return flow_out + corr_out - (self.sigma**2 / 2) * score_out

            with torch.no_grad():
                trajectory = odeint(ode_func, x0, ts, method="dopri5")

        return trajectory.cpu()


def wasserstein(
    x0: torch.Tensor, x1: torch.Tensor, method: str = "exact", reg: float = 0.05
) -> float:
    """Compute Wasserstein-2 distance between two distributions."""
    # Set up the OT function
    if method == "exact":
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get uniform weights for the samples
    a = pot.unif(x0.shape[0])
    b = pot.unif(x1.shape[0])

    # Reshape if needed
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    # Compute cost matrix (squared Euclidean distance)
    M = torch.cdist(x0, x1) ** 2

    # Compute Wasserstein distance
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)

    # Return square root for W2 distance
    return math.sqrt(ret)

def rbf_kernel(X, Y, gamma=None):
    if X.dim() > 2: X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2: Y = Y.reshape(Y.shape[0], -1)
    d = X.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    dist_sq = torch.cdist(X, Y)**2
    K = torch.exp(-gamma * dist_sq)
    return K, gamma

def mmd_squared(X, Y, kernel=rbf_kernel, sigma_list=None, **kernel_args):
    if X.dim() > 2: X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2: Y = Y.reshape(Y.shape[0], -1)
    
    if sigma_list is None:
        sigma_list = [0.01, 0.1, 1, 10, 100]
    
    mmd_values = []
    
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        
        K_XX, _ = kernel(X, X, gamma=gamma)
        K_YY, _ = kernel(Y, Y, gamma=gamma)
        K_XY, _ = kernel(X, Y, gamma=gamma)
        
        term1 = K_XX.mean()
        term2 = K_YY.mean()
        term3 = K_XY.mean()
        
        mmd2 = term1 + term2 - 2 * term3
        mmd_values.append(mmd2.clamp(min=0))
    
    avg_mmd = torch.stack(mmd_values).mean().item()
    return avg_mmd

def emd_samples(x, y, x_w = None, y_w = None):
    C = pot.utils.euclidean_distances(x, y, squared=True)
    nx = get_backend(x, y)
    p = nx.full((x.shape[0], ), 1/x.shape[0]) if x_w is None else x_w / x_w.sum()
    q = nx.full((y.shape[0], ), 1/y.shape[0]) if y_w is None else y_w / y_w.sum()
    return pot.emd2(p, q, C)

def sinkhorn_divergence(x, y, x_w = None, y_w = None, reg = 1.0):
    # p = np.full((x.shape[0], ), 1/x.shape[0]) if x_w is None else x_w / x_w.sum()
    # q = np.full((y.shape[0], ), 1/y.shape[0]) if y_w is None else y_w / y_w.sum()
    # return ot.bregman.empirical_sinkhorn_divergence(x, y, reg, a = p, b = q)
    p = torch.full((x.shape[0], ), 1/x.shape[0]) if x_w is None else x_w / x_w.sum()
    q = torch.full((y.shape[0], ), 1/y.shape[0]) if y_w is None else y_w / y_w.sum()
    loss = geomloss.SamplesLoss(loss = 'sinkhorn')
    return loss(p, x, q, y)

def energy_distance(x, y, x_w = None, y_w = None):
    nx = get_backend(x, y)
    x_w = nx.full((x.shape[0], ), 1/x.shape[0]) if x_w is None else x_w / x_w.sum()
    y_w = nx.full((y.shape[0], ), 1/y.shape[0]) if y_w is None else y_w / y_w.sum()
    xy=nx.dot(x_w, pot.utils.euclidean_distances(x, y, squared=False) @ y_w)
    xx=nx.dot(x_w, pot.utils.euclidean_distances(x, x, squared=False) @ x_w)
    yy=nx.dot(y_w, pot.utils.euclidean_distances(y, y, squared=False) @ y_w)
    return 2*xy-xx-yy

def energy_distance_paths(x, y):
    return energy_distance(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))

def emd_paths(x, y):
    return emd_samples(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))

def simulate_trajectory(
    flow_model,
    corr_model,  # This can now be None or a dummy module
    score_model,
    x0,
    dataset_idx,
    start_time,
    end_time,
    n_times=400,
    sigma=1.0,
    device="cpu",
    use_sde=False,
    cond_vector=None,
    T: int = 5,
):
    x0 = x0.to(device)
    dt = 1 / T
    t_start = start_time * dt
    t_end = end_time * dt
    ts = torch.linspace(t_start, t_end, n_times, device=device)

    if use_sde:
        class FlowSDE(torch.nn.Module):
            def __init__(self, flow_model, corr_model, score_model, sigma):
                super().__init__()
                self.flow_model = flow_model
                self.corr_model = corr_model
                self.score_model = score_model
                self.sigma = sigma
                self.noise_type = "diagonal"
                self.sde_type = "ito"

            def f(self, t, x):
                t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
                if dataset_idx is not None:
                    flow_out = self.flow_model(t_batch, x.unsqueeze(1), dataset_idx).squeeze(1)
                else:
                    flow_out = self.flow_model(t_batch, x.unsqueeze(1)).squeeze(1)
                corr_out = torch.zeros_like(x)
                if self.corr_model is not None and hasattr(self.corr_model, 'parameters') and list(self.corr_model.parameters()):
                    corr_out = self.corr_model(t_batch.unsqueeze(1), x)
                out = flow_out + corr_out
                if out.dim() == 4:                            # [E, B, M, D]
                    out = out.mean(dim=(0, 2))                # → [B, D]
                elif out.dim() == 3:                          # [E, B, D]
                    out = out.mean(dim=0)                     # → [B, D]
                elif out.dim() == 2:                          # already [B, D]
                    pass
                else:
                    raise RuntimeError(f"Unexpected flow_model output shape: {out.shape}")
                return out

            def g(self, t, x):
                return 0.1 * torch.ones_like(x)

        sde = FlowSDE(flow_model, corr_model, score_model, sigma)
        with torch.no_grad():
            trajectory = torchsde.sdeint(sde, x0, ts, method="euler")

    else:
        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
            if dataset_idx is not None:
                flow_out = flow_model(t_batch, x.unsqueeze(1), dataset_idx).squeeze(1)
            else:
                flow_out = flow_model(t_batch, x.unsqueeze(1)).squeeze(1)
            corr_out = torch.zeros_like(x)
            if corr_model is not None and hasattr(corr_model, 'parameters') and list(corr_model.parameters()):
                corr_out = corr_model(t_batch.unsqueeze(1), x)
            score_out = score_model(t_batch, x, cond_vector)
            out = flow_out + (sigma**2 / 2) * score_out
            if out.dim() == 4:                            # [E, B, M, D]
                out = out.mean(dim=(0, 2))                # → [B, D]
            elif out.dim() == 3:                          # [E, B, D]
                out = out.mean(dim=0)                     # → [B, D]
            elif out.dim() == 2:                          # already [B, D]
                pass
            else:
                raise RuntimeError(f"Unexpected flow_model output shape: {out.shape}")
            return out

        with torch.no_grad():
            trajectory = odeint(ode_func, x0, ts, method="euler")

    return trajectory.cpu()
