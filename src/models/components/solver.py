"""solver.py.

Implements ODE and SDE solvers for the model.

Joins the torchdyn and torchsde libraries.
"""

from math import prod

import torch
import torchsde
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
