import torch
import ot as pot
from functools import partial
import math


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, flow_model, score_model, sigma=1.0, ko_condition=None):
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma
        self.ko_condition = ko_condition

    def f(self, t, y):
        """Drift function"""
        if len(t.shape) == len(y.shape):
            t_input = t
        else:
            t_input = t.repeat(y.shape[0])[:, None]

        # Pass knockout condition only if it exists
        args = [y, t_input]
        if self.ko_condition is not None:
            args.append(self.ko_condition)

        drift = self.flow_model(*args)
        score = self.score_model(*args)
        return drift + score

    def g(self, t, y):
        """Diffusion function"""
        return torch.ones_like(y) * self.sigma


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, ko_condition=None):
        super().__init__()
        self.model = model
        self.ko_condition = ko_condition

    def forward(self, t, x, *args, **kwargs):
        t_input = t.repeat(x.shape[0])[:, None]

        # Pass knockout condition only if it exists
        args = [x, t_input]
        if self.ko_condition is not None:
            args.append(self.ko_condition)

        return self.model(*args)


def wasserstein(
    x0: torch.Tensor, x1: torch.Tensor, method: str = "exact", reg: float = 0.05
) -> float:
    """
    Compute Wasserstein-2 distance between two distributions.
    """
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
