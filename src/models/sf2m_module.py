import copy
import math
import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torchsde
from pytorch_lightning import LightningDataModule, LightningModule
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchvision import transforms

from .components.distribution_distances import compute_distribution_distances
from .components.optimal_transport import EntropicOTFM
from .components.plotting import (
    plot_comparison_heatmaps,
    plot_paths,
    plot_samples,
    plot_trajectory,
    store_trajectories,
)
from .components.schedule import ConstantNoiseScheduler, NoiseScheduler
from .components.solver import FlowSolver


class SF2MLitModule(LightningModule):
    """SF2M Module for training generative models and learning structure."""

    def __init__(
        self,
        net: Any,
        score_net: Any,
        corr_net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        partial_solver: FlowSolver,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma: Optional[NoiseScheduler] = None,
        sigma_min: float = 0.1,
        alpha: float = 0.1,
        corr_strength: float = 1e-3,
        reg: float = 1e-4,
        batch_size: int = 64,
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
        test_nfe: int = 100,
        plot: bool = False,
        nice_name: Optional[str] = "SF2M",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "score_net",
                "corr_net",
                "optimizer",
                "scheduler",
                "datamodule",
                "partial_solver",
            ],
            logger=False,
        )

        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        if hasattr(datamodule, "dim"):
            self.dim = datamodule.dim
        elif hasattr(datamodule, "dims"):
            self.dim = datamodule.dims
        else:
            raise NotImplementedError("Datamodule must have either dim or dims")
        self.net = net(dim=self.dim)
        self.score_net = score_net(dim=self.dim)
        self.corr_net = corr_net(dim=self.dim)

        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        self.alpha = alpha
        self.corr_strength = corr_strength
        self.reg = reg
        self.batch_size = batch_size
        self.sigma = sigma
        if sigma is None:
            self.sigma = ConstantNoiseScheduler(sigma_min)
        self.criterion = torch.nn.MSELoss()
        self.otfms = []
        self.cond_matrix = []

    def build_ko_mask(self, dim: int, ko_idx: int):
        mask = torch.ones((dim, dim), dtype=np.float32)
        if ko_idx is not None:
            mask[:, ko_idx] = 0.0
            mask[ko_idx, ko_idx] = 1.0
        return mask

    def build_cond_matrix(self, batch_size, ko, dim):
        cond_matrix = torch.zeros(batch_size, dim)
        if ko is not None:
            cond_matrix[:, ko] = 1
        return cond_matrix

    def forward(self, x, t, cond=None):
        """Forward pass for the flow model + correction and score."""
        v = self.net(t, x) + self.corr_net(t, x)
        s = self.score_net(t, x, cond)
        return v, s

    def training_step(self, batch, batch_idx):
        """Given a batch from your DataModule or custom dataset, do:

        1) sample bridging (X, s_true, u_true, t, t_orig) 2) run the flow + score networks 3)
        compute losses 4) log them
        """
        # 1. Randomly select which dataset we are using in this iteration
        ds_idx = np.random.randint(0, len(self.otfms))
        model = self.otfms[ds_idx]
        cond_vector = self.cond_matrix[ds_idx]

        # 2. Sample bridging from your OTFM or bridging method
        _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
            batch_size=self.batch_size, skip_time=self.skip_time
        )

        # 3. Possibly move them onto correct device
        _x = _x.to(self.device)
        _s = _s.to(self.device)
        _u = _u.to(self.device)
        _t = _t.to(self.device)
        _t_orig = _t_orig.to(self.device)
        cond_expanded = None
        if cond_vector is not None:
            cond_expanded = cond_vector[: len(_x)].to(self.device)

        # 4. Forward pass:
        #    s_fit = func_s(_t, _x, cond_expanded)
        #    v_fit = func_v(_t, _x) ...
        #    or adopt the logic you have in your snippet:
        if self.current_epoch <= 0 and self.global_step < 500:
            # example
            v_fit = self.net(_t, _x) - model.sigma**2 / 2 * self.score_net(_t, _x, cond_expanded)
        else:
            v_fit = (
                self.net(_t, _x)
                + self.corr_net(_t, _x)
                - model.sigma**2 / 2 * self.score_net(_t, _x, cond_expanded)
            )
        s_fit = self.score_net(_t, _x, cond_expanded)

        # 5. Calculate the losses (score + flow + reg)
        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * model.dt - _u) ** 2)

        L_reg_v = self.net.l2_reg() + self.net.fc1_reg()
        L_reg_corr = sum(p.pow(2).sum() for p in self.v_correction.parameters())

        if self.current_epoch < 1 and self.global_step < 100:
            loss = self.alpha * L_score
        elif self.global_step <= 500:
            loss = self.alpha * L_score + (1 - self.alpha) * L_flow + self.reg * L_reg_v
        else:
            loss = (
                self.alpha * L_score
                + (1 - self.alpha) * L_flow
                + self.reg * L_reg_v
                + self.corr_strength * L_reg_corr
            )

        # 6. Log
        self.log("train/score_loss", L_score, on_step=True, on_epoch=True)
        self.log("train/flow_loss", L_flow, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """If you want a simple bridging check during validation, do something akin to
        training_step but without gradient."""
        # e.g. pick a dataset, sample bridging, measure the same losses
        # For brevity, let's do a no-op or a small example
        ds_idx = 0
        model = self.otfms[ds_idx]
        with torch.no_grad():
            _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(
                batch_size=self.batch_size, skip_time=self.skip_time
            )
            # Move to device, compute v_fit, s_fit, compute a validation loss, etc.

        # Return or log your validation losses
        return {}

    def test_step(self, batch, batch_idx):
        """Evaluate on test data (e.g. held-out time, etc.).

        Could do the same bridging logic or call your trajectory simulation code.
        """
        return {}

    def configure_optimizers(self):
        """Return your optimizer (and optional LR scheduler)."""
        # For example, AdamW over all parameters
        params = (
            list(self.net.parameters())
            + list(self.score_net.parameters())
            + list(self.corr_net.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        return optimizer
