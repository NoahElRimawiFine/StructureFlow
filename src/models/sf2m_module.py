import copy
import math
import os
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchsde
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchvision import transforms

from src.datamodules.grn_datamodule import TrajectoryStructureDataModule

from .components.distribution_distances import compute_distribution_distances
from .components.optimal_transport import EntropicOTFM
from .components.plotting import (
    plot_aupr_curve,
    plot_comparison_heatmaps,
    plot_paths,
    plot_samples,
    plot_trajectory,
    store_trajectories,
)
from .components.schedule import ConstantNoiseScheduler, NoiseScheduler
from .components.solver import FlowSolver

seed = 42
seed_everything(seed, workers=True)


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
        ot_sampler: Optional[Union[str, Any]] = EntropicOTFM,
        sigma: Optional[NoiseScheduler] = None,
        sigma_min: float = 0.1,
        alpha: float = 0.1,
        corr_strength: float = 1e-3,
        reg: float = 1e-4,
        lr: float = 1e-3,
        batch_size: int = 64,
        skip_time=None,
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

        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        if hasattr(datamodule, "dim"):
            self.dim = datamodule.dim
        elif hasattr(datamodule, "dims"):
            self.dim = datamodule.dims
        else:
            raise NotImplementedError("Datamodule must have either dim or dims")
        self.net = net
        self.score_net = score_net
        self.corr_net = corr_net

        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        self.alpha = alpha
        self.corr_strength = corr_strength
        self.reg = reg
        self.batch_size = batch_size
        self.skip_time = skip_time
        self.sigma = sigma
        if sigma is None:
            self.sigma = ConstantNoiseScheduler(sigma_min)
        self.criterion = torch.nn.MSELoss()
        self.otfms = []
        self.cond_matrix = []
        self.automatic_optimization = False

    def build_knockout_mask(self, dim: int, ko_idx: int):
        """Build a knockout mask for a given dimension and knockout index.

        Returns a [dim, dim] tensor that is one-hot encoded per the original logic.
        """
        mask = torch.ones((dim, dim), dtype=torch.float32)
        if ko_idx is not None:
            mask[:, ko_idx] = 0.0
            mask[ko_idx, ko_idx] = 1.0
        return mask

    def build_cond_matrix(self, batch_size: int, dim: int, kos: list):
        """Build a list of conditional matrices.

        For each dataset (indexed by i), create a [batch_size, dim] matrix where, if the dataset
        has a knockout (kos[i] is not None), the i-th column is set to 1.
        """
        conditionals = []
        for i, ko in enumerate(kos):
            cond_matrix = torch.zeros(batch_size, dim)
            if ko is not None:
                cond_matrix[:, i] = 1
            conditionals.append(cond_matrix)
        return conditionals

    def build_entropic_otfms(self, adatas: list, T: int, sigma: float, dt: float):
        """Build a list of optimal transport flow models (OTFMs), one per dataset.

        Each model is constructed using the provided ot_sampler.
        """
        otfms = []
        for adata in adatas:
            x_tensor = torch.tensor(adata.X, dtype=torch.float32)
            t_idx = torch.tensor(adata.obs["t"], dtype=torch.long)
            model = self.ot_sampler(
                x=x_tensor,
                t_idx=t_idx,
                dt=dt,
                sigma=sigma,
                T=T,
                dim=x_tensor.shape[1],
                device=torch.device("cpu"),
            )
            otfms.append(model)
        return otfms

    def mlp_l2_reg(self, mlp):
        l2_sum = 0.0
        for param in mlp.parameters():
            l2_sum += torch.sum(param**2)
        return l2_sum

    def on_fit_start(self):
        """Called once when training begins.

        Here we access the datamodule (which must have been fully set up) to:
         - Build the knockout masks.
         - Build the conditional matrices.
         - Build the list of OTFMs.
        We then assign the knockout masks to the flow network.
        """
        dm = self.trainer.datamodule

        n = dm.adatas[0].X.shape[1]

        self.cond_matrix = self.build_cond_matrix(self.hparams.batch_size, n, dm.kos)
        self.otfms = self.build_entropic_otfms(dm.adatas, T=dm.T, sigma=1.0, dt=1 / dm.T)

    def on_validation_start(self):
        dm = self.trainer.datamodule
        val_adatas = dm.get_subset_adatas(split="val")
        self.otfms_val = self.build_entropic_otfms(val_adatas, T=5, sigma=1.0, dt=0.2)
        self.cond_matrix = self.build_cond_matrix(
            self.hparams.batch_size, dm.adatas[0].X.shape[1], dm.kos
        )

    def on_test_start(self):
        dm = self.trainer.datamodule
        test_adatas = dm.get_subset_adatas(split="test")
        self.otfms_test = self.build_entropic_otfms(test_adatas, T=5, sigma=1.0, dt=0.2)
        self.cond_matrix = self.build_cond_matrix(
            self.hparams.batch_size, dm.adatas[0].X.shape[1], dm.kos
        )

    def forward(self, x, t, cond=None):
        """Forward pass for the flow model + correction and score."""
        v = self.net(t, x) + self.corr_net(t, x)
        s = self.score_net(t, x, cond)
        return v, s

    def training_step(self, batch, batch_idx):
        """
        1. Randomly selects a dataset index.
        2. Samples a batch of bridging flows from the corresponding OTFM.
        3. Moves sampled data to the correct device.
        4. Expands the conditional vector (using the original repetition logic).
        5. Reshapes inputs to match the network expectations.
        6. Computes network outputs (v_fit and s_fit) with a conditional branch: for early iterations, the
           correction network is omitted.
        7. Computes the loss terms (score, flow, regularization) exactly as before.
        8. Logs the losses.
        """
        optimizer = self.optimizers()

        # (1) Randomly select dataset index.
        ds_idx = np.random.randint(0, len(self.otfms))
        otfm_model = self.otfms[ds_idx]
        cond_vector = self.cond_matrix[ds_idx]

        # (2) Sample bridging flows using the OTFM.
        _x, _s, _u, _t, _t_orig = otfm_model.sample_bridges_flows(
            batch_size=self.batch_size, skip_time=self.skip_time
        )
        optimizer.zero_grad()

        # (3) Move sampled tensors to the current device.
        _x = _x.to(self.device)
        _s = _s.to(self.device)
        _u = _u.to(self.device)
        _t = _t.to(self.device)
        _t_orig = _t_orig.to(self.device)

        # (4) Expand the conditional vector to match the batch size.
        B = _x.shape[0]
        cond_expanded = cond_vector.repeat(B // self.batch_size + 1, 1)[:B].to(self.device)

        # (5) Reshape inputs: unsqueeze to add a time dimension.
        s_input = _x.unsqueeze(1)  # [B, 1, dim]
        v_input = _x.unsqueeze(1)
        t_input = _t.unsqueeze(1)

        # (6) Compute network outputs:
        s_fit = self.score_net(_t, _x, cond_expanded).squeeze(1)

        if self.global_step <= 500:
            v_fit = self.net(t_input, v_input).squeeze(
                1
            ) - otfm_model.sigma**2 / 2 * self.score_net(_t, _x, cond_expanded)
        else:
            v_fit = self.net(t_input, v_input).squeeze(1) + self.corr_net(_t, _x)
            v_fit = v_fit - otfm_model.sigma**2 / 2 * self.score_net(_t, _x, cond_expanded)

        # (7) Compute losses as in the original code.
        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * otfm_model.dt - _u) ** 2)

        L_reg = self.net.l2_reg() + self.net.fc1_reg()
        L_reg_corr = self.mlp_l2_reg(self.corr_net)
        if self.global_step < 100:
            loss = self.alpha * L_score
        elif self.global_step >= 100 and self.global_step <= 500:
            loss = self.alpha * L_score + (1 - self.alpha) * L_flow + self.reg * L_reg
        else:
            loss = (
                self.alpha * L_score
                + (1 - self.alpha) * L_flow
                + self.reg * L_reg
                + self.corr_strength * L_reg_corr
            )

        # (8) Log losses.
        self.log("train/score_loss", L_score, on_step=True, on_epoch=True)
        self.log("train/flow_loss", L_flow, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/reg_loss", L_reg, on_step=True, on_epoch=True)
        self.log("train/reg_corr_loss", L_reg_corr, on_step=True, on_epoch=True)

        with torch.no_grad():
            if self.global_step % 100 == 0:
                print(
                    f"step={self.global_step}, dataset={ds_idx}, L_score={L_score.item():.4f}, L_flow={L_flow.item():.4f}, "
                    f"NGM_Reg={L_reg.item():.4f}, MLP_Reg={L_reg_corr.item():.4f}"
                )

        loss.backward()
        optimizer.step()

        # proximal(s.fc1.weight, s.dims, lam=s.GL_reg, eta=0.01)
        self.proximal(self.net.fc1.weight, self.net.dims, lam=self.net.GL_reg, eta=0.01)

        return loss

    def proximal(self, w, dims, lam=0.1, eta=0.1):
        """Applies a proximal update to the weight tensor w.

        This mimics the proximal update in your original training loop.
        """
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    def validation_step(self, batch, batch_idx):
        """If you want a simple bridging check during validation, do something akin to
        training_step but without gradient."""
        ds_idx = 0
        model = self.otfms[ds_idx]
        with torch.no_grad():
            _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(batch_size=self.batch_size)
            # Move to device, compute v_fit, s_fit, compute a validation loss, etc.

        # Return or log your validation losses
        return {}

    def test_step(self, batch, batch_idx):
        """Evaluate on test data (e.g. held-out time, etc.).

        Could do the same bridging logic or call your trajectory simulation code.
        """
        return {}

    def on_train_epoch_end(self):
        try:
            W_v = self.net.causal_graph(w_threshold=0.0)
            if isinstance(W_v, torch.Tensor):
                W_v = W_v.detach().cpu().numpy()
        except AttributeError:
            self.logger.warning(
                "The network does not implement causal_graph(); skipping plotting."
            )
            return

        def maskdiag(A):
            return A * (1 - np.eye(A.shape[0]))  # Shape should match dim

        # 3. Create the plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(maskdiag(W_v), cmap="Reds")
        ax.invert_yaxis()
        ax.set_title("Causal Graph (from MLPODEF)")
        fig.colorbar(cax)

        if self.logger is not None:
            self.logger.experiment.add_figure("Causal_Graph", fig, global_step=self.global_step)
            plt.close(fig)
        else:
            plt.show()

        # Log a dummy metric for tracking visualization steps
        self.log("epoch/plot_causal_graph", 1)

        fig_aupr = plot_aupr_curve(
            maskdiag(self.trainer.datamodule.true_matrix), maskdiag(W_v), prefix="val"
        )
        if self.logger is not None:
            self.logger.experiment.add_figure("AUPR_Curve", fig_aupr, global_step=self.global_step)
        plt.close(fig_aupr)

        self.log("epoch/plot_aupr", 1)

    def forward_sde_eval(self, ts, x0, x_rest, outputs, prefix="val"):
        """
        Evaluate an SDE solution piecewise from time i->i+1 for i in [0..ts-1].
        x0: [batch_size, dim], the initial states
        x_rest: reference states for distribution distances, shape depends on usage
        outputs: leftover from validation or test hooking
        prefix: prefix for logging, e.g. "val" or "test"

        Returns:
          (trajs, full_trajs) the partial final states and the entire piecewise trajectory
        """
        # 1) build a small time range [0..1]
        t_span = torch.linspace(0, 1, 2, device=x0.device)

        # 2) build or reuse partial_solver
        # we assume partial_solver is a function that returns a solver
        solver = self.partial_solver(
            self.net, self.dim, score_field=self.score_net, sigma=self.sigma
        )

        trajs = []
        full_trajs = []
        nfe = 0
        kldiv_total = 0
        x0_tmp = x0.clone()
        # piecewise from i=0..ts-2
        for i in range(ts - 1):
            # integrate from i to i+1 => t_span + i
            traj, kldiv = solver.sdeint(x0_tmp, t_span + i, logqp=True)
            kldiv_total += torch.mean(kldiv[-1])
            x0_tmp = traj[-1]
            trajs.append(traj[-1])
            full_trajs.append(traj)
            nfe += solver.nfe

        full_trajs = torch.cat(full_trajs)

        # measure distribution distance
        if True:
            # compute distribution distances
            # e.g. if x_rest is the reference states for each time i
            names, dists = compute_distribution_distances(trajs, x_rest)
            # prefix them
            names = [f"{prefix}/sde/{name}" for name in names]
            d = dict(zip(names, dists))
            d[f"{prefix}/sde/nfe"] = nfe
            d[f"{prefix}/sde/kldiv"] = kldiv_total
            self.log_dict(d, sync_dist=True)
        return trajs, full_trajs

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        # TODO: finish tomorrow aft.
        pass

    def configure_optimizers(self):
        """Return your optimizer (and optional LR scheduler)."""
        params = (
            list(self.net.parameters())
            + list(self.score_net.parameters())
            + list(self.corr_net.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        return optimizer
