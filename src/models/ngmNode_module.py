from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from geomloss import SamplesLoss
from lightning.pytorch import LightningDataModule, LightningModule
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint_adjoint as odeint


class NGMNodeModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        datamodule: LightningDataModule,
        l1_reg: float = 0.0,
        l2_reg: float = 0.05,
        lr: float = 0.005,
        batch_size: int = 128,
        n_steps_per_transition: int = 2500,
        plot_freq: int = 1000,
        plot: bool = True,
        nice_name: str = "NGM_Node",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "datamodule"])
        self.net = net
        self.datamodule = datamodule

        # Regularization and training hyperparameters.
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.lr = lr
        self.batch_size = batch_size
        self.n_steps_per_transition = n_steps_per_transition
        self.plot_freq = plot_freq
        self.plot = plot
        self.nice_name = nice_name

        # Placeholders to be initialized in on_fit_start.
        self.adatas_list = None  # List of data objects
        self.num_time_bins = None  # Total time bins (t)
        self.num_transitions = None  # t - 1
        self.num_variables = None  # Number of variables (e.g., genes)
        self.transition_times = None  # Always [0.0, 1.0]
        self.sinkhorn_loss = None  # Sinkhorn loss function instance

    def on_fit_start(self):
        """Called once when training starts.

        Retrieves the data from the datamodule and sets up necessary variables.
        """
        # Assume datamodule has been set up with attributes: adatas_list and t.
        dm = self.trainer.datamodule
        self.adatas_list = dm.adatas
        self.num_time_bins = dm.T
        self.num_transitions = self.num_time_bins - 1
        self.num_variables = self.adatas_list[0].shape[1]

        # Create transition times tensor [0.0, 1.0] on the correct device.
        self.transition_times = torch.tensor([0.0, 1.0], device=self.device)
        # Initialize the Sinkhorn loss from geomloss.
        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="auto")
        self.print(
            f"[on_fit_start] Loaded {len(self.adatas_list)} datasets, "
            f"{self.num_time_bins} time bins, {self.num_variables} variables."
        )

    def create_batch_for_transition(self, adata, t_start: int, batch_size: int):
        """Creates a batch for a specific transition from time bin t_start to t_start+1.

        Args:
            adata: An AnnData-like object with:
                   • adata.X (numpy array of expression data)
                   • adata.obs["t"] (time bin info)
            t_start (int): Starting time bin index.
            batch_size (int): Number of samples in the batch.

        Returns:
            Tuple (x0, t, x1):
              - x0: Tensor of states at time t_start, shape [batch_size, num_variables].
              - t: Transition times tensor, shape [2] (always [0.0, 1.0]).
              - x1: Tensor of target states at time t_start+1, shape [batch_size, num_variables].
        """
        # Extract cells from time bin t_start and t_start+1.
        cells_t0 = torch.from_numpy(adata.X[adata.obs["t"] == t_start]).float()
        cells_t1 = torch.from_numpy(adata.X[adata.obs["t"] == t_start + 1]).float()
        n0 = cells_t0.size(0)
        n1 = cells_t1.size(0)
        # Randomly sample indices.
        idx0 = np.random.randint(n0, size=batch_size)
        idx1 = np.random.randint(n1, size=batch_size)
        x0 = cells_t0[idx0]
        x1 = cells_t1[idx1]
        return x0, self.transition_times, x1

    def training_step(self, batch, batch_idx):
        """Performs one training step. The incoming batch is ignored; instead, we randomly sample a
        transition from the data provided by the datamodule.

        Steps:
          1. Randomly select a transition (t_start) and a dataset index.
          2. Create a batch for the transition using create_batch_for_transition.
          3. Unsqueeze x0 to add a time dimension.
          4. Solve the ODE from x0 over the transition times using odeint.
          5. Compute the Sinkhorn loss between the final state and target state x1.
          6. Add L₂ and L₁ regularization from the network's custom methods.
          7. Log the loss and optionally print it.
        """
        # (1) Randomly select a transition and dataset.
        t_start = np.random.randint(0, self.num_transitions)
        ds_idx = np.random.randint(0, len(self.adatas_list))
        adata = self.adatas_list[ds_idx]

        # (2) Create batch: x0, transition_times, x1.
        x0, t_tensor, x1 = self.create_batch_for_transition(adata, t_start, self.batch_size)
        # (3) Add a time dimension to x0: now shape [batch_size, 1, num_variables].
        x0 = x0.unsqueeze(1)
        z0 = x0

        # (4) Solve the ODE using the neural ODE function (self.net).
        #    odeint returns a tensor of shape [num_time_points, batch_size, 1, num_variables].
        z_pred = odeint(self.net, z0, t_tensor)
        # Take the final time output and remove the extra dimension.
        z_pred = z_pred[-1].squeeze(1)  # shape: [batch_size, num_variables]

        # (5) Compute the loss as Sinkhorn loss between prediction and target.
        loss = self.sinkhorn_loss(z_pred, x1)
        # (6) Add L₂ regularization if specified.
        if self.l2_reg != 0:
            loss += self.l2_reg * self.net.l2_reg()
        # (7) Add L₁ regularization if specified.
        if self.l1_reg != 0:
            loss += self.l1_reg * self.net.fc1_reg()

        # (8) Log the loss.
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        if self.plot and (self.global_step % self.plot_freq == 0):
            self.print(f"Step {self.global_step}: loss = {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        # Return or log your validation losses
        return {}

    def test_step(self, batch, batch_idx):
        """Evaluate on test data (e.g. held-out time, etc.)."""
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
        ax.set_title("Causal Graph (from NGM NODE)")
        fig.colorbar(cax)

        if self.logger is not None:
            # Check if it's a wandb logger
            if hasattr(self.logger.experiment, "log") and not hasattr(self.logger.experiment, "add_figure"):
                # It's a wandb logger
                import wandb
                self.logger.experiment.log({"Causal_Graph": wandb.Image(fig)}, step=self.global_step)
            else:
                # Assume it's a TensorBoard logger or something compatible
                self.logger.experiment.add_figure("Causal_Graph", fig, global_step=self.global_step)
            plt.close(fig)
        else:
            plt.show()

        # Log a dummy metric for tracking visualization steps
        self.log("epoch/plot_causal_graph", 1)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Overrides the default optimizer step.

        Calls the provided optimizer_closure to re-run the forward and backward passes, then steps
        the optimizer and applies a proximal update.
        """
        optimizer.step(closure=optimizer_closure)
        # Apply proximal update to the first fully connected layer of the network.
        self.proximal(self.net.fc1.weight, self.net.dims, lam=self.net.GL_reg, eta=0.01)

    def proximal(self, w, dims, lam=0.1, eta=0.1):
        """Applies a proximal update (group-lasso regularization) to the weight tensor w.

        Args:
            w: Weight tensor (flattened shape) from the network's fc1 layer.
            dims: A list/tuple, e.g. [d, hidden, d] representing dimensions.
            lam: Regularization parameter (lambda).
            eta: Scaling parameter.
        """
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)  # Reshape to [d, hidden, d]
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    def configure_optimizers(self):
        """Configure and return the optimizer and learning rate scheduler.

        Uses AdamW and a StepLR scheduler that steps every training step.
        """
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
        # Use "step" interval so the scheduler updates after every training step.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
