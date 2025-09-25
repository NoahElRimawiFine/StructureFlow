import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch

from .components import rf


class ReferenceFittingModule(pl.LightningModule):
    def __init__(self, use_cuda=True, iter=1000):
        super().__init__()
        # Set the device as before.
        self.my_device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.PLT_CELL = 3.5
        self.options = {
            "lr": 0.1,
            "reg_sinkhorn": 0.1,
            "reg_A": 1e-3,
            "reg_A_elastic": 0,
            "iter": iter,
            "ot_coupling": True,
            "optimizer": torch.optim.Adam,
        }
        # Placeholders for the two estimators
        self.estimator = None
        self.estimator_wt = None

        # Add a dummy parameter so that configure_optimizers has something to optimize.
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def fit_model(self, adatas, kos, also_wt=False):
        """Fits the reference model using both knockout and wild-type data.

        This method replicates your original 'train' method.
        """
        # Determine indices for knockouts (all) and wild-type (where ko is None).
        ko_idx = list(range(len(kos)))  # all indices
        wt_idx = [i for i, ko in enumerate(kos) if ko is None]

        # Sort adatas by first gene expression for determinism
        sorted_adatas = []
        for adata in adatas:
            sort_indices = np.argsort(adata.X[:, 0])
            sorted_adata = adata[sort_indices].copy()
            sorted_adatas.append(sorted_adata)

        print("Training reference model with knockouts...")
        self.estimator = rf.Estimator(
            [sorted_adatas[i] for i in ko_idx], [kos[i] for i in ko_idx], **self.options
        )
        self.estimator.fit(print_iter=100, alg="alternating", update_couplings_iter=250)

        if also_wt:
            print("Training reference model with wild type data only...")
            self.estimator_wt = rf.Estimator(
                [sorted_adatas[i] for i in wt_idx],
                [kos[i] for i in wt_idx],
                **self.options,
            )
            self.estimator_wt.fit(
                print_iter=100, alg="alternating", update_couplings_iter=250
            )

    def fit_model_with_holdout(self, adatas, kos, left_out_time, also_wt=False):
        """Fits the reference model using both knockout and wild-type data, taking into account a
        hold-out time.

        Replicates your original 'train_with_holdout' method.
        """
        all_idx = list(range(len(kos)))  # all indices
        wt_idx = [i for i, ko in enumerate(kos) if ko is None]

        # Sort adatas by first gene expression for determinism
        sorted_adatas = []
        for adata in adatas:
            sort_indices = np.argsort(adata.X[:, 0])
            sorted_adata = adata[sort_indices].copy()
            sorted_adatas.append(sorted_adata)

        print("Training reference model with knockouts...")
        self.estimator = rf.Estimator(
            [sorted_adatas[i] for i in all_idx],
            [kos[i] for i in all_idx],
            **self.options,
            num_timepoints=len(sorted_adatas[0].obs["t"].unique()),
        )
        self.estimator.fit(print_iter=100, alg="alternating", update_couplings_iter=250)

        if also_wt:
            print("Training reference model with wild type data only...")
            self.estimator_wt = rf.Estimator(
                [sorted_adatas[i] for i in wt_idx],
                [kos[i] for i in wt_idx],
                **self.options,
                num_timepoints=len(sorted_adatas[0].obs["t"].unique()),
            )
            self.estimator_wt.fit(
                print_iter=100, alg="alternating", update_couplings_iter=250
            )

    def get_interaction_matrix(self):
        """Return the interaction matrix from the full model."""
        return self.estimator.A if self.estimator else None

    def get_wild_type_matrix(self):
        """Return the interaction matrix from the wild-type only model."""
        return self.estimator_wt.A if self.estimator_wt else None

    def simulate_trajectory(
        self, x0, n_times, use_wildtype=False, n_points=400, ko_condition=None
    ):
        estimator = self.estimator_wt if use_wildtype else self.estimator
        A = estimator.A_orig
        x0 = x0.float()
        A = A.float()

        # t = 1.0 / estimator.T
        t = 1.0 / estimator.T
        P = torch.linalg.matrix_exp(t * A)

        x1 = (
            (x0 / estimator.std.float()) @ P + t * (estimator.b.float())
        ) * estimator.std.float()

        return x1

    def training_step(self, batch, batch_idx):
        """In our Lightning training_step we assume that the datamodule (e.g. grn_datamodule)
        provides a batch with keys "adatas" and "kos".

        On the first training_step we call fit_model to run the estimator fitting. Subsequent calls
        simply log a dummy loss.
        """
        # To ensure we run the fitting only once, check if the estimator has been created.
        if self.estimator is None:
            datamodule = self.trainer.datamodule
            adatas = datamodule.get_subset_adatas()
            kos = datamodule.kos
            self.fit_model(adatas, kos)
        # Return a dummy loss for compatibility with Lightning.
        loss = self.dummy_param * 0.0
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Fill in validation step with something."""
        return {}

    def test_step(self, batch, batch_idx):
        """Evaluate on test data (e.g. held-out time, etc.)."""
        return {}

    def on_train_epoch_end(self):
        try:
            W_v = self.get_interaction_matrix()
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
        cax = ax.imshow(maskdiag(W_v), vmin=-2.5, vmax=2.5, cmap="RdBu_r")
        ax.invert_yaxis()
        ax.set_title("Causal Graph (from ReferenceFitting)")
        fig.colorbar(cax)

        if self.logger is not None:
            self.logger.experiment.add_figure(
                "Causal_Graph", fig, global_step=self.global_step
            )
            plt.close(fig)
        else:
            plt.show()

        # Log a dummy metric for tracking visualization steps
        self.log("epoch/plot_causal_graph", 1)

    def configure_optimizers(self):
        """Return a dummy optimizer.

        Although the actual optimization is handled inside the estimator.fit() calls, Lightning
        requires an optimizer to be defined.
        """
        optimizer = self.options["optimizer"](self.parameters(), lr=self.options["lr"])
        return optimizer
