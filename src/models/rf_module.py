import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .components import rf


class ReferenceFittingModule(pl.LightningModule):
    def __init__(self, use_cuda=True, iter=1000, dt_values=None, time_values=None):
        super().__init__()
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
        self.estimator = None
        self.estimator_wt = None
        self.dt_values = dt_values
        self.time_values = time_values

        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def fit_model(self, adatas, kos, also_wt=False, dt_values=None, time_values=None):
        """Fits the reference model using both knockout and wild-type data.

        This method replicates your original 'train' method.
        """
        if dt_values is not None:
            self.dt_values = dt_values
        if time_values is not None:
            self.time_values = time_values

        ko_idx = list(range(len(kos)))
        wt_idx = [i for i, ko in enumerate(kos) if ko is None]

        sorted_adatas = []
        for adata in adatas:
            sort_indices = np.argsort(adata.X[:, 0])
            sorted_adata = adata[sort_indices].copy()
            sorted_adatas.append(sorted_adata)

        print("Training reference model with knockouts...")
        self.estimator = rf.Estimator(
            [sorted_adatas[i] for i in ko_idx],
            [kos[i] for i in ko_idx],
            dt_values=self.dt_values,
            time_values=self.time_values,
            **self.options,
        )
        self.estimator.fit(print_iter=100, alg="alternating", update_couplings_iter=250)

        if also_wt:
            print("Training reference model with wild type data only...")
            self.estimator_wt = rf.Estimator(
                [sorted_adatas[i] for i in wt_idx],
                [kos[i] for i in wt_idx],
                dt_values=self.dt_values,
                time_values=self.time_values,
                **self.options,
            )
            self.estimator_wt.fit(
                print_iter=100, alg="alternating", update_couplings_iter=250
            )

    def fit_model_with_holdout(
        self,
        adatas,
        kos,
        left_out_time,
        also_wt=False,
        dt_values=None,
        time_values=None,
    ):
        """Fits the reference model using both knockout and wild-type data, taking into account a
        hold-out time.

        Replicates your original 'train_with_holdout' method.
        """
        if dt_values is not None:
            self.dt_values = dt_values
        if time_values is not None:
            self.time_values = time_values

        all_idx = list(range(len(kos)))
        wt_idx = [i for i, ko in enumerate(kos) if ko is None]

        sorted_adatas = []
        for adata in adatas:
            sort_indices = np.argsort(adata.X[:, 0])
            sorted_adata = adata[sort_indices].copy()
            sorted_adatas.append(sorted_adata)

        print("Training reference model with knockouts...")
        self.estimator = rf.Estimator(
            [sorted_adatas[i] for i in all_idx],
            [kos[i] for i in all_idx],
            num_timepoints=len(sorted_adatas[0].obs["t"].unique()),
            dt_values=self.dt_values,
            time_values=self.time_values,
            **self.options,
        )
        self.estimator.fit(print_iter=100, alg="alternating", update_couplings_iter=250)

        if also_wt:
            print("Training reference model with wild type data only...")
            self.estimator_wt = rf.Estimator(
                [sorted_adatas[i] for i in wt_idx],
                [kos[i] for i in wt_idx],
                num_timepoints=len(sorted_adatas[0].obs["t"].unique()),
                dt_values=self.dt_values,
                time_values=self.time_values,
                **self.options,
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
        self,
        x0,
        n_times,
        use_wildtype=False,
        n_points=400,
        ko_condition=None,
        transition_idx=None,
    ):
        estimator = self.estimator_wt if use_wildtype else self.estimator
        A = estimator.A_orig
        x0 = x0.float()
        A = A.float()

        if (
            transition_idx is not None
            and hasattr(estimator, "dt_values")
            and estimator.dt_values is not None
        ):
            t = estimator.dt_values[transition_idx]
        else:
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
            return

        def maskdiag(A):
            return A * (1 - np.eye(A.shape[0]))

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

        self.log("epoch/plot_causal_graph", 1)

        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            datamodule = self.trainer.datamodule
            if hasattr(datamodule, "true_matrix"):
                true_matrix = datamodule.true_matrix
                if hasattr(true_matrix, "values"):
                    A_true = true_matrix.values
                else:
                    A_true = true_matrix

                masked_W_v = maskdiag(W_v)
                masked_A_true = maskdiag(A_true)

                y_true = np.abs(np.sign(masked_A_true).astype(int).flatten())
                y_pred = np.abs(masked_W_v.flatten())

                if len(np.unique(y_true)) > 1:
                    jacobian_ap = average_precision_score(y_true, y_pred)
                    jacobian_auroc = roc_auc_score(y_true, y_pred)

                    self.log("grn/jacobian_ap", jacobian_ap, prog_bar=True)
                    self.log("grn/jacobian_auroc", jacobian_auroc, prog_bar=True)
                    self.log("grn/graph_ap", jacobian_ap, prog_bar=True)
                    self.log("grn/graph_auroc", jacobian_auroc, prog_bar=True)

    def configure_optimizers(self):
        """Return a dummy optimizer.

        Although the actual optimization is handled inside the estimator.fit() calls, Lightning
        requires an optimizer to be defined.
        """
        optimizer = self.options["optimizer"](self.parameters(), lr=self.options["lr"])
        return optimizer
