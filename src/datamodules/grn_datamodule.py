import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .components import sc_dataset as util

T = 5


class TrajectoryStructureDataModule(pl.LightningDataModule):
    """A LightningDataModule that loads your custom dataset from disk and returns DataLoaders for
    train/val/test."""

    def __init__(
        self,
        data_path: str = "data/",
        dataset: str = "dyn-TF",
        dataset_type: str = "Synthetic",
        batch_size: int = 64,
        num_workers: int = 4,
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
    ):
        """
        Args:
            data_path: Path to data directory
            dataset_type: "Synthetic" or "Curated"
            batch_size: batch size for the DataLoader
            num_workers: how many workers for DataLoader
            train_val_test_split: ratio to split the entire dataset
        """
        super().__init__()
        self.data_path = os.path.join(data_path, dataset_type)
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

        # Will be filled in setup():
        self.adatas = None
        self.kos = None
        self.ko_indices = None
        self.true_matrix = None

        self._full_dataset = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def prepare_data(self):
        """
        - Called once on 1 GPU/CPU in a distributed environment.
        - Here download/untar data if needed.
        - Our data presumably is already local, so we do nothing special.
        """
        pass

    def setup(self, stage=None):
        """
        - Called on every GPU/CPU in a distributed environment.
        - This is where we typically load data from disk,
          build/transform Datasets, and split them.
        """
        if stage == "fit" or stage is None:
            # We'll load everything once
            paths = []
            if self.dataset_type == "Synthetic":
                paths = glob.glob(
                    os.path.join(self.data_path, f"{self.dataset}/{self.dataset}*-1")
                ) + glob.glob(
                    os.path.join(self.data_path, f"{self.dataset}_ko*/{self.dataset}*-1")
                )
            elif self.dataset_type == "Curated":
                paths = glob.glob(os.path.join(self.data_path, "HSC*/HSC*-1"))
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")

            self.adatas = [util.load_adata(p) for p in paths]

            # Build the reference matrix
            df = pd.read_csv(os.path.join(os.path.dirname(paths[0]), "refNetwork.csv"))
            n_genes = self.adatas[0].n_vars
            self.true_matrix = pd.DataFrame(
                np.zeros((n_genes, n_genes), int),
                index=self.adatas[0].var.index,
                columns=self.adatas[0].var.index,
            )
            for i in range(df.shape[0]):
                _i = df.iloc[i, 1]  # target gene
                _j = df.iloc[i, 0]  # source gene
                _v = {"+": 1, "-": -1}[df.iloc[i, 2]]
                self.true_matrix.loc[_i, _j] = _v

            # Bin timepoints
            t_bins = np.linspace(0, 1, T + 1)[:-1]
            for adata in self.adatas:
                adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1

            # Identify the knockouts
            self.kos = []
            for p in paths:
                try:
                    self.kos.append(os.path.basename(p).split("_ko_")[1].split("-")[0])
                except IndexError:
                    self.kos.append(None)

            # gene_to_index for knockouts
            self.gene_to_index = {gene: idx for idx, gene in enumerate(self.adatas[0].var.index)}
            self.ko_indices = []
            for ko in self.kos:
                if ko is None:
                    self.ko_indices.append(None)
                else:
                    self.ko_indices.append(self.gene_to_index[ko])

            # Now build a single "full" dataset from all adatas,
            # or keep them separate.
            # For a minimal example, let's merge them into one dataset.
            all_datasets = []
            for adata in self.adatas:
                ds = adata
                all_datasets.append(ds)

            # Merge them by just concatenating in a single ConcatDataset
            from torch.utils.data import ConcatDataset

            self._full_dataset = ConcatDataset(all_datasets)

            # Next, we do train/val/test split:
            full_len = len(self._full_dataset)
            train_len = int(full_len * self.train_val_test_split[0])
            val_len = int(full_len * self.train_val_test_split[1])
            test_len = full_len - train_len - val_len
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                self._full_dataset, [train_len, val_len, test_len]
            )
            print(len(self.dataset_train))

        # if stage == "test" or something, we could do different logic
        # but often we do all in one go

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    _ = TrajectoryStructureDataModule()
