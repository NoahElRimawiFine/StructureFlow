import glob
import os

import anndata as ad
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split, IterableDataset

from .components import sc_dataset as util


class AnnDataDataset(Dataset):
    """Wraps an AnnData object so that each sample is a dictionary with the cell's expression data
    and its corresponding pseudo-time."""

    def __init__(self, adata, source_id: int = None):
        """
        Args:
            adata: An AnnData object.
        """
        self.adata = adata
        self.source_id = source_id
        # If the data matrix is sparse, convert it to dense (or you can use .toarray())
        if hasattr(adata.X, "toarray"):
            self.X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            self.X = torch.tensor(adata.X, dtype=torch.float32)
        # Assume pseudo-time is stored in adata.obs["t"]
        self.t = torch.tensor(adata.obs["t"].values, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx], "t": self.t[idx], "source_id": self.source_id}


class TrajectoryStructureDataModule(pl.LightningDataModule):
    """A LightningDataModule that loads your custom dataset from disk and returns DataLoaders for
    train/val/test."""

    pass_to_model = True

    def __init__(
        self,
        data_path: str = "data/",
        dataset: str = "dyn-TF",
        dataset_type: str = "Synthetic",
        batch_size: int = 64,
        num_workers: int = 4,
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
        T: int = 5,
        use_dummy_train_loader: bool = False,
        dummy_loader_steps: int = 10000,
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
        self.use_dummy_train_loader = use_dummy_train_loader
        self.dummy_loader_steps = dummy_loader_steps
        self.data_path = os.path.join(data_path, dataset_type)
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.T = T

        # Will be filled in setup():
        self.adatas = None
        self.kos = None
        self.ko_indices = None
        self.true_matrix = None
        self.dim = None

        self._full_dataset = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def prepare_data(self):
        """
        - Called once on 1 GPU/CPU in a distributed environment.
        - Here download/untar data if needed.
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
            self.dim = n_genes
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
            t_bins = np.linspace(0, 1, self.T + 1)[:-1]
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
            all_datasets = []
            for adata in self.adatas:
                ds = adata
                all_datasets.append(ds)

            from torch.utils.data import ConcatDataset

            wrapped_datasets = [
                AnnDataDataset(adata, source_id=i) for i, adata in enumerate(self.adatas)
            ]
            self._dataset_lengths = [len(ds) for ds in wrapped_datasets]
            self._full_dataset = ConcatDataset(wrapped_datasets)

            # train/val/test split:
            full_len = len(self._full_dataset)
            train_len = int(full_len * self.train_val_test_split[0])
            val_len = int(full_len * self.train_val_test_split[1])
            test_len = full_len - train_len - val_len
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                self._full_dataset, [train_len, val_len, test_len]
            )

        # if stage == "test" or something, we could do different logic
        # but often we do all in one go

    def get_subset_adatas(self, split: str = "train"):
        """Returns a list of AnnData objects, each containing only the cells used in the specified
        split.

        Args:
            split (str): One of "train", "val", or "test" indicating which split's cells to extract.

        Returns:
            List[AnnData]: A list of AnnData objects containing only the cells for the specified split.
        """
        # Select the appropriate Subset from the random split
        if split == "train":
            subset = self.dataset_train
        elif split == "val":
            subset = self.dataset_val
        elif split == "test":
            subset = self.dataset_test
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        # Get the global indices of the selected cells from the ConcatDataset.
        indices = subset.indices  # This is a list of integers.

        # Compute the cumulative lengths for each wrapped AnnDataDataset.
        # For example, if we have 3 datasets with lengths L1, L2, L3, then:
        # cum_lengths = [0, L1, L1+L2, L1+L2+L3]
        cum_lengths = np.cumsum(
            [0] + self._dataset_lengths
        )  # self._dataset_lengths was computed in setup().

        # Create a dictionary to accumulate local indices for each original AnnData.
        indices_by_file = {i: [] for i in range(len(self.adatas))}

        # Map each global index to its corresponding dataset.
        for idx in indices:
            for i in range(len(cum_lengths) - 1):
                if cum_lengths[i] <= idx < cum_lengths[i + 1]:
                    local_idx = idx - cum_lengths[i]
                    indices_by_file[i].append(local_idx)
                    break

        # Now, for each original AnnData, select the subset of cells corresponding to the computed local indices.
        subset_adatas = []
        for i, adata in enumerate(self.adatas):
            if indices_by_file[i]:
                # AnnData supports advanced indexing: adata[indices, :] returns a new AnnData with those cells.
                subset_adatas.append(adata[indices_by_file[i], :])

        return subset_adatas

    def train_dataloader(self):
        if self.use_dummy_train_loader:
            print(f"Using dummy infinite train dataloader for approx {self.dummy_loader_steps} steps.")
            dummy_dataset = DummyInfiniteDataset(data_shape=(1,), length=self.dummy_loader_steps + 10) # Add buffer
            return DataLoader(dummy_dataset, batch_size=1, num_workers=0)
        else:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.dim,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.identity_collate
        )
        
    def identity_collate(self, batch):
        return batch

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=len(self.dataset_test),
            shuffle=False,
            num_workers=self.num_workers,
        )

class DummyInfiniteDataset(IterableDataset):
    """
    An IterableDataset that yields dummy tensors indefinitely or up to a specified length.
    Useful when the training step doesn't depend on the dataloader's output
    but needs to run for a fixed number of steps.
    """
    def __init__(self, data_shape=(1,), length=None):
        """
        Args:
            data_shape (tuple): The shape of the dummy tensor to yield.
            length (int, optional): If provided, stop iteration after yielding this many items.
                                     If None, yield indefinitely. Defaults to None.
        """
        super().__init__()
        self.data_shape = data_shape
        self.length = length

    def __iter__(self):
        count = 0
        while True:
            if self.length is not None and count >= self.length:
                return # Stop iteration
            # Yield a dummy tensor (content doesn't matter)
            # Ensure it's on the correct device type (CPU in this case) if needed,
            # though Lightning usually handles placement.
            yield torch.zeros(self.data_shape)
            count += 1


if __name__ == "__main__":
    _ = TrajectoryStructureDataModule()
    _.setup(stage="fit")
    data = _.val_dataloader()
    for batch in data:
        print(batch)
    print(_.get_subset_adatas("val"))
