import glob
import os
import logging
log = logging.getLogger(__name__)

import anndata as ad
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split, IterableDataset, ConcatDataset
import scanpy as sc
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
            dataset_type: "Synthetic", "Curated", or "Renge"
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
            if self.dataset_type == "Renge":
                self._setup_renge_data()
            else:
                self._setup_synthetic_or_curated_data()

            # Create datasets from the loaded AnnData objects
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

    def _setup_synthetic_or_curated_data(self):
        """Load Synthetic or Curated data from disk."""
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

    def _setup_renge_data_old(self):
        """Load Renge data from disk and convert to AnnData objects."""
        # Load the Renge data files
        x_renge_path = os.path.join(self.data_path, "X_renge_d2_80.csv")
        e_renge_path = os.path.join(self.data_path, "E_renge_d2_80.csv")

        # Load the reference network if available
        ref_network_path = os.path.join(self.data_path, "A_ref_thresh_0.csv")
        try:
            ref_network = pd.read_csv(ref_network_path, index_col=0)
            has_ref_network = True
        except FileNotFoundError:
            has_ref_network = False
        
        # Load the data
        x_renge = pd.read_csv(x_renge_path, index_col=0)
        e_renge = pd.read_csv(e_renge_path, index_col=0)

        # Extract time column from X_RENGE
        time_column = x_renge.pop('t').values
        
        # Get gene names (all columns except the last one from X_RENGE, and first one is cell IDs)
        gene_names = x_renge.columns.tolist()[1:]
        self.dim = len(gene_names)
        
        # Set the reference network matrix - keep original dimensions
        if has_ref_network:
            self.true_matrix = ref_network
        else:
            # Create an empty matrix if no reference is available
            self.true_matrix = pd.DataFrame(
                np.zeros((self.dim, self.dim), int),
                index=gene_names,
                columns=gene_names,
            )
        
        # Create a dictionary to group cells by knockout gene
        ko_groups = {}
        
        # For each row, determine the knockout gene (if any)
        for idx, row in x_renge.iterrows():
            ko_gene = None
            for gene in gene_names:
                if row[gene] == 1.0:
                    ko_gene = gene 
                    break
            
            # Add to appropriate group (None for wildtype)
            if ko_gene not in ko_groups:
                ko_groups[ko_gene] = []
            ko_groups[ko_gene].append(idx)
        
        # Create separate AnnData objects for each knockout condition
        self.adatas = []
        self.kos = []
        self.ko_indices = []
        
        # gene_to_index mapping
        self.gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}
        
        for ko_gene, cell_indices in ko_groups.items():
            # Extract expression data for these cells
            subset_expr = e_renge.loc[cell_indices]
            # Extract time data for these cells
            subset_time = pd.Series(time_column[np.where(np.isin(x_renge.index, cell_indices))[0]], 
                                index=cell_indices)
            
            # Create AnnData object
            adata = ad.AnnData(X=subset_expr.values)
            adata.obs_names = subset_expr.index
            adata.var_names = e_renge.columns
            
            # Store time in obs
            unique_times = np.unique(subset_time.values)
            time_mapping = {t: i for i, t in enumerate(sorted(unique_times))}
            adata.obs['t'] = np.array([time_mapping[t] for t in subset_time.values])

            self.adatas.append(adata)
            self.kos.append(ko_gene)
            self.ko_indices.append(None if ko_gene is None else self.gene_to_index[ko_gene])

    def _setup_renge_data(self):
        """Load Renge data from disk and use the hipsc AnnData object."""
        # Load the hipsc.h5ad file
        hipsc_path = os.path.join(self.data_path, "hipsc.h5ad")
        hipsc = sc.read_h5ad(hipsc_path)
        
        # Load the reference network if available
        ref_network_path = os.path.join(self.data_path, "A_ref_thresh_0.csv")
        try:
            ref_network = pd.read_csv(ref_network_path, index_col=0)
            has_ref_network = True
        except FileNotFoundError:
            has_ref_network = False
        
        # Set dimensions based on genes in the hipsc data
        gene_names = hipsc.var_names.tolist()
        self.dim = len(gene_names)
        
        # Set the reference network matrix
        if has_ref_network:
            self.true_matrix = ref_network
        else:
            # Create an empty matrix if no reference is available
            self.true_matrix = pd.DataFrame(
                np.zeros((self.dim, self.dim), int),
                index=gene_names,
                columns=gene_names,
            )
        
        # Shift timepoints to start from 0
        min_t = hipsc.obs['t'].min()
        hipsc.obs['t'] = hipsc.obs['t'] - min_t
        
        # Convert sparse matrix to dense if needed
        if hasattr(hipsc.X, 'toarray'):
            hipsc.X = hipsc.X.toarray()
        
        # Create separate AnnData objects for each knockout condition
        ko_groups = hipsc.obs.groupby('ko')
        
        self.adatas = []
        self.kos = []
        self.ko_indices = []
        
        # gene_to_index mapping
        self.gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}
        
        for ko_gene, indices in ko_groups.indices.items():
            # Extract subset for this condition
            adata_subset = hipsc[indices].copy()
            
            # Handle the case where ko_gene might be NaN or a special value for wildtype
            if pd.isna(ko_gene) or ko_gene == 'wt' or ko_gene == 'WT' or ko_gene == '':
                ko_gene = None
            
            self.adatas.append(adata_subset)
            self.kos.append(ko_gene)
            self.ko_indices.append(None if ko_gene is None else self.gene_to_index.get(ko_gene))

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

    def _identity_collate(self, x):
        """Identity collate function that returns the input batch."""
        return x

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
            collate_fn=self._identity_collate
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
