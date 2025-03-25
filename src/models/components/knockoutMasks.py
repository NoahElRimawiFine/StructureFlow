import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import torch


def build_knockout_mask(dim: int, ko_idx: int):
    mask = torch.ones((dim, dim), dtype=torch.float32)
    if ko_idx is not None:
        mask[:, ko_idx] = 0.0
        mask[ko_idx, ko_idx] = 1.0
    return mask


class KnockoutMaskProvider:
    """A callable provider that computes knockout masks from a given data path.

    You can extend this to load a datamodule or AnnData objects as needed.
    """

    def __init__(
        self, data_path: str = "data/", dataset: str = "dyn-TF", dataset_type: str = "Synthetic"
    ):
        self.data_path = os.path.join(data_path, dataset_type)
        self.dataset = dataset
        self.dataset_type = dataset_type

    def __call__(self):
        # Load your data using your preferred logic.
        # Here we use glob to load paths and then a helper (e.g. util.load_adata)
        # Assume util.load_adata is a function that returns an AnnData object.
        paths = []
        if self.dataset_type == "Synthetic":
            paths = glob.glob(
                os.path.join(self.data_path, f"{self.dataset}/{self.dataset}*-1")
            ) + glob.glob(os.path.join(self.data_path, f"{self.dataset}_ko*/{self.dataset}*-1"))
        elif self.dataset_type == "Curated":
            paths = glob.glob(os.path.join(self.data_path, "HSC*/HSC*-1"))
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        from src.datamodules.components import (
            sc_dataset as util,  # adjust import as needed
        )

        adatas = [util.load_adata(p) for p in paths]

        # Get knockouts
        self.kos = []
        for p in paths:
            try:
                self.kos.append(os.path.basename(p).split("_ko_")[1].split("-")[0])
            except (IndexError, ValueError, AttributeError):
                self.kos.append(None)

        self.gene_to_index = {gene: idx for idx, gene in enumerate(adatas[0].var.index)}
        self.ko_indices = []
        for ko in self.kos:
            if ko is None:
                self.ko_indices.append(None)
            else:
                self.ko_indices.append(self.gene_to_index[ko])

        knockout_masks = []
        # Use the first dataset to determine data dimension.
        dim = adatas[0].X.shape[1]
        for i, ko_idx in enumerate(self.ko_indices):
            mask_i = build_knockout_mask(dim, ko_idx)
            knockout_masks.append(mask_i)
        return knockout_masks
