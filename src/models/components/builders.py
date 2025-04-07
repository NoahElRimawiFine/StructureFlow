import torch


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

    For each dataset (indexed by i), create a [batch_size, dim] matrix where, if the dataset has a
    knockout (kos[i] is not None), the i-th column is set to 1.
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
