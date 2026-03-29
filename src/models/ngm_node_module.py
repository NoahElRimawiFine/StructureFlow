import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
from geomloss import SamplesLoss

from src.models.components.base import MLPODEF


class _ZeroCorrModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, x):
        return torch.zeros_like(x)


class _ZeroScoreModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditional = False
        self.conditional_dim = 0

    def forward(self, t, x, cond=None):
        return torch.zeros_like(x)


class _MLPODEFWrapper(torch.nn.Module):
    """Wraps MLPODEF to accept an optional dataset_idx argument (ignored)."""

    def __init__(self, mlpodef):
        super().__init__()
        self.mlpodef = mlpodef
        self.dims = mlpodef.dims
        self.GL_reg = mlpodef.GL_reg
        self.fc1 = mlpodef.fc1

    def forward(self, t, x, dataset_idx=None):
        return self.mlpodef(t, x)

    def causal_graph(self, w_threshold=0.0):
        return self.mlpodef.causal_graph(w_threshold)

    def l2_reg(self):
        return self.mlpodef.l2_reg()

    def fc1_reg(self):
        return self.mlpodef.fc1_reg()

    def parameters(self):
        return self.mlpodef.parameters()

    def train(self, mode=True):
        self.mlpodef.train(mode)
        return self

    def eval(self):
        self.mlpodef.eval()
        return self

    def to(self, device):
        self.mlpodef.to(device)
        return self


class NGMNodeModule:
    """NGM-NODE: Neural ODE with Sinkhorn optimal transport loss for GRN inference.

    Uses plain MLPODEF (no knockout masks) exactly as in the scaling experiment.
    Trains via Sinkhorn pushforward matching across all datasets jointly.
    Compatible with leave_one_out.py evaluation via func_v, v_correction, score_net.
    """

    def __init__(
        self,
        adatas,
        kos,
        n_steps=4000,
        lr=0.005,
        gl_reg=0.05,
        hidden_dim=128,
        batch_size=64,
        device="cpu",
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.kos = kos
        self.gl_reg = gl_reg

        n = adatas[0].X.shape[1]
        self.n_genes = n

        dims = [n, hidden_dim, 1]
        self.dims = dims

        _mlpodef = MLPODEF(dims=dims, GL_reg=gl_reg, bias=True)
        self.func_v = _MLPODEFWrapper(_mlpodef)

        self.v_correction = _ZeroCorrModule()
        self.score_net = _ZeroScoreModule()
        self.conditionals = [torch.zeros(1, n) for _ in range(len(adatas))]

        self.optimizer = torch.optim.Adam(self.func_v.parameters(), lr=lr)
        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

    def _proximal(self):
        with torch.no_grad():
            w = self.func_v.fc1.weight
            d = self.dims[0]
            d_hidden = self.dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - self.gl_reg * 0.01
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    def fit_model_with_holdout(self, fold_adatas, kos, held_out_time, **kwargs):
        """Train on fold_adatas (held_out_time already excluded from fold_adatas)."""
        self.func_v.train()
        n_datasets = len(fold_adatas)

        transition_times = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, device=self.device
        )

        for step in range(self.n_steps):
            self.optimizer.zero_grad()

            ds_idx = np.random.randint(0, n_datasets)
            adata = fold_adatas[ds_idx]

            available_times = sorted(adata.obs["t"].unique())
            available_transitions = []
            for j in range(len(available_times) - 1):
                available_transitions.append(
                    (available_times[j], available_times[j + 1])
                )

            if not available_transitions:
                continue

            tb, tb_next = available_transitions[
                np.random.randint(0, len(available_transitions))
            ]

            x0_np = adata.X[adata.obs["t"] == tb]
            x1_np = adata.X[adata.obs["t"] == tb_next]

            if x0_np.shape[0] == 0 or x1_np.shape[0] == 0:
                continue

            bs = min(self.batch_size, x0_np.shape[0], x1_np.shape[0])
            idx0 = np.random.choice(x0_np.shape[0], bs, replace=bs > x0_np.shape[0])
            idx1 = np.random.choice(x1_np.shape[0], bs, replace=bs > x1_np.shape[0])

            x0_batch = torch.from_numpy(x0_np[idx0]).float().to(self.device)
            x1_observed = torch.from_numpy(x1_np[idx1]).float().to(self.device)

            captured_ds_idx = ds_idx

            def ode_func(t, x):
                t_val = t.item() if hasattr(t, "item") else float(t)
                t_in = torch.full(
                    (x.shape[0], 1), t_val, device=x.device, dtype=torch.float32
                )
                return self.func_v(t_in, x, captured_ds_idx)

            x_traj = odeint(ode_func, x0_batch.unsqueeze(1), transition_times)
            x1_predicted = x_traj[-1].squeeze(1)

            loss = self.sinkhorn_loss(x1_predicted, x1_observed)
            loss.backward()
            self.optimizer.step()
            self._proximal()

            if step % 500 == 0:
                print(
                    f"NGM-NODE Step {step}/{self.n_steps}, Loss: {loss.item():.4f}",
                    flush=True,
                )

    def eval(self):
        self.func_v.eval()

    def to(self, device):
        self.func_v.to(device)
        return self
