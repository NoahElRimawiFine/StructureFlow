import numpy as np
import ot
import torch
import torch.nn as nn
from torchdiffeq import odeint


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


class _SimpleFlowMLP(nn.Module):
    """Plain time-varying MLP for vanilla flow matching.

    Input: [t, x] concatenated → output: velocity (same dim as x).
    Compatible with simulate_trajectory and compute_global_jacobian call signatures.
    """

    def __init__(self, d, hidden_sizes=(128, 128)):
        super().__init__()
        self.d = d
        layers = []
        in_dim = d + 1
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.GELU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x, dataset_idx=None):
        if x.dim() == 3:
            B = x.shape[0]
            x_2d = x.reshape(B, self.d)
        else:
            x_2d = x
            B = x.shape[0]

        if t.dim() == 0:
            t_1d = t.unsqueeze(0).expand(B)
        elif t.dim() == 1:
            t_1d = t
        else:
            t_1d = t.reshape(-1)

        inp = torch.cat([x_2d, t_1d.unsqueeze(-1)], dim=-1)
        out = self.net(inp)
        return out.unsqueeze(1)

    def causal_graph(self, w_threshold=0.0):
        return np.zeros((self.d, self.d))


def _compute_ot_plan(x0_np, x1_np, reg=0.05):
    a = ot.unif(x0_np.shape[0])
    b = ot.unif(x1_np.shape[0])
    M = np.sum((x0_np[:, None, :] - x1_np[None, :, :]) ** 2, axis=2)
    return ot.sinkhorn(a, b, M, reg=reg, numItermax=5000)


def _sample_from_plan(x0_np, x1_np, pi, batch_size):
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(len(p), size=batch_size, replace=True, p=p)
    i_idx = choices // pi.shape[1]
    j_idx = choices % pi.shape[1]
    return x0_np[i_idx], x1_np[j_idx]


class VanillaFlowMatchingModule:
    """Vanilla flow matching: OT coupling + straight-line interpolation + simple MLP.

    No knockout masks, no group lasso, no score matching, no correction network.
    All datasets are trained with the same shared MLP (no conditioning on KO status).
    """

    def __init__(
        self,
        adatas,
        kos,
        n_steps=4000,
        lr=1e-3,
        hidden_sizes=(128, 128),
        batch_size=64,
        ot_reg=0.05,
        device="cpu",
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.kos = kos
        self.ot_reg = ot_reg

        n = adatas[0].X.shape[1]
        self.n_genes = n

        self.func_v = _SimpleFlowMLP(n, hidden_sizes).to(self.device)
        self.v_correction = _ZeroCorrModule()
        self.score_net = _ZeroScoreModule()
        self.conditionals = [torch.zeros(1, n) for _ in range(len(adatas))]

        self.optimizer = torch.optim.Adam(self.func_v.parameters(), lr=lr)

    def fit_model_with_holdout(self, fold_adatas, kos, held_out_time, **kwargs):
        """Pre-compute OT plans then train via straight-line flow matching."""
        T_max = int(max(adata.obs["t"].max() for adata in fold_adatas)) + 1

        print("Pre-computing OT plans...")
        ot_plans = []
        for adata in fold_adatas:
            available_times = sorted(adata.obs["t"].unique())
            ds_plans = {}
            for j in range(len(available_times) - 1):
                tb = available_times[j]
                tb_next = available_times[j + 1]
                x0_np = adata.X[adata.obs["t"] == tb]
                x1_np = adata.X[adata.obs["t"] == tb_next]
                if x0_np.shape[0] > 0 and x1_np.shape[0] > 0:
                    ds_plans[(tb, tb_next)] = (
                        x0_np,
                        x1_np,
                        _compute_ot_plan(x0_np, x1_np, self.ot_reg),
                    )
            ot_plans.append(ds_plans)

        print(f"Training vanilla flow matching for {self.n_steps} steps...")
        self.func_v.train()
        n_datasets = len(fold_adatas)

        for step in range(self.n_steps):
            self.optimizer.zero_grad()

            ds_idx = np.random.randint(0, n_datasets)
            ds_plans = ot_plans[ds_idx]

            if not ds_plans:
                continue

            transition = list(ds_plans.keys())[
                np.random.randint(0, len(ds_plans))
            ]
            tb, tb_next = transition
            x0_np, x1_np, pi = ds_plans[transition]

            x0_samp, x1_samp = _sample_from_plan(x0_np, x1_np, pi, self.batch_size)
            x0 = torch.from_numpy(x0_samp).float().to(self.device)
            x1 = torch.from_numpy(x1_samp).float().to(self.device)

            tau = torch.rand(self.batch_size, device=self.device)
            t_global = (
                torch.tensor(float(tb), device=self.device) + tau
            ) / T_max

            tau_col = tau.unsqueeze(1)
            x_tau = (1 - tau_col) * x0 + tau_col * x1
            u = x1 - x0

            v_pred = self.func_v(t_global, x_tau).squeeze(1)
            loss = torch.mean((v_pred - u) ** 2)

            loss.backward()
            self.optimizer.step()

            if step % 500 == 0:
                print(
                    f"VanillaFM Step {step}/{self.n_steps}, Loss: {loss.item():.4f}",
                    flush=True,
                )

    def eval(self):
        self.func_v.eval()

    def to(self, device):
        self.func_v.to(device)
        return self
