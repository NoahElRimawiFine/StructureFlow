import random
import numpy as np
import torch
import networkx as nx
import pandas as pd
import time
from typing import List, Dict, Any, Tuple
import os
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.linear_model import LinearRegression
import multiprocessing as mp

# SF2M specific imports
import anndata as ad
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

# Add src to path for imports
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.datamodules.grn_datamodule import TrajectoryStructureDataModule
from src.models.sf2m_module import SF2MLitModule

# SF2M component imports for DirectSF2MMethod
from src.models.components.base import MLPODEF
from src.models.components.cond_mlp import MLP as CONDMLP
from src.models.components.simple_mlp import MLP
from src.models.components.optimal_transport import EntropicOTFM

# Additional imports for NGM-NODE
from torchdiffeq import odeint
from geomloss import SamplesLoss


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    seed_everything(seed, workers=True)


def set_fixed_cores(num_cores: int = 4):
    """Set fixed number of cores for reproducible runtime analysis."""
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
    torch.set_num_threads(num_cores)


def generate_causal_system(
    num_vars: int, edge_prob: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random causal system.

    Args:
        num_vars: Number of variables in the system
        edge_prob: Probability of edge existence in random graph
        seed: Random seed

    Returns:
        adjacency_matrix: True causal adjacency matrix (num_vars x num_vars)
                         adjacency[i,j] means variable i causally affects variable j
        dynamics_matrix: System dynamics matrix for simulation
                        dynamics[i,j] means variable j affects derivative of variable i
    """
    set_random_seeds(seed)

    # Generate random directed graph
    graph = nx.erdos_renyi_graph(num_vars, edge_prob, directed=True, seed=seed)

    # Create adjacency matrix with random weights
    adjacency_matrix = np.zeros((num_vars, num_vars))
    for u, v in graph.edges():
        # Random weight between -1 and 1, excluding small values
        weight = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0)
        adjacency_matrix[u, v] = weight

    # Dynamics matrix is simply the transpose - no artificial terms
    dynamics_matrix = adjacency_matrix.T

    return adjacency_matrix, dynamics_matrix


def simulate_time_series(
    dynamics_matrix: np.ndarray,
    num_timepoints: int = 5,
    num_samples: int = 1000,
    noise_std: float = 0.1,
    dt: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate time series data from linear dynamics.

    Args:
        dynamics_matrix: System dynamics matrix
        num_timepoints: Number of time points to simulate
        num_samples: Number of samples per timepoint
        noise_std: Standard deviation of observation noise
        dt: Integration time step
        seed: Random seed

    Returns:
        time_series: Array of shape (num_timepoints, num_samples, num_vars)
    """
    set_random_seeds(seed)

    num_vars = dynamics_matrix.shape[0]
    time_series = []

    for t in range(num_timepoints):
        # Initialize random samples
        samples = np.random.normal(0, 0.5, (num_samples, num_vars))

        # Simulate forward in time
        time_horizon = t * dt
        current_time = 0

        while current_time < time_horizon:
            # Linear dynamics: dx/dt = A * x
            dx_dt = dynamics_matrix @ samples.T
            samples += dx_dt.T * dt
            current_time += dt

        # Add observation noise
        samples += np.random.normal(0, noise_std, samples.shape)
        time_series.append(samples)

    return np.array(time_series)


def create_anndata_from_time_series(time_series_data: np.ndarray) -> ad.AnnData:
    """
    Convert time series data to AnnData format expected by SF2M.

    Args:
        time_series_data: Array of shape (num_timepoints, num_samples, num_vars)

    Returns:
        adata: AnnData object with time series data
    """
    num_timepoints, num_samples, num_vars = time_series_data.shape

    # Reshape data: (total_samples, num_vars)
    data_flat = time_series_data.reshape(-1, num_vars)

    # Create time labels
    time_labels = np.repeat(np.arange(num_timepoints), num_samples)

    # Create AnnData object
    adata = ad.AnnData(X=data_flat)
    adata.obs["t"] = time_labels

    # Add variable names
    adata.var_names = [f"gene_{i}" for i in range(num_vars)]

    return adata


class CausalDiscoveryMethod:
    """Base class for causal discovery methods."""

    def __init__(self, name: str):
        self.name = name
        self._training_time = 0.0

    def fit(self, time_series_data: np.ndarray) -> np.ndarray:
        """
        Fit the causal discovery method to time series data.

        Args:
            time_series_data: Shape (num_timepoints, num_samples, num_vars)

        Returns:
            predicted_adjacency: Predicted adjacency matrix
        """
        raise NotImplementedError

    def get_training_time(self) -> float:
        """Return the training time in seconds."""
        return self._training_time


class CorrelationBasedMethod(CausalDiscoveryMethod):
    """Correlation-based causal discovery using linear regression."""

    def __init__(self, method_type: str = "pearson"):
        super().__init__(f"Correlation-{method_type}")
        self.method_type = method_type

    def fit(self, time_series_data: np.ndarray) -> np.ndarray:
        """
        Fit correlation-based causal discovery.

        Uses regression to predict each variable from all others,
        treating regression coefficients as causal strengths.
        """
        start_time = time.time()

        # Flatten time series data: (num_timepoints * num_samples, num_vars)
        num_timepoints, num_samples, num_vars = time_series_data.shape
        data_flat = time_series_data.reshape(-1, num_vars)

        # Initialize predicted adjacency matrix
        predicted_adjacency = np.zeros((num_vars, num_vars))

        # For each target variable, regress against all other variables
        for target_var in range(num_vars):
            # Prepare predictors (all variables except target)
            predictor_vars = [i for i in range(num_vars) if i != target_var]

            if len(predictor_vars) == 0:
                continue

            X = data_flat[:, predictor_vars]
            y = data_flat[:, target_var]

            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, y)

            # Store regression coefficients as causal strengths
            for i, predictor_var in enumerate(predictor_vars):
                predicted_adjacency[predictor_var, target_var] = np.abs(reg.coef_[i])

        self._training_time = time.time() - start_time
        return predicted_adjacency


class SF2MConfig:
    """Configuration class for SF2M hyperparameters that can scale with system size."""

    def __init__(
        self,
        base_n_steps: int = 5000,
        base_lr: float = 3e-3,
        base_alpha: float = 0.1,
        base_reg: float = 5e-6,
        base_gl_reg: float = 0.04,
        base_knockout_hidden: int = 64,
        base_score_hidden: List[int] = None,
        base_correction_hidden: List[int] = None,
        base_batch_size: int = 32,
        sigma: float = 1.0,
        device: str = "cpu",
        scaling_factor: float = 1.0,
        size_specific_configs: Dict[int, Dict[str, Any]] = None,
    ):
        """
        Initialize SF2M configuration with scaling options.

        Args:
            base_*: Base hyperparameters for small systems (used for auto-scaling)
            scaling_factor: Factor to scale hyperparameters based on system size
            size_specific_configs: Dictionary mapping system size to specific hyperparameters
                                 Example: {10: {'n_steps': 3000, 'lr': 1e-3, 'gl_reg': 0.02},
                                          50: {'n_steps': 8000, 'lr': 5e-4, 'gl_reg': 0.05}}
        """
        self.base_n_steps = base_n_steps
        self.base_lr = base_lr
        self.base_alpha = base_alpha
        self.base_reg = base_reg
        self.base_gl_reg = base_gl_reg
        self.base_knockout_hidden = base_knockout_hidden
        self.base_score_hidden = base_score_hidden or [64, 64]
        self.base_correction_hidden = base_correction_hidden or [32, 32]
        self.base_batch_size = base_batch_size
        self.sigma = sigma
        self.device = device
        self.scaling_factor = scaling_factor
        self.size_specific_configs = size_specific_configs or {}

    def get_scaled_config(self, num_vars: int) -> Dict[str, Any]:
        """Get scaled hyperparameters based on system size with linear scaling."""

        # If specific config exists for this size, use it as base
        if num_vars in self.size_specific_configs:
            specific_config = self.size_specific_configs[num_vars]
            print(f"Using size-specific config for N={num_vars}: {specific_config}")

            # Start with defaults and override with specific values
            config = {
                "n_steps": self.base_n_steps,
                "lr": self.base_lr,
                "alpha": self.base_alpha,
                "reg": self.base_reg,
                "gl_reg": self.base_gl_reg,
                "knockout_hidden": max(self.base_knockout_hidden, num_vars * 2),
                "score_hidden": [max(dim, num_vars) for dim in self.base_score_hidden],
                "correction_hidden": [
                    max(dim, num_vars // 2) for dim in self.base_correction_hidden
                ],
                "sigma": self.sigma,
                "device": self.device,
                "batch_size": self.base_batch_size,
            }

            # Override with specific values
            for key, value in specific_config.items():
                if key in config:
                    config[key] = value
                else:
                    print(f"Warning: Unknown config key '{key}' for size {num_vars}")

            return config

        # Linear scaling based on dimension size (baseline: N=10)
        scaling_factor = num_vars / 10.0

        print(
            f"Using scaling for N={num_vars} (linear factor: {scaling_factor:.1f}x, quadratic factor: {scaling_factor**2:.1f}x)"
        )

        # Scale training steps quadratically with dimension
        n_steps = int(self.base_n_steps * (scaling_factor))

        # Scale model layer sizes linearly with dimension
        knockout_hidden = int(self.base_knockout_hidden * scaling_factor)
        score_hidden = [int(dim * scaling_factor) for dim in self.base_score_hidden]
        correction_hidden = [
            int(dim * scaling_factor) for dim in self.base_correction_hidden
        ]

        # Keep other hyperparameters fixed (they were optimized for this problem)
        config = {
            "n_steps": n_steps,
            "lr": self.base_lr,  # Keep fixed
            "alpha": self.base_alpha,  # Keep fixed
            "reg": self.base_reg,  # Keep fixed
            "gl_reg": self.base_gl_reg,  # Keep fixed
            "knockout_hidden": knockout_hidden,
            "score_hidden": score_hidden,
            "correction_hidden": correction_hidden,
            "sigma": self.sigma,
            "device": self.device,
            "batch_size": self.base_batch_size,  # Keep fixed
        }

        return config


class DirectSF2MMethod(CausalDiscoveryMethod):
    """Direct SF2M implementation using core components."""

    def __init__(self, hyperparams: Dict[str, Any], silent: bool = False):
        super().__init__("DirectSF2M")
        self.hyperparams = hyperparams
        self.needs_true_adjacency = (
            True  # Flag to indicate this method needs true adjacency for evaluation
        )
        self.model = None
        self.silent = silent  # Flag to suppress debug output

    def fit(
        self, time_series_data: np.ndarray, true_adjacency: np.ndarray = None
    ) -> np.ndarray:
        """
        Fit SF2M model directly to time series data using core SF2M training logic.

        Args:
            time_series_data: Shape (num_timepoints, num_samples, num_vars)

        Returns:
            predicted_adjacency: Predicted adjacency matrix from SF2M
        """
        start_time = time.time()

        num_timepoints, num_samples, num_vars = time_series_data.shape
        if not self.silent:
            print(f"  SF2M Debug - Input data shape: {time_series_data.shape}")

        # Convert data to AnnData format
        adata = create_anndata_from_time_series(time_series_data)
        if not self.silent:
            print(
                f"  SF2M Debug - Created AnnData: {adata.shape}, times: {adata.obs['t'].unique()}"
            )

        # Calculate time parameters
        T_max = int(adata.obs["t"].max())
        T_times = T_max + 1
        DT_data = 1.0 / T_times

        if not self.silent:
            print(
                f"  SF2M Debug - T_max: {T_max}, T_times: {T_times}, DT_data: {DT_data}"
            )

        # Set device
        device = torch.device(self.hyperparams["device"])

        # Create OTFM model for our single dataset
        x_tensor = torch.tensor(adata.X, dtype=torch.float32)
        t_idx = torch.tensor(adata.obs["t"], dtype=torch.long)

        otfm = EntropicOTFM(
            x=x_tensor,
            t_idx=t_idx,
            dt=DT_data,
            sigma=self.hyperparams["sigma"],
            T=T_times,
            dim=num_vars,
            device=device,
            held_out_time=None,
        )

        if not self.silent:
            print(f"  SF2M Debug - Created OTFM model")

        # Create SF2M neural networks
        dims = [num_vars, self.hyperparams["knockout_hidden"], 1]

        func_v = MLPODEF(
            dims=dims,
            GL_reg=self.hyperparams["gl_reg"],
            bias=True,
        )

        score_net = CONDMLP(
            d=num_vars,
            hidden_sizes=self.hyperparams["score_hidden"],
            time_varying=True,
            conditional=True,
            conditional_dim=num_vars,
            device=device,
        )

        v_correction = MLP(
            d=num_vars,
            hidden_sizes=self.hyperparams["correction_hidden"],
            time_varying=True,
        )

        # Move models to device
        func_v = func_v.to(device)
        score_net = score_net.to(device)
        v_correction = v_correction.to(device)

        if not self.silent:
            print(f"  SF2M Debug - Created neural networks")

        # Create conditional vector (all zeros for wild-type)
        # Use adaptive batch size based on dataset size
        adaptive_batch_size = min(self.hyperparams["batch_size"], num_samples // 4)
        cond_vector = torch.zeros(adaptive_batch_size, num_vars).to(device)

        # Setup optimizer
        params_to_optimize = (
            list(func_v.parameters())
            + list(score_net.parameters())
            + list(v_correction.parameters())
        )

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.hyperparams["lr"],
            eps=1e-7,
        )

        if not self.silent:
            print(
                f"  SF2M Debug - Starting training for {self.hyperparams['n_steps']} steps with batch_size={adaptive_batch_size}..."
            )
        else:
            print(
                f"  SF2M: Training {self.hyperparams['n_steps']} steps (batch_size={adaptive_batch_size})..."
            )

        # Training loop
        for step in range(self.hyperparams["n_steps"]):
            optimizer.zero_grad()

            # Sample bridging flows from OTFM
            _x, _s, _u, _t, _t_orig = otfm.sample_bridges_flows(
                batch_size=adaptive_batch_size, skip_time=None
            )

            # Move to device
            _x = _x.to(device)
            _s = _s.to(device)
            _u = _u.to(device)
            _t = _t.to(device)
            _t_orig = _t_orig.to(device)

            B = _x.shape[0]

            # Expand conditional vector to match batch size
            cond_expanded = cond_vector.repeat(B // cond_vector.shape[0] + 1, 1)[:B]

            # Prepare inputs
            v_input = _x.unsqueeze(1)
            t_input = _t.unsqueeze(1)

            # Score net output
            s_fit = score_net(_t, _x, cond_expanded).squeeze(1)

            # Flow net output with correction (after warmup)
            if step <= 500:
                # Warmup phase
                v_fit = func_v(t_input, v_input).squeeze(1) - (
                    self.hyperparams["sigma"] ** 2 / 2
                ) * score_net(_t, _x, cond_expanded)
            else:
                # Full training phase with correction
                v_fit = func_v(t_input, v_input).squeeze(1) + v_correction(_t, _x)
                v_fit = v_fit - (self.hyperparams["sigma"] ** 2 / 2) * score_net(
                    _t, _x, cond_expanded
                )

            # Compute losses
            L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
            L_flow = torch.mean((_t_orig * (1 - _t_orig)) * (v_fit * DT_data - _u) ** 2)
            L_reg = func_v.l2_reg() + func_v.fc1_reg()

            # L2 regularization for correction network
            L_reg_correction = 0.0
            for param in v_correction.parameters():
                L_reg_correction += torch.sum(param**2)

            # Loss combination logic
            if step < 100:
                # Train only score initially
                L = self.hyperparams["alpha"] * L_score
            elif step <= 500:
                # Mix score + flow + small reg
                L = (
                    self.hyperparams["alpha"] * L_score
                    + (1 - self.hyperparams["alpha"]) * L_flow
                    + self.hyperparams["reg"] * L_reg
                )
            else:
                # Full combined loss with correction reg
                L = (
                    self.hyperparams["alpha"] * L_score
                    + (1 - self.hyperparams["alpha"]) * L_flow
                    + self.hyperparams["reg"] * L_reg
                    + self.hyperparams["reg"] * L_reg_correction
                )

            # Backprop and update
            L.backward()
            optimizer.step()

            # Proximal step (group-lasso style)
            with torch.no_grad():
                # Proximal operator for group lasso regularization
                w = func_v.fc1.weight
                d = dims[0]
                d_hidden = dims[1]
                wadj = w.view(d, d_hidden, d)
                tmp = (
                    torch.sum(wadj**2, dim=1).sqrt() - self.hyperparams["gl_reg"] * 0.01
                )
                alpha_ = torch.clamp(tmp, min=0)
                v_ = torch.nn.functional.normalize(wadj, dim=1) * alpha_[:, None, :]
                w.copy_(v_.view(-1, d))

            if step % 200 == 0:
                if not self.silent:
                    print(
                        f"    Step {step}/{self.hyperparams['n_steps']}, Loss: {L.item():.4f}"
                    )
                elif step % 1000 == 0:  # Show progress every 1000 steps in silent mode
                    print(f"    Step {step}/{self.hyperparams['n_steps']}")

        if not self.silent:
            print(f"  SF2M Debug - Training completed")

        # Extract causal graph
        func_v.eval()
        with torch.no_grad():
            # Get the causal graph from the velocity field
            W_v_raw = func_v.causal_graph(w_threshold=0.0)

            # Handle both tensor and numpy array cases
            if isinstance(W_v_raw, torch.Tensor):
                W_v_np = W_v_raw.cpu().numpy()
            else:
                W_v_np = W_v_raw

            if not self.silent:
                print(f"  SF2M Debug - Raw W_v shape: {W_v_np.shape}")

            # Use consistent orientation based on our matrix convention understanding
            # SF2M's causal_graph() returns W[i,j] meaning variable j influences variable i's dynamics
            # Our convention: adjacency[i,j] means variable i → variable j
            # Therefore, we use W.T to match our convention
            predicted_adjacency = W_v_np.T

            # Zero out diagonal values - we don't need self-loops
            np.fill_diagonal(predicted_adjacency, 0)

            if not self.silent:
                print(
                    f"  SF2M Debug - Using W_v.T (transposed) based on matrix conventions"
                )

            # If we have true adjacency, show both orientations for debugging but don't use for selection
            if true_adjacency is not None:
                if not self.silent:
                    print(f"  SF2M Debug - Matrix comparison:")
                    print("Ground Truth:")
                    np.set_printoptions(precision=2, suppress=True)
                    print(true_adjacency)

                    print(f"SF2M direct:")
                    W_v_direct = W_v_np.copy()
                    np.fill_diagonal(W_v_direct, 0)
                    print(W_v_direct)

                    print(f"SF2M.T (used):")
                    print(predicted_adjacency)

                # Zero out diagonal for evaluation (self-loops not considered)
                true_adj_eval = true_adjacency.copy()
                np.fill_diagonal(true_adj_eval, 0)

                # Evaluate version 1 (direct W_v)
                pred_v1 = W_v_np.copy()
                np.fill_diagonal(pred_v1, 0)
                metrics_v1 = evaluate_causal_discovery(true_adj_eval, pred_v1)
                if not self.silent:
                    print(
                        f"    Direct W_v: AUROC={metrics_v1['AUROC']:.4f}, AUPRC={metrics_v1['AUPRC']:.4f}"
                    )

                # Evaluate version 2 (W_v.T) - this is what we're using
                metrics_v2 = evaluate_causal_discovery(
                    true_adj_eval, predicted_adjacency
                )
                if not self.silent:
                    print(
                        f"    W_v.T (used): AUROC={metrics_v2['AUROC']:.4f}, AUPRC={metrics_v2['AUPRC']:.4f}"
                    )

        self._training_time = time.time() - start_time
        return predicted_adjacency


class NGMNodeMethod(CausalDiscoveryMethod):
    """NGM-NODE implementation using Neural ODEs and optimal transport."""

    def __init__(self, hyperparams: Dict[str, Any], silent: bool = False):
        super().__init__("NGM-NODE")
        self.hyperparams = hyperparams
        self.needs_true_adjacency = True
        self.model = None
        self.silent = silent

    def fit(
        self, time_series_data: np.ndarray, true_adjacency: np.ndarray = None
    ) -> np.ndarray:
        """
        Fit NGM-NODE model using Neural ODEs and optimal transport.

        Args:
            time_series_data: Shape (num_timepoints, num_samples, num_vars)

        Returns:
            predicted_adjacency: Predicted adjacency matrix from Jacobian
        """
        start_time = time.time()

        num_timepoints, num_samples, num_vars = time_series_data.shape
        if not self.silent:
            print(f"  NGM-NODE Debug - Input data shape: {time_series_data.shape}")

        # Convert to list of tensors for each timepoint
        data_by_timepoint = []
        for t in range(num_timepoints):
            data_t = torch.from_numpy(time_series_data[t]).float()
            data_by_timepoint.append(data_t)

        if not self.silent:
            print(
                f"  NGM-NODE Debug - Created {len(data_by_timepoint)} timepoint tensors"
            )

        # Set device
        device = torch.device(self.hyperparams["device"])

        # Create Neural ODE function using MLPODEF
        dims = [num_vars, self.hyperparams["hidden_dim"], 1]
        func_ode = MLPODEF(
            dims=dims,
            GL_reg=self.hyperparams["gl_reg"],
            bias=True,
        )
        func_ode = func_ode.to(device)

        if not self.silent:
            print(f"  NGM-NODE Debug - Created MLPODEF with dims {dims}")

        # Setup optimizer
        optimizer = torch.optim.Adam(
            func_ode.parameters(),
            lr=self.hyperparams["lr"],
        )

        # Setup optimal transport loss
        sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

        if not self.silent:
            print(
                f"  NGM-NODE Debug - Starting training for {self.hyperparams['n_steps']} steps..."
            )
        else:
            print(f"  NGM-NODE: Training {self.hyperparams['n_steps']} steps...")

        # Training loop
        num_transitions = num_timepoints - 1
        transition_times = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device)

        for step in range(self.hyperparams["n_steps"]):
            optimizer.zero_grad()

            # 1. Sample a random transition
            transition_idx = np.random.randint(0, num_transitions)

            # 2. Sample random cells from start and end times
            x0_data = data_by_timepoint[transition_idx]
            x1_data = data_by_timepoint[transition_idx + 1]

            batch_size = min(
                self.hyperparams["batch_size"], x0_data.shape[0], x1_data.shape[0]
            )

            # Sample batches
            indices_0 = torch.randint(0, x0_data.shape[0], (batch_size,))
            indices_1 = torch.randint(0, x1_data.shape[0], (batch_size,))

            x0_batch = x0_data[indices_0].to(device)  # Starting states
            x1_observed = x1_data[indices_1].to(device)  # Target states

            # 3. Use Neural ODE to predict where x0 goes
            # Add batch dimension for ODE integration
            x0_ode_input = x0_batch.unsqueeze(1)  # Shape: (batch_size, 1, num_vars)

            # Integrate ODE forward in time
            x_trajectory = odeint(func_ode, x0_ode_input, transition_times)
            x1_predicted = x_trajectory[-1].squeeze(
                1
            )  # Final timepoint, remove time dim

            # 4. Compare predicted vs observed distributions using optimal transport
            loss = sinkhorn_loss(x1_predicted, x1_observed)

            # Add regularization
            if self.hyperparams.get("l2_reg", 0) > 0:
                loss += self.hyperparams["l2_reg"] * func_ode.l2_reg()
            if self.hyperparams.get("l1_reg", 0) > 0:
                loss += self.hyperparams["l1_reg"] * func_ode.fc1_reg()

            # 5. Backprop and update
            loss.backward()
            optimizer.step()

            # Apply proximal operator for group lasso
            with torch.no_grad():
                w = func_ode.fc1.weight
                d = dims[0]
                d_hidden = dims[1]
                wadj = w.view(d, d_hidden, d)
                tmp = (
                    torch.sum(wadj**2, dim=1).sqrt() - self.hyperparams["gl_reg"] * 0.01
                )
                alpha_ = torch.clamp(tmp, min=0)
                v_ = torch.nn.functional.normalize(wadj, dim=1) * alpha_[:, None, :]
                w.copy_(v_.view(-1, d))

            if step % 200 == 0:
                if not self.silent:
                    print(
                        f"    Step {step}/{self.hyperparams['n_steps']}, Loss: {loss.item():.4f}"
                    )
                elif step % 1000 == 0:
                    print(f"    Step {step}/{self.hyperparams['n_steps']}")

        if not self.silent:
            print(f"  NGM-NODE Debug - Training completed, extracting causal graph...")

        # Extract causal graph directly from network weights (same as SF2M)
        func_ode.eval()
        with torch.no_grad():
            # Get the causal graph from the network weights
            W_v_raw = func_ode.causal_graph(w_threshold=0.0)

            # Handle both tensor and numpy array cases
            if isinstance(W_v_raw, torch.Tensor):
                W_v_np = W_v_raw.cpu().numpy()
            else:
                W_v_np = W_v_raw

            if not self.silent:
                print(f"  NGM-NODE Debug - Raw W_v shape: {W_v_np.shape}")

            # Use consistent orientation (same as SF2M)
            predicted_adjacency = W_v_np.T

            # Zero out diagonal values
            np.fill_diagonal(predicted_adjacency, 0)

            if not self.silent:
                print(
                    f"  NGM-NODE Debug - Using W_v.T (transposed) for consistency with SF2M"
                )
                if true_adjacency is not None:
                    print("Ground Truth:")
                    np.set_printoptions(precision=2, suppress=True)
                    print(true_adjacency)
                    print("NGM-NODE Predicted:")
                    print(predicted_adjacency)

        self._training_time = time.time() - start_time
        return predicted_adjacency


def maskdiag(A):
    """Mask diagonal elements to zero, following the approach in plotting.py."""
    A_masked = A.copy()
    np.fill_diagonal(A_masked, 0)
    return A_masked


def evaluate_causal_discovery(
    true_adjacency: np.ndarray, predicted_adjacency: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate causal discovery performance with AUROC and AUPRC.

    Uses the same approach as plotting.py for standardization.

    Args:
        true_adjacency: True adjacency matrix
        predicted_adjacency: Predicted adjacency matrix

    Returns:
        metrics: Dictionary containing AUROC and AUPRC
    """
    # Mask diagonal elements (following plotting.py approach)
    masked_true = maskdiag(true_adjacency)
    masked_pred = maskdiag(predicted_adjacency)

    # Convert to binary ground truth (following plotting.py)
    y_true = np.abs(np.sign(masked_true).astype(int).flatten())

    # Use absolute values of predictions as confidence scores (following plotting.py)
    y_pred = np.abs(masked_pred.flatten())

    # Calculate metrics using sklearn (same as plotting.py)
    if len(np.unique(y_true)) < 2:
        # Handle edge case: all edges are 0 or all edges are 1
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "num_true_edges": np.sum(y_true),
        "num_possible_edges": len(y_true),
    }


def generate_test_causal_system(num_vars: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a specific test causal system to distinguish transpose orientations.

    Creates a graph where variable 0 affects all other variables (first row all 1s).
    This creates a clear pattern that should only match one orientation.

    Args:
        num_vars: Number of variables in the system

    Returns:
        adjacency_matrix: True causal adjacency matrix
        dynamics_matrix: System dynamics matrix for simulation
    """
    # Create adjacency matrix: variable 0 affects all others
    adjacency_matrix = np.zeros((num_vars, num_vars))
    adjacency_matrix[0, 1:] = 1.0  # Variable 0 → all others

    # Add a few more edges to make it more realistic but keep the pattern clear
    if num_vars >= 4:
        adjacency_matrix[1, 3] = 0.5  # Variable 1 → Variable 3
    if num_vars >= 5:
        adjacency_matrix[2, 4] = 0.5  # Variable 2 → Variable 4

    # Dynamics matrix is transpose
    dynamics_matrix = adjacency_matrix.T

    return adjacency_matrix, dynamics_matrix


def run_transpose_test():
    """Run a specific test to determine correct transpose orientation."""
    print("\n" + "=" * 60)
    print("TRANSPOSE ORIENTATION TEST")
    print("=" * 60)

    # Create test system with clear pattern
    test_size = 10
    true_adjacency, dynamics_matrix = generate_test_causal_system(test_size)

    print("Test system: Variable 0 affects all others")
    print("True adjacency matrix (adjacency[i,j] = variable i → variable j):")
    np.set_printoptions(precision=2, suppress=True)
    print(true_adjacency)

    print(
        "\nDynamics matrix (dynamics[i,j] = variable j affects derivative of variable i):"
    )
    print(dynamics_matrix)

    # Simulate data
    time_series_data = simulate_time_series(
        dynamics_matrix, seed=random.randint(0, 1000000)
    )

    # Test both Correlation and SF2M methods
    correlation_method = CorrelationBasedMethod("pearson")

    # Create SF2M method with base config
    sf2m_config = SF2MConfig(
        base_n_steps=2000,
        base_lr=0.001,
        base_alpha=0.3,
        base_reg=1e-06,
        base_gl_reg=0.02,
        base_knockout_hidden=64,
        base_score_hidden=[32, 32],
        base_correction_hidden=[16, 16],
        base_batch_size=32,
        sigma=1.0,
        device="cpu",
    )

    sf2m_method = DirectSF2MMethod(
        {
            "n_steps": sf2m_config.base_n_steps,
            "lr": sf2m_config.base_lr,
            "alpha": sf2m_config.base_alpha,
            "reg": sf2m_config.base_reg,
            "gl_reg": sf2m_config.base_gl_reg,
            "knockout_hidden": sf2m_config.base_knockout_hidden,
            "score_hidden": sf2m_config.base_score_hidden,
            "correction_hidden": sf2m_config.base_correction_hidden,
            "batch_size": sf2m_config.base_batch_size,
            "sigma": sf2m_config.sigma,
            "device": sf2m_config.device,
        },
        silent=False,
    )

    # Run Correlation method first
    print(f"\nRunning Correlation method...")
    correlation_pred = correlation_method.fit(time_series_data)
    print("Correlation predicted matrix:")
    print(correlation_pred)

    # Analyze correlation pattern
    print("\nCorrelation analysis:")
    print(f"Row 0 sum: {np.sum(correlation_pred[0, :]):.3f}")
    print(f"Col 0 sum: {np.sum(correlation_pred[:, 0]):.3f}")
    print(f"Row 0 values: {correlation_pred[0, :]}")
    print(f"Col 0 values: {correlation_pred[:, 0]}")

    # Run SF2M
    print(f"\nRunning SF2M method...")
    sf2m_pred = sf2m_method.fit(time_series_data, true_adjacency)

    # Analyze SF2M pattern
    print("\nSF2M analysis:")
    print(f"Row 0 sum: {np.sum(sf2m_pred[0, :]):.3f}")
    print(f"Col 0 sum: {np.sum(sf2m_pred[:, 0]):.3f}")
    print(f"Row 0 values: {sf2m_pred[0, :]}")
    print(f"Col 0 values: {sf2m_pred[:, 0]}")

    print(f"\nTraining completed in {sf2m_method.get_training_time():.2f}s")

    # Final diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("If Correlation shows ONLY row 0 pattern → Data generation is correct")
    print("If Correlation shows BOTH row 0 AND col 0 pattern → Data generation issue")
    print("If SF2M differs from Correlation → SF2M-specific issue")
    print("=" * 60)


def run_single_experiment(
    num_vars: int, methods: List[CausalDiscoveryMethod], seed: int = 42
) -> Dict[str, Any]:
    """
    Run causal discovery experiment for a single system size.

    Args:
        num_vars: Number of variables in the system
        methods: List of causal discovery methods to test
        seed: Random seed

    Returns:
        results: Dictionary containing results for all methods
    """
    print(f"Running experiment for {num_vars} variables, seed {seed}...")

    # Generate causal system
    true_adjacency, dynamics_matrix = generate_causal_system(num_vars, seed=seed)

    # Simulate time series data
    time_series_data = simulate_time_series(dynamics_matrix, seed=seed)

    # Basic system info
    results = {
        "num_vars": num_vars,
        "seed": seed,
        "true_edges": np.sum(np.abs(true_adjacency) > 0),
        "edge_density": np.sum(np.abs(true_adjacency) > 0) / (num_vars * num_vars),
        "max_eigenvalue_real": np.max(np.real(np.linalg.eigvals(dynamics_matrix))),
    }

    # Store predicted matrices for inspection (only for quick test)
    predicted_matrices = {}

    # Test each method
    for method in methods:
        print(f"  Testing {method.name}...")

        # Fit method and get predictions
        if hasattr(method, "needs_true_adjacency") and method.needs_true_adjacency:
            predicted_adjacency = method.fit(time_series_data, true_adjacency)
        else:
            predicted_adjacency = method.fit(time_series_data)

        # Store the predicted matrix for inspection during quick tests
        predicted_matrices[method.name] = predicted_adjacency

        # Evaluate performance
        metrics = evaluate_causal_discovery(true_adjacency, predicted_adjacency)
        training_time = method.get_training_time()

        # Store results
        results[f"{method.name}_AUROC"] = metrics["AUROC"]
        results[f"{method.name}_AUPRC"] = metrics["AUPRC"]
        results[f"{method.name}_training_time"] = training_time
        results[f"{method.name}_num_true_edges"] = metrics["num_true_edges"]

        print(
            f"    AUROC: {metrics['AUROC']:.4f}, AUPRC: {metrics['AUPRC']:.4f}, Time: {training_time:.4f}s"
        )

    return results


def run_single_experiment_silent(
    num_vars: int, methods: List[CausalDiscoveryMethod], seed: int = 42
) -> Dict[str, Any]:
    """
    Run causal discovery experiment for a single system size with minimal output.
    This version suppresses all debug printing and matrix output for hyperparameter sweeps.

    Args:
        num_vars: Number of variables in the system
        methods: List of causal discovery methods to test
        seed: Random seed

    Returns:
        results: Dictionary containing results for all methods
    """
    # Generate causal system
    true_adjacency, dynamics_matrix = generate_causal_system(num_vars, seed=seed)

    # Simulate time series data
    time_series_data = simulate_time_series(dynamics_matrix, seed=seed)

    # Basic system info
    results = {
        "num_vars": num_vars,
        "seed": seed,
        "true_edges": np.sum(np.abs(true_adjacency) > 0),
        "edge_density": np.sum(np.abs(true_adjacency) > 0) / (num_vars * num_vars),
        "max_eigenvalue_real": np.max(np.real(np.linalg.eigvals(dynamics_matrix))),
    }

    # Test each method
    for method in methods:
        # Fit method and get predictions
        if hasattr(method, "needs_true_adjacency") and method.needs_true_adjacency:
            predicted_adjacency = method.fit(time_series_data, true_adjacency)
        else:
            predicted_adjacency = method.fit(time_series_data)

        # Evaluate performance
        metrics = evaluate_causal_discovery(true_adjacency, predicted_adjacency)
        training_time = method.get_training_time()

        # Store results
        results[f"{method.name}_AUROC"] = metrics["AUROC"]
        results[f"{method.name}_AUPRC"] = metrics["AUPRC"]
        results[f"{method.name}_training_time"] = training_time
        results[f"{method.name}_num_true_edges"] = metrics["num_true_edges"]

        # Print performance metrics even in silent mode
        print(
            f"    {method.name}: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, Time={training_time:.4f}s"
        )

    return results


def run_scaling_experiment(
    system_sizes: List[int],
    sf2m_config: SF2MConfig,
    seeds: List[int] = [42],
    num_cores: int = 4,
) -> pd.DataFrame:
    """
    Run causal discovery scaling experiment with fixed hyperparameters.

    Args:
        system_sizes: List of system sizes to test
        sf2m_config: SF2M configuration object (uses base config for all sizes)
        seeds: List of random seeds for multiple runs
        num_cores: Number of cores to use for reproducible timing

    Returns:
        results_df: DataFrame containing all experimental results
    """
    # Set fixed cores for reproducible runtime analysis
    set_fixed_cores(num_cores)

    all_results = []
    total_experiments = len(seeds) * len(system_sizes)
    current_exp = 0

    start_time = time.time()

    # Create fixed hyperparameters (same for all system sizes)
    fixed_hyperparams_sf2m = {
        "n_steps": sf2m_config.base_n_steps,
        "lr": sf2m_config.base_lr,
        "alpha": sf2m_config.base_alpha,
        "reg": sf2m_config.base_reg,
        "gl_reg": sf2m_config.base_gl_reg,
        "knockout_hidden": sf2m_config.base_knockout_hidden,
        "score_hidden": sf2m_config.base_score_hidden,
        "correction_hidden": sf2m_config.base_correction_hidden,
        "batch_size": sf2m_config.base_batch_size,
        "sigma": sf2m_config.sigma,
        "device": sf2m_config.device,
    }

    # NGM-NODE hyperparameters (simpler, fewer steps due to ODE cost)
    fixed_hyperparams_ngm = {
        "n_steps": 4000,  # Fewer steps due to ODE integration cost
        "lr": 0.005,
        "gl_reg": 0.05,
        "hidden_dim": 128,  # Single hidden layer size
        "batch_size": 32,
        "l2_reg": 0.01,
        "l1_reg": 0.0,
        "device": sf2m_config.device,
    }

    print(
        f"  SF2M steps={fixed_hyperparams_sf2m['n_steps']}, knockout_hidden={fixed_hyperparams_sf2m['knockout_hidden']}"
    )
    print(
        f"  NGM-NODE steps={fixed_hyperparams_ngm['n_steps']}, hidden_dim={fixed_hyperparams_ngm['hidden_dim']}"
    )

    for seed in seeds:
        for num_vars in system_sizes:
            current_exp += 1
            print(
                f"\n[{current_exp}/{total_experiments}] System size: {num_vars}, Seed: {seed}"
            )

            # Create methods with fixed configurations
            methods = [
                CorrelationBasedMethod("pearson"),
                # DirectSF2MMethod(fixed_hyperparams_sf2m, silent=True),
                NGMNodeMethod(fixed_hyperparams_ngm, silent=True),
            ]

            result = run_single_experiment_silent(num_vars, methods, seed)
            result["experiment_id"] = current_exp
            result["total_experiments"] = total_experiments
            all_results.append(result)

    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time:.2f}s")

    results_df = pd.DataFrame(all_results)
    return results_df


def save_and_print_results(
    results_df: pd.DataFrame, output_file: str = "causal_discovery_results.csv"
):
    """Save results to CSV and print summary statistics."""

    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print detailed summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Get method names
    method_names = [
        col.replace("_AUROC", "")
        for col in results_df.columns
        if col.endswith("_AUROC")
    ]

    print(f"System sizes tested: {sorted(results_df['num_vars'].unique())}")
    print(f"Seeds used: {sorted(results_df['seed'].unique())}")
    print(f"Methods tested: {method_names}")
    print(f"Total experiments: {len(results_df)}")

    # Performance summary by system size
    print("\nPERFORMANCE BY SYSTEM SIZE:")
    print("-" * 40)

    for method_name in method_names:
        print(f"\n{method_name}:")
        summary = (
            results_df.groupby("num_vars")
            .agg(
                {
                    f"{method_name}_AUROC": ["mean", "std", "min", "max"],
                    f"{method_name}_AUPRC": ["mean", "std", "min", "max"],
                    f"{method_name}_training_time": ["mean", "std", "min", "max"],
                }
            )
            .round(4)
        )

        print(summary)

    # Overall statistics
    print("\nOVERALL STATISTICS:")
    print("-" * 20)
    for method_name in method_names:
        auroc_mean = results_df[f"{method_name}_AUROC"].mean()
        auprc_mean = results_df[f"{method_name}_AUPRC"].mean()
        time_mean = results_df[f"{method_name}_training_time"].mean()

        print(
            f"{method_name}: AUROC={auroc_mean:.4f}, AUPRC={auprc_mean:.4f}, Time={time_mean:.4f}s"
        )


def main():
    """Main experiment runner."""

    # Set fixed cores for reproducible timing
    NUM_CORES = 4

    print("=" * 60)
    print("SF2M CAUSAL DISCOVERY SCALING EXPERIMENT")
    print("=" * 60)

    # from hparam sweep, top config: n_steps=2000, batch_size=64, reg=1e-06, alpha=0.3, lr=0.001, knockout_hidden=256
    sf2m_config = SF2MConfig(
        # Base parameters for N=10 (will be scaled linearly)
        base_n_steps=4000,
        base_lr=5e-4,
        base_alpha=0.3,
        base_reg=5e-07,
        base_gl_reg=0.01,
        base_knockout_hidden=256,
        base_score_hidden=[128, 128],
        base_correction_hidden=[64, 64],
        base_batch_size=64,
        sigma=1.0,
        device="cpu",
    )

    # Define system sizes to test
    system_sizes = [10, 20, 50, 100, 200]

    print(f"\nScaling experiment setup:")
    print(f"  System sizes: {system_sizes}")
    print(f"  Seeds per size: 5 (for averaging over different random graphs)")
    print(f"  Total experiments: {len(system_sizes) * 5}")
    print(f"  Fixed hyperparameters (no scaling with system size)")

    # Run experiments
    print(f"\nStarting scaling experiment with {NUM_CORES} cores...")

    # Generate 5 random seeds
    random_seeds = [random.randint(0, 10000) for _ in range(5)]
    print(f"Using random seeds: {random_seeds}")

    results_df = run_scaling_experiment(
        system_sizes, sf2m_config, seeds=random_seeds, num_cores=NUM_CORES
    )

    # Save results and print summary
    save_and_print_results(results_df, "scaling_experiment_results.csv")


if __name__ == "__main__":
    main()
