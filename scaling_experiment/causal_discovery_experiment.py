import numpy as np
import torch
import networkx as nx
import pandas as pd
import time
from typing import List, Dict, Any, Tuple
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression
import multiprocessing as mp


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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
    num_timepoints: int = 10,
    num_samples: int = 500,
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


def evaluate_causal_discovery(
    true_adjacency: np.ndarray, predicted_adjacency: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate causal discovery performance with AUROC and AUPRC.

    Args:
        true_adjacency: True adjacency matrix
        predicted_adjacency: Predicted adjacency matrix

    Returns:
        metrics: Dictionary containing AUROC and AUPRC
    """
    # Convert to binary ground truth (any non-zero edge is a true edge)
    true_edges = (np.abs(true_adjacency) > 0).astype(int).flatten()

    # Use absolute values of predictions as confidence scores
    pred_scores = np.abs(predicted_adjacency).flatten()

    # Calculate metrics
    try:
        if len(np.unique(true_edges)) < 2:
            # Handle edge case: all edges are 0 or all edges are 1
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = roc_auc_score(true_edges, pred_scores)
            auprc = average_precision_score(true_edges, pred_scores)
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        auroc = float("nan")
        auprc = float("nan")

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "num_true_edges": np.sum(true_edges),
        "num_possible_edges": len(true_edges),
    }


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

    # Test each method
    for method in methods:
        print(f"  Testing {method.name}...")

        # Fit method and get predictions
        predicted_adjacency = method.fit(time_series_data)

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


def run_scaling_experiment(
    system_sizes: List[int],
    methods: List[CausalDiscoveryMethod],
    seeds: List[int] = [42],
    num_cores: int = 4,
) -> pd.DataFrame:
    """
    Run causal discovery scaling experiment.

    Args:
        system_sizes: List of system sizes to test
        methods: List of causal discovery methods
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

    for seed in seeds:
        for num_vars in system_sizes:
            current_exp += 1
            print(
                f"\n[{current_exp}/{total_experiments}] System size: {num_vars}, Seed: {seed}"
            )

            result = run_single_experiment(num_vars, methods, seed)
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

    # Define system sizes to test
    system_sizes = [10, 20, 50]

    # Define methods to test
    methods = [
        CorrelationBasedMethod("pearson"),
    ]

    # Run experiments
    print("Starting causal discovery scaling experiment...")
    print(f"Using {NUM_CORES} cores for reproducible timing")

    results_df = run_scaling_experiment(
        system_sizes, methods, seeds=[42, 123, 456], num_cores=NUM_CORES
    )

    # Save results and print summary
    save_and_print_results(results_df)


if __name__ == "__main__":
    main()
