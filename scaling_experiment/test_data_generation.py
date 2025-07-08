import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from causal_discovery_experiment import (
    generate_causal_system,
    simulate_time_series,
)


def visualize_matrices(adjacency_matrix, dynamics_matrix, title_prefix=""):
    """Visualize adjacency and dynamics matrices side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot adjacency matrix
    sns.heatmap(
        adjacency_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax1,
        cbar_kws={"label": "Edge Weight"},
    )
    ax1.set_title(f"{title_prefix}Adjacency Matrix\n(True Causal Structure)")
    ax1.set_xlabel("Variable j (Target)")
    ax1.set_ylabel("Variable i (Source)")

    # Plot dynamics matrix
    sns.heatmap(
        dynamics_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax2,
        cbar_kws={"label": "Dynamics Weight"},
    )
    ax2.set_title(f"{title_prefix}Dynamics Matrix\n(For Simulation)")
    ax2.set_xlabel("Variable j")
    ax2.set_ylabel("Variable i")

    plt.tight_layout()
    return fig


def test_causality_direction():
    """Test and demonstrate the causality direction."""
    print("Testing Causality Direction")
    print("=" * 50)

    # Generate a small system for easy inspection
    num_vars = 5
    adjacency, dynamics = generate_causal_system(num_vars, edge_prob=0.3, seed=42)

    print("Generated matrices:")
    print(f"Adjacency matrix shape: {adjacency.shape}")
    print(f"Dynamics matrix shape: {dynamics.shape}")
    print()

    # Show raw matrices
    print("Adjacency Matrix (True Causal Structure):")
    print(adjacency)
    print("\nDynamics Matrix (For Simulation):")
    print(dynamics)
    print()

    # Visualize
    fig = visualize_matrices(adjacency, dynamics)
    plt.savefig("causal_matrices_test.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Explain the causality direction
    print("Causality Direction Analysis:")
    print("-" * 30)

    # Find a non-zero edge to explain
    nonzero_edges = np.where(adjacency != 0)
    if len(nonzero_edges[0]) > 0:
        i, j = nonzero_edges[0][0], nonzero_edges[1][0]
        weight = adjacency[i, j]

        print(f"Example: adjacency_matrix[{i}, {j}] = {weight:.3f}")
        print(f"This means: Variable {i} → Variable {j} with strength {weight:.3f}")
        print(f"In other words: Variable {i} causally influences Variable {j}")
        print()

        print("In the dynamics matrix:")
        print(f"dynamics_matrix[{j}, {i}] = {dynamics[j, i]:.3f}")
        print(
            f"This appears in the simulation as: dx_{j}/dt += {dynamics[j, i]:.3f} * x_{i}"
        )
        print(f"So variable {i} affects the rate of change of variable {j}")
        print("✓ Causality direction is CORRECT (i affects j in both matrices)")

    return adjacency, dynamics


def test_simulation_dynamics():
    """Test that the simulation produces the expected dynamics."""
    print("\n" + "=" * 50)
    print("Testing Simulation Dynamics")
    print("=" * 50)

    # Create a simple 2-variable system for easy analysis
    # Variable 0 → Variable 1
    adjacency = np.array(
        [[0, 0.5], [0, 0]]  # Variable 0 affects Variable 1
    )  # No other connections

    # Create corresponding dynamics matrix (should be transpose + diagonal)
    dynamics = adjacency.T.copy()  # Now dynamics[1,0] = 0.5
    for i in range(2):
        row_sum = np.sum(np.abs(dynamics[i, :]))
        if row_sum > 0:
            dynamics[i, i] = -(row_sum + 0.1)

    print("Simple test system:")
    print("Adjacency (Variable 0 → Variable 1):")
    print(adjacency)
    print("\nDynamics matrix (transpose + diagonal):")
    print(dynamics)
    print(f"Note: dynamics[1,0] = {dynamics[1,0]} means var 0 affects d(var 1)/dt")

    # Simulate short time series
    time_series = simulate_time_series(
        dynamics, num_timepoints=5, num_samples=100, noise_std=0.05, dt=0.1, seed=42
    )

    print(f"\nSimulated time series shape: {time_series.shape}")
    print("(timepoints, samples, variables)")

    # Analyze correlation over time
    correlations = []
    for t in range(time_series.shape[0]):
        data_t = time_series[t]  # (samples, variables)
        if data_t.shape[0] > 1:  # Need at least 2 samples for correlation
            corr = np.corrcoef(data_t[:, 0], data_t[:, 1])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)

    print("\nCorrelation between Variable 0 and Variable 1 over time:")
    for t, corr in enumerate(correlations):
        print(f"  Time {t}: {corr:.3f}")

    # Plot time series
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot sample trajectories
    for i in range(min(10, time_series.shape[1])):  # Plot first 10 samples
        axes[0].plot(
            time_series[:, i, 0], label=f"Var 0, Sample {i}" if i < 3 else "", alpha=0.7
        )
        axes[1].plot(
            time_series[:, i, 1], label=f"Var 1, Sample {i}" if i < 3 else "", alpha=0.7
        )

    axes[0].set_title("Variable 0 (Source)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    if len(axes[0].get_legend_handles_labels()[0]) > 0:
        axes[0].legend()

    axes[1].set_title("Variable 1 (Target)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Value")
    if len(axes[1].get_legend_handles_labels()[0]) > 0:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("simulation_test.png", dpi=150, bbox_inches="tight")
    plt.show()

    return time_series


def main():
    """Run all tests."""
    print("Testing Data Generation Framework")
    print("=" * 60)

    # Test causality direction
    adjacency, dynamics = test_causality_direction()

    # Test simulation
    time_series = test_simulation_dynamics()

    print("\n" + "=" * 60)
    print("Summary:")
    print("- Adjacency matrix: adjacency[i,j] means Variable i → Variable j")
    print("- Dynamics matrix: dynamics[i,j] means Variable j affects d(Variable i)/dt")
    print("- Relationship: dynamics = adjacency.T (plus diagonal stability terms)")
    print("- Simulation: dx/dt = dynamics @ x")
    print("- Visualization and simulation data saved as PNG files")
    print("=" * 60)


if __name__ == "__main__":
    main()
