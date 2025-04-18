import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- Input data: ODE means and stds for Synthetic dataset -----
# Mean values
w_mean = {
    "Overall": {"sf2m": 0.6782, "rf": 0.8732, "mlp_baseline": 0.6491},
    "Left‑out 1": {"sf2m": 0.9004, "rf": 1.1446, "mlp_baseline": 0.8043},
    "Left‑out 2": {"sf2m": 0.9126, "rf": 1.1214, "mlp_baseline": 0.9299},
    "Left‑out 3": {"sf2m": 0.5121, "rf": 0.8457, "mlp_baseline": 0.4740},
    "Left‑out 4": {"sf2m": 0.3878, "rf": 0.3813, "mlp_baseline": 0.3883},
}
w_std = {
    "Overall": {"sf2m": 0.0191, "rf": 0.0105, "mlp_baseline": 0.0226},
    "Left‑out 1": {"sf2m": 0.0256, "rf": 0.0062, "mlp_baseline": 0.0225},
    "Left‑out 2": {"sf2m": 0.0108, "rf": 0.0073, "mlp_baseline": 0.0189},
    "Left‑out 3": {"sf2m": 0.0140, "rf": 0.0075, "mlp_baseline": 0.0248},
    "Left‑out 4": {"sf2m": 0.0259, "rf": 0.0208, "mlp_baseline": 0.0244},
}

mmd_mean = {
    "Overall": {"sf2m": 0.0564, "rf": 0.1197, "mlp_baseline": 0.0475},
    "Left‑out 1": {"sf2m": 0.1196, "rf": 0.2254, "mlp_baseline": 0.0801},
    "Left‑out 2": {"sf2m": 0.0752, "rf": 0.1556, "mlp_baseline": 0.0880},
    "Left‑out 3": {"sf2m": 0.0260, "rf": 0.0945, "mlp_baseline": 0.0182},
    "Left‑out 4": {"sf2m": 0.0046, "rf": 0.0032, "mlp_baseline": 0.0037},
}
mmd_std = {
    "Overall": {"sf2m": 0.0044, "rf": 0.0019, "mlp_baseline": 0.0048},
    "Left‑out 1": {"sf2m": 0.0110, "rf": 0.0035, "mlp_baseline": 0.0101},
    "Left‑out 2": {"sf2m": 0.0036, "rf": 0.0019, "mlp_baseline": 0.0062},
    "Left‑out 3": {"sf2m": 0.0020, "rf": 0.0017, "mlp_baseline": 0.0022},
    "Left‑out 4": {"sf2m": 0.0009, "rf": 0.0005, "mlp_baseline": 0.0008},
}

# Convert to DataFrames for easy plotting
df_w_mean = pd.DataFrame(w_mean).T
df_w_std  = pd.DataFrame(w_std).T

df_mmd_mean = pd.DataFrame(mmd_mean).T
df_mmd_std  = pd.DataFrame(mmd_std).T

def plot_grouped_bar(df_mean, df_std, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_mean))
    bar_width = 0.25
    offsets = np.linspace(-bar_width, bar_width, len(df_mean.columns))
    
    for i, model in enumerate(df_mean.columns):
        ax.bar(
            x + offsets[i],
            df_mean[model],
            yerr=df_std[model],
            width=bar_width,
            capsize=4,
            label=model,
            edgecolor='black'
        )
    
    ax.set_xlabel("Left‑out timepoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df_mean.index)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot Wasserstein distance
plot_grouped_bar(
    df_w_mean,
    df_w_std,
    ylabel="Wasserstein distance (ODE)",
    title="Synthetic Data – ODE W‑Dist per Model & Left‑out Timepoint (±1 σ)"
)

# Plot MMD²
plot_grouped_bar(
    df_mmd_mean,
    df_mmd_std,
    ylabel="MMD² (ODE)",
    title="Synthetic Data – ODE MMD² per Model & Left‑out Timepoint (±1 σ)"
)
