import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scprep
import seaborn as sb
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_scatter(obs, model, title="fig", wandb_logger=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    batch_size, ts, dim = obs.shape
    obs = obs.reshape(-1, dim).detach().cpu().numpy()
    ts = np.tile(np.arange(ts), batch_size)
    scprep.plot.scatter2d(obs, c=ts, ax=ax)
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    if wandb_logger:
        wandb_logger.log_image(key=title, images=[f"figs/{title}.png"])
    plt.close()


def plot_scatter_and_flow(obs, model, title="stream", wandb_logger=None):
    batch_size, ts, dim = obs.shape
    device = obs.device
    obs = obs.reshape(-1, dim).detach().cpu().numpy()
    diff = obs.max() - obs.min()
    wmin = obs.min() - diff * 0.1
    wmax = obs.max() + diff * 0.1
    points = 50j
    points_real = 50
    Y, X, T = np.mgrid[wmin:wmax:points, wmin:wmax:points, 0 : ts - 1 : 7j]
    gridpoints = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), requires_grad=True, device=device
    ).type(torch.float32)
    times = torch.tensor(T.flatten(), requires_grad=True, device=device).type(torch.float32)[
        :, None
    ]
    out = model(times, gridpoints)
    out = out.reshape([points_real, points_real, 7, dim])
    out = out.cpu().detach().numpy()
    # Stream over time
    fig, axes = plt.subplots(1, 7, figsize=(20, 4), sharey=True)
    axes = axes.flatten()
    tts = np.tile(np.arange(ts), batch_size)
    for i in range(7):
        scprep.plot.scatter2d(obs, c=tts, ax=axes[i])
        axes[i].streamplot(
            X[:, :, 0],
            Y[:, :, 0],
            out[:, :, i, 0],
            out[:, :, i, 1],
            color=np.sum(out[:, :, i] ** 2, axis=-1),
        )
        axes[i].set_title(f"t = {np.linspace(0, ts-1, 7)[i]:0.2f}")
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key="flow", images=[f"figs/{title}.png"])


def store_trajectories(obs: Union[torch.Tensor, list], model, title="trajs", start_time=0):
    n = 2000
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)
    from torchdyn.core import NeuralODE

    with torch.no_grad():
        node = NeuralODE(model)
        # For consistency with DSB
        traj = node.trajectory(start, t_span=torch.linspace(0, ts - 1, 20 * (ts - 1)))
        traj = traj.cpu().detach().numpy()
        os.makedirs("figs", exist_ok=True)
        np.save(f"figs/{title}.npy", traj)


def plot_trajectory(
    obs: Union[torch.Tensor, list],
    traj: torch.Tensor,
    title="traj",
    key="traj",
    start_time=0,
    n=200,
    wandb_logger=None,
):
    plt.figure(figsize=(6, 6))
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)
        scprep.plot.scatter2d(obs, c=tts)
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.3, alpha=0.2, c="black", label="Flow")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=6, alpha=1, c="purple", marker="x")
    for i in range(20):
        plt.plot(traj[:, i, 0], traj[:, i, 1], c="red", alpha=0.5)
    # plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key=key, images=[f"figs/{title}.png"])


def plot_paths(
    obs: Union[torch.Tensor, list],
    model,
    title="paths",
    start_time=0,
    n=200,
    wandb_logger=None,
):
    plt.figure(figsize=(6, 6))
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)
        scprep.plot.scatter2d(obs, c=tts)
    from torchdyn.core import NeuralODE

    with torch.no_grad():
        node = NeuralODE(model)
        traj = node.trajectory(start, t_span=torch.linspace(0, ts - 1, max(20 * ts, 100)))
        traj = traj.cpu().detach().numpy()
    # plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.3, alpha=0.2, c="black", label="Flow")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=6, alpha=1, c="purple", marker="x")
    # plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key="paths", images=[f"figs/{title}.png"])


def plot_samples(trajs, title="samples", wandb_logger=None):
    import PIL
    from torchvision.utils import save_image

    images = trajs[:100]
    os.makedirs("figs", exist_ok=True)
    save_image(images, fp=f"figs/{title}.jpg", nrow=10, normalize=True, padding=0)
    if wandb_logger:
        try:
            wandb_logger.log_image(key="paths", images=[f"figs/{title}.jpg"])
        except PIL.UnidentifiedImageError:
            print(f"ERROR logging {title}")


def plot_comparison_heatmaps(
    matrices_and_titles,
    row_gene_names=None,
    col_gene_names=None,
    gene_names=None,
    main_title="Heatmaps Representing Gene-Gene Interactions",
    default_vrange=(-2.5, 2.5),
    special_titles_for_range=None,
    special_vrange=(-1.0, 1.0),
    cmap="RdBu_r",
    figsize_per_plot=(3.5, 3.5),
    invert_yaxis=True,
    mask_diagonal=True,
):
    """Plots a row of heatmaps, one for each (title, matrix) pair in `matrices_and_titles`.

    Args:
        matrices_and_titles (list of (str, 2D array-like)):
            A list of tuples: (title, matrix). The matrix can be a numpy array or something convertible to DataFrame.

        row_gene_names (list of str, optional):
            If provided, used for row labels in the DataFrame.

        col_gene_names (list of str, optional):
            If provided, used for column labels in the DataFrame.

        gene_names (list of str, optional):
            If provided and row_gene_names/col_gene_names are not provided, used for both row/col labels in the DataFrame.
            Kept for backward compatibility.

        main_title (str):
            A main title displayed above all subplots.

        default_vrange (tuple):
            The default (vmin, vmax) for the heatmap's color scale.

        special_titles_for_range (set or list, optional):
            Some titles might need a different color scale.
            For example, if you want "SF2M" or "True" to have a smaller range.
            If so, you can supply titles here, and they will use `special_vrange`.

        special_vrange (tuple):
            (vmin, vmax) for special titles. By default, e.g. (-1, 1).

        cmap (str):
            Name of the matplotlib/seaborn colormap.

        figsize_per_plot (tuple):
            Each subplot's (width, height) in inches. The overall figure will be len(matrices_and_titles) * width.

        invert_yaxis (bool):
            If True, calls `plt.gca().invert_yaxis()` for each subplot.
            
        mask_diagonal (bool):
            If True, masks the diagonal of each matrix before plotting.

    Example:
        plot_comparison_heatmaps(
            matrices_and_titles=[
                ("Model A KO", matrixA_ko),
                ("Model A", matrixA_wt),
                ("Model B KO", matrixB_ko),
                ("Model B", matrixB_wt),
                ("True", true_matrix),
            ],
            row_gene_names=row_genes,
            col_gene_names=col_genes,
            main_title="Comparison of Gene Interactions",
            default_vrange=(-2.5, 2.5),
            special_titles_for_range={"Model B KO", "Model B", "True"},
            special_vrange=(-1.0, 1.0),
        )
    """
    if special_titles_for_range is None:
        special_titles_for_range = set()

    # For backward compatibility
    if row_gene_names is None and col_gene_names is None and gene_names is not None:
        row_gene_names = gene_names
        col_gene_names = gene_names

    n = len(matrices_and_titles)
    figwidth = figsize_per_plot[0] * n
    figheight = figsize_per_plot[1]

    plt.figure(figsize=(figwidth, figheight))

    for i, (title, matrix) in enumerate(matrices_and_titles, start=1):
        plt.subplot(1, n, i)
        
        # Apply maskdiag if requested
        matrix_to_plot = maskdiag(matrix) if mask_diagonal else matrix

        # Build DataFrame for seaborn
        if row_gene_names is not None and col_gene_names is not None:
            df = pd.DataFrame(matrix_to_plot, index=row_gene_names, columns=col_gene_names)
        else:
            df = pd.DataFrame(matrix_to_plot)

        # Decide the color range
        if title in special_titles_for_range:
            vmin, vmax = special_vrange
        else:
            vmin, vmax = default_vrange

        # Plot the heatmap
        sb.heatmap(df, vmin=vmin, vmax=vmax, cmap=cmap)
        if invert_yaxis:
            plt.gca().invert_yaxis()

        plt.title(title)

    # Add a main title for the entire figure
    plt.suptitle(main_title, y=1.05, fontsize=14)

    plt.tight_layout()
    plt.show()


def maskdiag(A):
    """Zero out the diagonal entries from matrix A, correctly handling non-square matrices.
    Args:
        A: numpy array of shape (rows, cols)
    Returns:
        numpy array with diagonal elements zeroed out
    """
    # Create a copy to avoid modifying the original
    A_masked = A.copy()
    
    # Zero out the diagonal elements (only where they exist)
    min_dim = min(A.shape[0], A.shape[1])
    for i in range(min_dim):
        A_masked[i, i] = 0
        
    return A_masked

def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

def maskdiag_np(A):
    A = to_numpy(A)
    n = A.shape[0]
    return A * (1 - np.eye(n, dtype=A.dtype))


def plot_aupr_curve(A_true, W_v, prefix="val"):
    """Plots the precision-recall curve based on the true and estimated adjacency matrices.

    Args:
        A_true (np.ndarray): The true adjacency matrix.
        W_v (np.ndarray): The estimated adjacency matrix from your model.
        prefix (str): Prefix for the title and logged metric.

    Returns:
        fig: The matplotlib figure object.
    """
    # Compute binary ground truth from A_true.
    if isinstance(A_true, pd.DataFrame):
        A_true = A_true.values
    y_true = np.abs(np.sign(maskdiag(A_true)).astype(int).flatten())
    # Estimated predictions: absolute value of the estimated adjacency matrix.
    y_pred = np.abs(maskdiag(W_v).flatten())

    # Compute precision, recall and thresholds
    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)

    # Compute AUPR ratio if needed (e.g., average precision divided by fraction of nonzero edges in A_true)
    nonzero_ratio = np.mean(np.abs(A_true) > 0)
    aupr_ratio = avg_prec / nonzero_ratio if nonzero_ratio > 0 else float("nan")

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"MLPODEF-based (AP = {avg_prec:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (MLPODEF)\nAUPR ratio = {aupr_ratio:.2f}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def get_weights_hist(model):
    """Extracts all weight parameters (from parameters whose names contain 'weight' and that
    require gradients) and concatenates them into a single 1D numpy array."""
    weights = []
    for name, param in model.named_parameters():
        # Check that the parameter is a weight and is trainable
        if "weight" in name and param.requires_grad:
            # Detach, move to CPU, flatten, and add to our list
            weights.append(param.detach().cpu().numpy().flatten())
    if weights:
        return np.concatenate(weights)
    else:
        return np.array([])


def plot_histograms(model_before, model_after, bins=50):
    """Plots side-by-side histograms of the weight values for two models.

    Parameters:
        model_before: The original NN model (before integration into Lightning).
        model_after: The NN model after being incorporated into the Lightning module.
        bins: Number of bins for the histogram.
    """
    weights_before = get_weights_hist(model_before)
    weights_after = get_weights_hist(model_after)

    # Create subplots for side-by-side histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(weights_before, bins=bins, color="blue", alpha=0.7)
    axes[0].set_title("Weights Before Lightning Module")
    axes[0].set_xlabel("Weight Value")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(weights_after, bins=bins, color="green", alpha=0.7)
    axes[1].set_title("Weights After Lightning Module")
    axes[1].set_xlabel("Weight Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def compute_global_jacobian(v, adatas, dt, device=torch.device("cpu")):
    """Compute a single adjacency from a big set of states across all datasets.

    Returns a [d, d] numpy array representing an average Jacobian.
    """
    # Move model to the specified device if it's not already there
    v = v.to(device)
    
    all_x_list = []
    for ds_idx, adata in enumerate(adatas):
        x0 = adata.X[adata.obs["t"] == 0]
        all_x_list.append(x0)
    if len(all_x_list) == 0:
        return None

    X_all = np.concatenate(all_x_list, axis=0)
    if X_all.shape[0] == 0:
        return None

    # Move data to device once
    X_all_torch = torch.from_numpy(X_all).float().to(device)
    t_val = torch.tensor(0.0, device=device)

    def get_flow(t, x):
        # No need to move to device again as input tensors are already on the right device
        x_input = x.unsqueeze(0).unsqueeze(0)
        t_input = t.unsqueeze(0).unsqueeze(0)
        return v(t_input, x_input).squeeze(0).squeeze(0)

    # Move the function to device context to ensure all intermediates stay on device
    Ju = torch.func.jacrev(get_flow, argnums=1)

    Js = []
    batch_size = 256
    
    # Using a context manager to ensure all operations happen on the specified device
    with torch.device(device):
        for start in range(0, X_all_torch.shape[0], batch_size):
            end = start + batch_size
            batch_x = X_all_torch[start:end]
            
            # Using a lambda that ensures inputs stay on device
            J_local = torch.vmap(lambda x: Ju(t_val, x))(batch_x)
            J_avg = J_local.mean(dim=0)
            Js.append(J_avg)

    if len(Js) == 0:
        return None
        
    # Stack and compute mean, ensuring it stays on device
    J_final = torch.stack(Js, dim=0).mean(dim=0)
    A_est = J_final

    # Only move to CPU at the very end
    return A_est.detach().cpu().numpy().T


def plot_auprs(causal_graph, jacobian, true_graph, logger=None, global_step=0, mask_diagonal=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    if mask_diagonal:
        masked_true_graph = maskdiag(true_graph)
        masked_jacobian = maskdiag(jacobian)
        masked_causal_graph = maskdiag(causal_graph)
    else:
        masked_true_graph = true_graph
        masked_jacobian = jacobian
        masked_causal_graph = causal_graph

    y_true = np.abs(np.sign(masked_true_graph).astype(int).flatten())

    # --- Jacobian-based ---
    y_pred = np.abs(masked_jacobian.flatten())
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)

    axs[0].plot(rec, prec, label=f"Jacobian-based (AP = {avg_prec:.2f})")
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Precision")
    axs[0].set_title(
        f"Precision-Recall Curve (Jacobian)\nAUPR ratio = {avg_prec / np.mean(np.abs(masked_true_graph) > 0):.2f}"
    )
    axs[0].legend()
    axs[0].grid(True)

    # --- MLPODEF-based ---
    y_pred_mlp = np.abs(masked_causal_graph.flatten())
    prec, rec, _ = precision_recall_curve(y_true, y_pred_mlp)
    avg_prec_mlp = average_precision_score(y_true, y_pred_mlp)

    axs[1].plot(rec, prec, label=f"MLPODEF-based (AP = {avg_prec_mlp:.2f})")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_title(
        f"Precision-Recall Curve (MLPODEF)\nAUPR ratio = {avg_prec_mlp / np.mean(np.abs(masked_true_graph) > 0):.2f}"
    )
    axs[1].legend()
    axs[1].grid(True)

    fig.tight_layout()

    # --- Logging or showing ---
    if logger is not None:
        logger.experiment.add_figure("Causal_Graph", fig, global_step=global_step)
        plt.close(fig)
    else:
        plt.show()
    
    print("AP: ", avg_prec_mlp)
    print("AUPR ratio: ", avg_prec_mlp / np.mean(np.abs(masked_true_graph) > 0))


def log_causal_graph_matrices(A_estim=None, W_v=None, A_true=None, logger=None, global_step=0, mask_diagonal=True, prefix="grn/"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Apply masking based on parameter
    if A_estim is None:
        A_estim_plot = np.zeros_like(A_true)
    else:
        A_estim_plot = maskdiag(A_estim) if mask_diagonal else A_estim

    W_v_plot = maskdiag_np(W_v.T) if mask_diagonal else W_v.T
    A_true_plot = maskdiag_np(A_true) if mask_diagonal else A_true

    # --- A_estim ---
    im1 = axs[0].imshow(A_estim_plot, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    axs[0].invert_yaxis()
    axs[0].set_title("A_estim (from Jacobian)")
    fig.colorbar(im1, ax=axs[0])

    # --- W_v ---
    im2 = axs[1].imshow(W_v_plot, cmap="Reds")
    axs[1].invert_yaxis()
    axs[1].set_title("Causal Graph (from MLPODEF)")
    fig.colorbar(im2, ax=axs[1])

    # --- A_true ---
    im3 = axs[2].imshow(A_true_plot, vmin=-1, vmax=1, cmap="RdBu_r")
    axs[2].invert_yaxis()
    axs[2].set_title("A_true")
    fig.colorbar(im3, ax=axs[2])

    fig.tight_layout()

    # --- Logging ---
    logged = False
    if logger is not None and hasattr(logger, "experiment"):
        exp = logger.experiment
        # W&B via Lightning's WandbLogger
        if hasattr(exp, "log"):  # wandb.run
            try:
                import wandb
                exp.log({f"{prefix}Causal_Graph_Matrices": wandb.Image(fig),
                         "trainer/step": global_step}, step=global_step)
                logged = True
            except Exception:
                pass
        # TensorBoard
        if not logged and hasattr(exp, "add_figure"):
            exp.add_figure(f"{prefix}Causal_Graph_Matrices", fig, global_step=global_step)
            logged = True

    # Direct W&B (no Lightning logger passed)
    if not logged:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"{prefix}Causal_Graph_Matrices": wandb.Image(fig),
                           "trainer/step": global_step}, step=global_step)
                logged = True
        except Exception:
            pass

    if logged:
        plt.close(fig)
    else:
        plt.show()
