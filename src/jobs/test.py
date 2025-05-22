import matplotlib as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

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

causal_graph = pd.read_csv("./results/dyn-TF/full/sf2m_full_42.csv")

def log_causal_graph_matrices(W_v, logger=None, global_step=0):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # --- W_v ---
    im2 = axs[1].imshow(maskdiag(W_v), cmap="Reds")
    axs[1].invert_yaxis()
    axs[1].set_title("Causal Graph (from MLPODEF)")
    fig.colorbar(im2, ax=axs[1])

    if logger is not None:
        logger.experiment.add_figure("Causal_Graph_Matrices", fig, global_step=global_step)
        plt.close(fig)
    else:
        plt.show()

log_causal_graph_matrices(causal_graph)