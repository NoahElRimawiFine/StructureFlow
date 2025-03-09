from collections import Counter

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def compare_graphs(true_graph, estimated_graph):
    """Compute performance measures on (binary) adjacency matrix.

    Input:
     - true_graph: (dxd) np.array, the true adjacency matrix
     - estimated graph: (dxd) np.array, the estimated adjacency matrix (weighted or unweighted)
    """
    # Handle new case where we encode information in the negative numbers
    true_graph = np.maximum(0, true_graph)

    # mask the diagonal of our matrices
    n = true_graph.shape[0]
    true_graph = true_graph * (1-np.eye(n))
    estimated_graph = estimated_graph * (1-np.eye(n))

    def structural_hamming_distance(W_true, W_est):
        """Computes the structural hamming distance."""
        pred = np.flatnonzero(W_est != 0)
        cond = np.flatnonzero(W_true)
        cond_reversed = np.flatnonzero(W_true.T)
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        pred_lower = np.flatnonzero(np.tril(W_est + W_est.T))
        cond_lower = np.flatnonzero(np.tril(W_true + W_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
        return shd

    num_edges = len(true_graph[np.where(true_graph != 0.0)])

    tam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in true_graph])
    eam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in estimated_graph])

    tp = len(np.argwhere((tam + eam) == 2))
    fp = len(np.argwhere((tam - eam) < 0))
    tn = len(np.argwhere((tam + eam) == 0))
    fn = num_edges - tp
    x = [tp, fp, tn, fn]

    if x[0] + x[1] == 0:
        precision = 0
    else:
        precision = float(x[0]) / float(x[0] + x[1])
    if tp + fn == 0:
        tpr = 0
    else:
        tpr = float(tp) / float(tp + fn)
    if x[2] + x[1] == 0:
        specificity = 0
    else:
        specificity = float(x[2]) / float(x[2] + x[1])
    if precision + tpr == 0:
        f1 = 0
    else:
        f1 = 2 * precision * tpr / (precision + tpr)
    if fp + tp == 0:
        fdr = 0
    else:
        fdr = float(fp) / (float(fp) + float(tp))

    shd = float(structural_hamming_distance(true_graph, estimated_graph))
    thresh_shd = float(
        structural_hamming_distance(true_graph, (estimated_graph > 0.5).astype(float))
    )

    if np.all(true_graph.flatten()):
        AUC = -1
        AP = -1
    else:
        AUC = roc_auc_score(true_graph.flatten(), estimated_graph.flatten())
        AP = average_precision_score(true_graph.flatten(), estimated_graph.flatten())

    metrics = ["tpr", "fdr", "shd", "tshd", "auc", "ap", "f1", "specificity"]
    values = [tpr, fdr, shd, thresh_shd, AUC, AP, f1, specificity]
    return dict(zip(metrics, values))