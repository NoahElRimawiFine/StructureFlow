import os
import hydra
from omegaconf import OmegaConf
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rootutils

# Set up the project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.plotting import (
    compute_global_jacobian,
    plot_auprs,
    plot_comparison_heatmaps,
)
from src.models.sf2m_module import SF2MLitModule
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_path: Optional path to the config file
        
    Returns:
        The loaded model
    """
    # If config_path is not provided, try to find it in the checkpoint directory
    if config_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        else:
            print(f"Loading configuration from {config_path}")
            cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.load(config_path)
    
    # Initialize the datamodule
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    
    # Create a copy of the model config without the datamodule key
    model_config = {k: v for k, v in cfg.model.items() if k != "datamodule"}
    
    # Load the model from checkpoint
    model = SF2MLitModule.load_from_checkpoint(
        checkpoint_path,
        datamodule=datamodule,
        **model_config
    )
    
    model.eval()
    return model, datamodule

def load_reference_network(csv_path):
    """Load the reference network from CSV.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing the reference network
    """
    # Load the reference network
    ref_network = pd.read_csv(csv_path, index_col=0)
    
    # Fill NaN values with zeros if needed
    ref_network = ref_network.fillna(0)
    
    return ref_network

def infer_and_evaluate(model, datamodule, ref_network):
    """Infer the GRN using the model and evaluate against the reference network.
    
    Args:
        model: Trained SF2M model
        datamodule: The data module
        ref_network: DataFrame with reference network
        
    Returns:
        Dictionary with evaluation results
    """
    # Get the subset of data used for evaluation
    adatas = datamodule.get_subset_adatas("test")
    
    # Get all gene names from the model (full set of 101 genes)
    model_gene_names = datamodule.adatas[0].var_names
    
    # Get reference network gene names (subset of 18 genes)
    ref_rows = ref_network.index
    ref_cols = ref_network.columns
    
    print(f"Reference network shape: {ref_network.shape}")
    print(f"Reference network rows: {len(ref_rows)} genes")
    print(f"Reference network columns: {len(ref_cols)} genes")
    
    # Compute the Jacobian (estimated causal graph) for all 101 genes
    with torch.no_grad():
        A_estim = compute_global_jacobian(model.func_v, adatas, dt=1/model.T, device=torch.device("cpu"))
    
    # Get the directly extracted causal graph from the model for all 101 genes
    W_v = model.func_v.causal_graph(w_threshold=0.0).T
    
    # Create DataFrames for the estimated graphs with all 101 genes
    A_estim_df = pd.DataFrame(A_estim, index=model_gene_names, columns=model_gene_names)
    W_v_df = pd.DataFrame(W_v, index=model_gene_names, columns=model_gene_names)
    
    # Extract the exact 18×101 subset that corresponds to the reference network dimensions
    # Use ref_rows for the rows (18 genes) and ref_cols for the columns (can be different)
    A_estim_subset = A_estim_df.loc[ref_rows, ref_cols]
    W_v_subset = W_v_df.loc[ref_rows, ref_cols]
    
    print(f"Extracted matrices with shape: {A_estim_subset.shape}")
    
    # Convert to numpy arrays for evaluation
    A_estim_np = A_estim_subset.values
    W_v_np = W_v_subset.values
    ref_network_np = ref_network.values
    
    # Evaluate using the plotting functions
    fig_aupr = plot_auprs(W_v_np, A_estim_np, ref_network_np)
    
    # Return results for further processing
    return {
        "A_estim": A_estim_subset,
        "W_v": W_v_subset,
        "ref_network": ref_network,
        "fig_aupr": fig_aupr
    }

def visualize_results(results):
    """Visualize the evaluation results.
    
    Args:
        results: Dictionary with evaluation results
    """
    # Get the data from results
    A_estim = results["A_estim"]
    W_v = results["W_v"]
    ref_network = results["ref_network"]
    
    # Create directory for figures if it doesn't exist
    os.makedirs("evaluation_figs", exist_ok=True)
    
    # Get row and column gene names
    row_gene_names = A_estim.index
    col_gene_names = A_estim.columns
    
    print(f"Visualization matrix shapes: A_estim={A_estim.shape}, W_v={W_v.shape}, ref_network={ref_network.shape}")
    
    # Plot a comparison of the matrices without gene labels
    plot_comparison_heatmaps(
        matrices_and_titles=[
            ("Jacobian (SF2M)", A_estim.values),
            ("Causal Graph (SF2M)", W_v.values),
            ("Reference Network", ref_network.values)
        ],
        row_gene_names=None,  # No gene labels
        col_gene_names=None,  # No gene labels
        main_title="Comparison of Inferred vs. True Gene Regulatory Networks",
        default_vrange=(-0.5, 0.5),
        special_titles_for_range={"Reference Network"},
        special_vrange=(-1, 1),
        figsize_per_plot=(6, 5)  # Larger plot size
    )
    
    # Print some statistics
    print(f"Matrix dimensions: Rows={len(row_gene_names)}, Columns={len(col_gene_names)}")
    print(f"Number of true edges: {np.sum(np.abs(ref_network.values) > 0)}")
    
    # Calculate precision/recall at different thresholds
    for thresh in [0.1, 0.05, 0.01]:
        W_thresh = np.abs(W_v.values) > thresh
        true_edges = np.abs(ref_network.values) > 0
        
        true_positives = np.sum(np.logical_and(W_thresh, true_edges))
        false_positives = np.sum(np.logical_and(W_thresh, ~true_edges))
        false_negatives = np.sum(np.logical_and(~W_thresh, true_edges))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"Threshold {thresh}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 score: {2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0:.4f}")

def main(checkpoint_path, ref_network_path, config_path=None):
    """Main function to run the evaluation.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        ref_network_path: Path to the reference network CSV
        config_path: Path to the config file
    """
    # Load the model and datamodule
    model, datamodule = load_model_from_checkpoint(checkpoint_path, config_path)
    
    # Load the reference network
    ref_network = load_reference_network(ref_network_path)
    
    # Run inference and evaluation
    results = infer_and_evaluate(model, datamodule, ref_network)
    
    # Visualize the results
    visualize_results(results)

if __name__ == "__main__":
    # Replace with your actual paths
    checkpoint_path = "/Users/lucasnelson/sf2m-grn-hydra/logs/train/runs/2025-04-20_21-32-35/checkpoints/last.ckpt"
    ref_network_path = "/Users/lucasnelson/sf2m-grn-hydra/data/Renge/A_ref_thresh_0.csv"
    config_path = "/Users/lucasnelson/sf2m-grn-hydra/logs/train/runs/2025-04-20_21-32-35/.hydra/config.yaml"
    main(checkpoint_path, ref_network_path, config_path)