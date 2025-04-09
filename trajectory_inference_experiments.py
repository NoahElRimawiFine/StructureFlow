#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def run_experiment(dataset_type, model_type, seed, base_results_dir="loo_results"):
    """Run a single experiment with specific configuration."""
    cmd = [
        "python", "-m", "src.leave_one_out",
        "--dataset_type", dataset_type,
        "--model_type", model_type,
        "--seed", str(seed),
        "--results_dir", base_results_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def aggregate_results(base_results_dir, dataset_types, model_types, seeds):
    """Aggregate results across seeds for each model and dataset type."""
    print("\nAggregating results across seeds...")
    
    # Create directories for aggregate results
    agg_dir = os.path.join(base_results_dir, "aggregate_results")
    os.makedirs(agg_dir, exist_ok=True)
    
    # Prepare a summary table for all models and datasets
    all_summary_rows = []
    
    # Process each dataset type
    for dataset_type in dataset_types:
        for model_type in model_types:
            print(f"\nAggregating {model_type} results for {dataset_type} dataset...")
            
            # Collect summary results from all seeds
            all_seed_summaries = []
            all_seed_details = []
            
            for seed in seeds:
                # Load summary results
                summary_path = os.path.join(
                    base_results_dir, 
                    f"{dataset_type}_{model_type}_seed{seed}", 
                    f"loo_summary_{model_type}_seed{seed}.csv"
                )
                
                detailed_path = os.path.join(
                    base_results_dir, 
                    f"{dataset_type}_{model_type}_seed{seed}", 
                    f"loo_detailed_metrics_{model_type}_seed{seed}.csv"
                )
                
                if os.path.exists(summary_path):
                    summary_df = pd.read_csv(summary_path)
                    all_seed_summaries.append(summary_df)
                    
                if os.path.exists(detailed_path):
                    detailed_df = pd.read_csv(detailed_path)
                    all_seed_details.append(detailed_df)
            
            if not all_seed_summaries:
                print(f"  No results found for {model_type} on {dataset_type}")
                continue
            
            # Combine results from all seeds
            combined_summary = pd.concat(all_seed_summaries, ignore_index=True)
            
            # Group by timepoint and calculate mean and std across seeds
            agg_summary = combined_summary.groupby('held_out_time').agg({
                'avg_ode_distance': ['mean', 'std'],
                'avg_sde_distance': ['mean', 'std'],
                'avg_mmd2_ode': ['mean', 'std'],
                'avg_mmd2_sde': ['mean', 'std']
            }).reset_index()
            
            # Flatten column names
            agg_summary.columns = ['_'.join(col).strip() for col in agg_summary.columns.values]
            agg_summary = agg_summary.rename(columns={'held_out_time_': 'held_out_time'})
            
            # Add model and dataset info
            agg_summary['model_type'] = model_type
            agg_summary['dataset_type'] = dataset_type
            
            # Save aggregated summary
            agg_summary_path = os.path.join(agg_dir, f"{dataset_type}_{model_type}_agg_summary.csv")
            agg_summary.to_csv(agg_summary_path, index=False)
            print(f"  Saved aggregated summary to {agg_summary_path}")
            
            # Calculate overall average metrics across all timepoints
            overall_avg = {
                'Model': model_type,
                'Dataset': dataset_type,
                'W-Dist (ODE)': f"{agg_summary['avg_ode_distance_mean'].mean():.4f} ± {agg_summary['avg_ode_distance_std'].mean():.4f}",
                'W-Dist (SDE)': f"{agg_summary['avg_sde_distance_mean'].mean():.4f} ± {agg_summary['avg_sde_distance_std'].mean():.4f}",
                'MMD2 (ODE)': f"{agg_summary['avg_mmd2_ode_mean'].mean():.4f} ± {agg_summary['avg_mmd2_ode_std'].mean():.4f}",
                'MMD2 (SDE)': f"{agg_summary['avg_mmd2_sde_mean'].mean():.4f} ± {agg_summary['avg_mmd2_sde_std'].mean():.4f}",
            }
            all_summary_rows.append(overall_avg)
            
            # If detailed metrics are available, combine them
            if all_seed_details:
                combined_details = pd.concat(all_seed_details, ignore_index=True)
                agg_detailed_path = os.path.join(agg_dir, f"{dataset_type}_{model_type}_detailed.csv")
                combined_details.to_csv(agg_detailed_path, index=False)
                print(f"  Saved combined detailed metrics to {agg_detailed_path}")
    
    # Create overall comparative summary table
    if all_summary_rows:
        overall_summary = pd.DataFrame(all_summary_rows)
        overall_summary_path = os.path.join(agg_dir, "overall_comparison.csv")
        overall_summary.to_csv(overall_summary_path, index=False)
        print(f"\nSaved overall comparison to {overall_summary_path}")
        
        # Print summary table
        print("\n===== Overall Comparison =====")
        print(overall_summary.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Run multiple leave-one-out experiments")
    parser.add_argument("--base_results_dir", type=str, default="loo_results", help="Base directory to save results")
    parser.add_argument("--only_aggregate", action="store_true", help="Only aggregate existing results without running new experiments")
    args = parser.parse_args()
    
    # Configuration
    model_types = ["sf2m", "rf", "mlp_baseline"]
    dataset_types = ["Synthetic", "Curated"]
    seeds = [1, 2, 3, 4, 5]
    
    # Create base results directory
    os.makedirs(args.base_results_dir, exist_ok=True)
    
    # Run experiments
    start_time = datetime.now()
    print(f"Starting experiments at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.only_aggregate:
        total_runs = len(model_types) * len(dataset_types) * len(seeds)
        completed_runs = 0
        
        for dataset_type in dataset_types:
            for model_type in model_types:
                for seed in seeds:
                    print(f"\n[{completed_runs+1}/{total_runs}] Running {model_type} on {dataset_type} dataset with seed {seed}")
                    run_experiment(dataset_type, model_type, seed, args.base_results_dir)
                    completed_runs += 1
                    
                    # Calculate and display progress
                    elapsed = datetime.now() - start_time
                    avg_time_per_run = elapsed / completed_runs
                    remaining_runs = total_runs - completed_runs
                    est_remaining = avg_time_per_run * remaining_runs
                    
                    print(f"Progress: {completed_runs}/{total_runs} runs completed")
                    print(f"Elapsed time: {elapsed}")
                    print(f"Estimated time remaining: {est_remaining}")
    
    # Aggregate results
    aggregate_results(args.base_results_dir, dataset_types, model_types, seeds)
    
    # Calculate total runtime
    total_runtime = datetime.now() - start_time
    print(f"\nTotal runtime: {total_runtime}")
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()