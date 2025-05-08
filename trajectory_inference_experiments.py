#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def run_experiment(dataset_type, model_type, seed, dataset=None, base_results_dir="loo_results"):
    """Run a single experiment with specific configuration."""
    cmd = [
        "python", "-m", "src.leave_one_out",
        "--dataset_type", dataset_type,
        "--model_type", model_type,
        "--seed", str(seed),
        "--results_dir", base_results_dir
    ]
    
    # Add dataset parameter if applicable (for Synthetic dataset_type)
    if dataset_type == "Synthetic" and dataset:
        cmd.extend(["--dataset", dataset])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def aggregate_results(base_results_dir, dataset_types, model_types, seeds, synthetic_datasets=None):
    """Aggregate results across seeds for each model and dataset type."""
    print("\nAggregating results across seeds...")
    
    # Create directories for aggregate results
    agg_dir = os.path.join(base_results_dir, "aggregate_results")
    os.makedirs(agg_dir, exist_ok=True)
    
    # Prepare summary tables
    all_summary_rows = []  # For overall comparison across models
    timepoint_comparisons = {}  # For per-timepoint comparison across models
    
    # Process each dataset type
    for dataset_type in dataset_types:
        # For Synthetic, iterate over all datasets
        datasets_to_process = synthetic_datasets if dataset_type == "Synthetic" else [None]
        
        for dataset in datasets_to_process:
            dataset_suffix = f"_{dataset}" if dataset_type == "Synthetic" else ""
            
            for model_type in model_types:
                print(f"\nAggregating {model_type} results for {dataset_type}{dataset_suffix} dataset...")
                
                # Collect summary results from all seeds
                all_seed_summaries = []
                all_seed_details = []
                
                for seed in seeds:
                    # Load summary results
                    summary_path = os.path.join(
                        base_results_dir, 
                        f"{dataset_type}_{model_type}{dataset_suffix}_seed{seed}", 
                        f"loo_summary_{model_type}_seed{seed}.csv"
                    )
                    
                    detailed_path = os.path.join(
                        base_results_dir, 
                        f"{dataset_type}_{model_type}{dataset_suffix}_seed{seed}", 
                        f"loo_detailed_metrics_{model_type}_seed{seed}.csv"
                    )
                    
                    if os.path.exists(summary_path):
                        summary_df = pd.read_csv(summary_path)
                        all_seed_summaries.append(summary_df)
                        
                    if os.path.exists(detailed_path):
                        detailed_df = pd.read_csv(detailed_path)
                        all_seed_details.append(detailed_df)
                
                if not all_seed_summaries:
                    print(f"  No results found for {model_type} on {dataset_type}{dataset_suffix}")
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
                if dataset_type == "Synthetic":
                    agg_summary['dataset'] = dataset
                
                # Save aggregated summary
                agg_summary_path = os.path.join(agg_dir, f"{dataset_type}{dataset_suffix}_{model_type}_agg_summary.csv")
                agg_summary.to_csv(agg_summary_path, index=False)
                print(f"  Saved aggregated summary to {agg_summary_path}")
                
                # Calculate overall average metrics across all timepoints
                overall_avg = {
                    'Model': model_type,
                    'Dataset': f"{dataset_type}{dataset_suffix}",
                    'W-Dist (ODE)': f"{agg_summary['avg_ode_distance_mean'].mean():.4f} ± {agg_summary['avg_ode_distance_std'].mean():.4f}",
                    'W-Dist (SDE)': f"{agg_summary['avg_sde_distance_mean'].mean():.4f} ± {agg_summary['avg_sde_distance_std'].mean():.4f}",
                    'MMD2 (ODE)': f"{agg_summary['avg_mmd2_ode_mean'].mean():.4f} ± {agg_summary['avg_mmd2_ode_std'].mean():.4f}",
                    'MMD2 (SDE)': f"{agg_summary['avg_mmd2_sde_mean'].mean():.4f} ± {agg_summary['avg_mmd2_sde_std'].mean():.4f}",
                }
                all_summary_rows.append(overall_avg)
                
                # Store per-timepoint data for comparison across models
                for _, row in agg_summary.iterrows():
                    timepoint = int(row['held_out_time'])
                    dataset_key = f"{dataset_type}{dataset_suffix}"
                    if (dataset_key, timepoint) not in timepoint_comparisons:
                        timepoint_comparisons[(dataset_key, timepoint)] = []
                    
                    # Add this model's results for this timepoint
                    timepoint_comparisons[(dataset_key, timepoint)].append({
                        'Model': model_type,
                        'W-Dist (ODE)': f"{row['avg_ode_distance_mean']:.4f} ± {row['avg_ode_distance_std']:.4f}",
                        'W-Dist (SDE)': f"{row['avg_sde_distance_mean']:.4f} ± {row['avg_sde_distance_std']:.4f}",
                        'MMD2 (ODE)': f"{row['avg_mmd2_ode_mean']:.4f} ± {row['avg_mmd2_ode_std']:.4f}",
                        'MMD2 (SDE)': f"{row['avg_mmd2_sde_mean']:.4f} ± {row['avg_mmd2_sde_std']:.4f}",
                    })
                
                # If detailed metrics are available, combine them
                if all_seed_details:
                    combined_details = pd.concat(all_seed_details, ignore_index=True)
                    agg_detailed_path = os.path.join(agg_dir, f"{dataset_type}{dataset_suffix}_{model_type}_detailed.csv")
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
    
    # Create per-timepoint comparison tables
    if timepoint_comparisons:
        print("\n===== Per-Timepoint Comparisons =====")
        
        # Create directory for timepoint comparisons if needed
        timepoint_dir = os.path.join(agg_dir, "timepoint_comparisons")
        os.makedirs(timepoint_dir, exist_ok=True)
        
        # For each dataset type and timepoint
        for (dataset_type, timepoint), models_data in timepoint_comparisons.items():
            if models_data:
                # Create a DataFrame for this timepoint
                timepoint_df = pd.DataFrame(models_data)
                
                # Save to CSV
                timepoint_path = os.path.join(timepoint_dir, f"{dataset_type}_timepoint_{timepoint}.csv")
                timepoint_df.to_csv(timepoint_path, index=False)
                
                # Print the comparison
                print(f"\nDataset: {dataset_type}, Timepoint: {timepoint}")
                print(timepoint_df.to_string(index=False))
        
        print(f"\nPer-timepoint comparisons saved to {timepoint_dir}")
    
    # Create a combined per-timepoint table for each dataset
    for dataset_type in dataset_types:
        # Get all timepoints for this dataset
        timepoints = sorted(t for ds, t in timepoint_comparisons.keys() if ds == dataset_type)
        
        if timepoints:
            # For each model, collect data across timepoints
            model_timepoint_data = {model: [] for model in model_types if any(m['Model'] == model for tp in timepoints for m in timepoint_comparisons.get((dataset_type, tp), []))}
            
            for timepoint in timepoints:
                models_at_tp = timepoint_comparisons.get((dataset_type, timepoint), [])
                
                # Add each model's data for this timepoint
                for model_data in models_at_tp:
                    model = model_data['Model']
                    if model in model_timepoint_data:
                        model_timepoint_data[model].append({
                            'Timepoint': timepoint,
                            'W-Dist (ODE)': model_data['W-Dist (ODE)'],
                            'W-Dist (SDE)': model_data['W-Dist (SDE)'],
                            'MMD2 (ODE)': model_data['MMD2 (ODE)'],
                            'MMD2 (SDE)': model_data['MMD2 (SDE)'],
                        })
            
            # Create a combined table for each model
            for model, timepoint_rows in model_timepoint_data.items():
                if timepoint_rows:
                    # Sort by timepoint
                    timepoint_rows.sort(key=lambda x: x['Timepoint'])
                    
                    # Create DataFrame
                    model_tp_df = pd.DataFrame(timepoint_rows)
                    
                    # Save to CSV
                    model_tp_path = os.path.join(agg_dir, f"{dataset_type}_{model}_by_timepoint.csv")
                    model_tp_df.to_csv(model_tp_path, index=False)
                    print(f"Saved {model} timepoint analysis for {dataset_type} to {model_tp_path}")

def main():
    parser = argparse.ArgumentParser(description="Run multiple leave-one-out experiments")
    parser.add_argument("--base_results_dir", type=str, default="loo_results", help="Base directory to save results")
    parser.add_argument("--only_aggregate", action="store_true", help="Only aggregate existing results without running new experiments")
    args = parser.parse_args()
    
    # Configuration
    # model_types = ["sf2m", "rf", "mlp_baseline"]
    model_types = ["mlp_baseline"]
    dataset_types = ["Synthetic", "Curated"]
    synthetic_datasets = ["dyn-TF", "dyn-BF", "dyn-CY", "dyn-LL", "dyn-SW"]
    seeds = [1, 2, 3, 4, 5]
    
    # Create base results directory
    os.makedirs(args.base_results_dir, exist_ok=True)
    
    # Run experiments
    start_time = datetime.now()
    print(f"Starting experiments at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.only_aggregate:
        # Calculate total experiments
        total_runs = 0
        for dataset_type in dataset_types:
            if dataset_type == "Synthetic":
                total_runs += len(model_types) * len(synthetic_datasets) * len(seeds)
            else:
                total_runs += len(model_types) * len(seeds)
                
        completed_runs = 0
        
        # Run experiments for each dataset type
        for dataset_type in dataset_types:
            if dataset_type == "Synthetic":
                # For Synthetic, run each dataset
                for dataset in synthetic_datasets:
                    for model_type in model_types:
                        for seed in seeds:
                            print(f"\n[{completed_runs+1}/{total_runs}] Running {model_type} on {dataset_type}_{dataset} with seed {seed}")
                            run_experiment(dataset_type, model_type, seed, dataset, args.base_results_dir)
                            completed_runs += 1
                            
                            # Calculate and display progress
                            elapsed = datetime.now() - start_time
                            avg_time_per_run = elapsed / completed_runs
                            remaining_runs = total_runs - completed_runs
                            est_remaining = avg_time_per_run * remaining_runs
                            
                            print(f"Progress: {completed_runs}/{total_runs} runs completed")
                            print(f"Elapsed time: {elapsed}")
                            print(f"Estimated time remaining: {est_remaining}")
            else:
                # For Curated, just run once without dataset parameter
                for model_type in model_types:
                    for seed in seeds:
                        print(f"\n[{completed_runs+1}/{total_runs}] Running {model_type} on {dataset_type} with seed {seed}")
                        run_experiment(dataset_type, model_type, seed, None, args.base_results_dir)
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
    aggregate_results(args.base_results_dir, dataset_types, model_types, seeds, synthetic_datasets)
    
    # Calculate total runtime
    total_runtime = datetime.now() - start_time
    print(f"\nTotal runtime: {total_runtime}")
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()