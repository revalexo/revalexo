#!/usr/bin/env python
# loso_multi_horizon.py - Leave-One-Subject-Out Cross Validation script

import os
import sys
import argparse
import yaml
import json
import subprocess
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from typing import Dict, List, Any, Tuple

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Leave-One-Subject-Out Cross Validation')
    
    parser.add_argument('--base-config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Root directory for all LOSO outputs (defaults to config log_dir + "_loso")')
    parser.add_argument('--subjects', type=str, nargs='+',
                        help='List of subjects to use (if not specified, extracts from config)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use for training')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--excluded-metrics', type=str, nargs='+', default=[],
                        help='Metrics to exclude from aggregation (e.g., loss)')
    parser.add_argument('--skip-subjects', type=str, nargs='+', default=[],
                        help='Subjects to skip in cross-validation')
    parser.add_argument('--transition-window', type=float, default=0.5,
                        help='Transition window size in seconds (passed to train.py)')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")

def get_subjects_from_config(config: Dict[str, Any]) -> List[str]:
    """Extract all subjects from the config file."""
    all_subjects = set()
    
    # Get subjects from train, val, test splits
    splits = config.get('dataset', {}).get('splits', {})
    for split_name in ['train_subjects', 'val_subjects', 'test_subjects']:
        subjects = splits.get(split_name, [])
        all_subjects.update(subjects)
    
    return sorted(list(all_subjects))

def create_loso_config(config: Dict[str, Any], test_subject: str, output_dir: str) -> Tuple[Dict[str, Any], str]:
    """Create a new config for leave-one-subject-out with the specified test subject."""
    # Deep copy the config to avoid modifying the original
    loso_config = yaml.safe_load(yaml.dump(config))  # Deep copy via YAML serialization
    
    # Get all subjects
    all_subjects = get_subjects_from_config(config)
    
    # Ensure the test subject is in the list
    if test_subject not in all_subjects:
        raise ValueError(f"Test subject {test_subject} not found in the config")
    
    # Create training set (all subjects except the test subject)
    train_subjects = [s for s in all_subjects if s != test_subject]
    
    # Use the same subject for validation and testing
    val_subjects = [test_subject]
    test_subjects = [test_subject]
    
    # Update the splits in the config
    loso_config['dataset']['splits'] = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects
    }
    
    # Update the log directory to include the test subject
    base_log_dir = loso_config.get('logging', {}).get('log_dir', 'outputs/logs')
    subject_log_dir = os.path.join(output_dir, f"subject_{test_subject}")
    loso_config['logging']['log_dir'] = subject_log_dir
    
    # Create the output config file path
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"config_subject_{test_subject}.yaml")
    
    # Write the config to a file
    with open(config_path, 'w') as f:
        yaml.dump(loso_config, f, default_flow_style=False, sort_keys=False)
    
    return loso_config, config_path

def run_training(config_path: str, gpu_id: int, transition_window: float, dry_run: bool = False) -> bool:
    """Run the training script with the specified config."""
    cmd = [
        "python3", "train.py",
        "--config", config_path,
        "--transition-window", str(transition_window),
        "--no-cuda" if gpu_id < 0 else "",
    ]
    
    # Add CUDA_VISIBLE_DEVICES if GPU is specified
    env = os.environ.copy()
    if gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Remove empty elements
    cmd = [x for x in cmd if x]
    
    print(f"Running command: {' '.join(cmd)}")
    
    if dry_run:
        return True
    
    try:
        # Use subprocess.run with no pipe redirection to properly display tqdm bars
        process = subprocess.run(
            cmd,
            env=env,
            check=False,  # Don't raise exception on non-zero return code
        )
        
        # Check return code
        if process.returncode != 0:
            print(f"Training failed with return code {process.returncode}")
            return False
        
        return True
    except Exception as e:
        print(f"Error running training: {e}")
        return False

def find_latest_log_dir(base_dir: str) -> str:
    """Find the most recent log directory in the base directory."""
    log_dirs = glob.glob(os.path.join(base_dir, "*"))
    if not log_dirs:
        raise FileNotFoundError(f"No log directories found in {base_dir}")
    
    # Sort by modification time (newest first)
    latest_dir = max(log_dirs, key=os.path.getmtime)
    return latest_dir

def collect_results(output_dir: str, excluded_metrics: List[str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Collect results from all subjects and create summary DataFrames.
    
    Returns:
        Tuple of (overall_df, horizon_dfs) where:
        - overall_df: DataFrame with all metrics
        - horizon_dfs: Dict mapping horizon values to DataFrames with horizon-specific metrics
    """
    # Find all results.json files
    result_files = glob.glob(os.path.join(output_dir, "subject_*/*/results.json"))
    
    if not result_files:
        print(f"No result files found in {output_dir}")
        return None, {}
    
    all_results = []
    
    for result_file in result_files:
        # Extract subject from path
        subject_dir = os.path.basename(os.path.dirname(os.path.dirname(result_file)))
        subject = subject_dir.replace("subject_", "")
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
                
                # Add subject to results
                results['subject'] = subject
                all_results.append(results)
        except Exception as e:
            print(f"Error loading results from {result_file}: {e}")
    
    if not all_results:
        return None, {}
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Exclude specified metrics
    for metric in excluded_metrics:
        if metric in df.columns:
            df = df.drop(columns=[metric])
    
    # Extract prediction horizons if available
    prediction_horizons = []
    if 'prediction_horizons' in df.columns:
        # Assume all rows have the same prediction horizons
        prediction_horizons = df.iloc[0]['prediction_horizons']
        df = df.drop(columns=['prediction_horizons'])
    
    # Separate horizon-specific metrics
    horizon_dfs = {}
    if prediction_horizons and len(prediction_horizons) > 1:
        # Multi-horizon model
        for h_idx, h_val in enumerate(prediction_horizons):
            horizon_str = f"{h_val:.1f}s"
            horizon_cols = ['subject']
            
            # Find columns for this horizon
            for col in df.columns:
                if f'_horizon_{horizon_str}' in col or f'_horizon_{h_idx}' in col:
                    horizon_cols.append(col)
            
            if len(horizon_cols) > 1:  # More than just 'subject'
                horizon_df = df[horizon_cols].copy()
                
                # Rename columns to remove horizon suffix for cleaner display
                rename_dict = {}
                for col in horizon_cols:
                    if col != 'subject':
                        # Remove horizon suffix
                        new_name = col.replace(f'_horizon_{horizon_str}', '').replace(f'_horizon_{h_idx}', '')
                        rename_dict[col] = new_name
                
                horizon_df = horizon_df.rename(columns=rename_dict)
                
                # Calculate statistics
                horizon_df = add_aggregate_statistics(horizon_df)
                horizon_dfs[horizon_str] = horizon_df
    
    # Calculate aggregate statistics for overall DataFrame
    df = add_aggregate_statistics(df, exclude_horizon_specific=True)
    
    return df, horizon_dfs

def add_aggregate_statistics(df: pd.DataFrame, exclude_horizon_specific: bool = False) -> pd.DataFrame:
    """Add average and standard deviation rows to a DataFrame."""
    # Get numeric columns
    numeric_cols = [col for col in df.columns if col != 'subject' and pd.api.types.is_numeric_dtype(df[col])]
    
    if exclude_horizon_specific:
        # Exclude horizon-specific columns from overall statistics
        numeric_cols = [col for col in numeric_cols if '_horizon_' not in col]
    
    if numeric_cols:
        # Create average row
        avg_row = {'subject': 'AVERAGE'}
        for col in numeric_cols:
            avg_row[col] = df[col].mean()
        
        # Create standard deviation row
        std_row = {'subject': 'STD'}
        for col in numeric_cols:
            std_row[col] = df[col].std()
        
        # Add rows to DataFrame
        df = pd.concat([df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    
    return df

def print_results_summary(overall_df: pd.DataFrame, horizon_dfs: Dict[str, pd.DataFrame]) -> None:
    """Print a formatted summary of LOSO results."""
    print("\n" + "="*80)
    print("LEAVE-ONE-SUBJECT-OUT CROSS VALIDATION SUMMARY")
    print("="*80)
    
    if overall_df is not None:
        # Find the AVERAGE row
        avg_row = overall_df[overall_df['subject'] == 'AVERAGE']
        std_row = overall_df[overall_df['subject'] == 'STD']
        
        if not avg_row.empty and not std_row.empty:
            avg_dict = avg_row.iloc[0].to_dict()
            std_dict = std_row.iloc[0].to_dict()
            
            # Print overall metrics (non-horizon specific)
            print("\nOVERALL METRICS:")
            for metric, value in avg_dict.items():
                if metric != 'subject' and '_horizon_' not in metric and isinstance(value, (int, float)):
                    std_val = std_dict.get(metric, 0)
                    print(f"  {metric}: {value:.4f} ± {std_val:.4f}")
    
    # Print horizon-specific summaries
    if horizon_dfs:
        print("\nHORIZON-SPECIFIC METRICS:")
        for horizon, h_df in sorted(horizon_dfs.items()):
            print(f"\n  Horizon {horizon}:")
            
            avg_row = h_df[h_df['subject'] == 'AVERAGE']
            std_row = h_df[h_df['subject'] == 'STD']
            
            if not avg_row.empty and not std_row.empty:
                avg_dict = avg_row.iloc[0].to_dict()
                std_dict = std_row.iloc[0].to_dict()
                
                # Group metrics by type
                accuracy_metrics = {}
                other_metrics = {}
                
                for metric, value in avg_dict.items():
                    if metric != 'subject' and isinstance(value, (int, float)):
                        if 'accuracy' in metric:
                            accuracy_metrics[metric] = (value, std_dict.get(metric, 0))
                        else:
                            other_metrics[metric] = (value, std_dict.get(metric, 0))
                
                # Print accuracy metrics first
                for metric, (avg_val, std_val) in sorted(accuracy_metrics.items()):
                    print(f"    {metric}: {avg_val:.4f} ± {std_val:.4f}")
                
                # Then other metrics
                for metric, (avg_val, std_val) in sorted(other_metrics.items()):
                    print(f"    {metric}: {avg_val:.4f} ± {std_val:.4f}")
    
    print("\n" + "="*80)

def collect_classification_reports(output_dir: str) -> pd.DataFrame:
    """Collect classification reports from all subjects and aggregate them."""
    # Find all best model metrics files
    csv_files = glob.glob(os.path.join(output_dir, "subject_*/*/best_model_metrics.csv"))
    
    if not csv_files:
        print(f"No classification report files found in {output_dir}")
        return None
    
    all_dfs = []
    
    for csv_file in csv_files:
        # Extract subject from path
        subject_dir = os.path.basename(os.path.dirname(os.path.dirname(csv_file)))
        subject = subject_dir.replace("subject_", "")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Add subject to dataframe
            df['subject'] = subject
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading classification report from {csv_file}: {e}")
    
    if all_dfs:
        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Group by class and calculate aggregate statistics
        grouped = combined_df.groupby('class').agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1-score': ['mean', 'std'],
            'support': 'sum'
        })
        
        # Flatten multi-level column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        # Reset index to make 'class' a regular column again
        grouped = grouped.reset_index()
        
        return grouped
    
    return None

def save_aggregate_results(df: pd.DataFrame, output_dir: str, filename: str):
    """Save aggregate results to a CSV file."""
    if df is not None:
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"Saved aggregate results to {output_path}")

def main():
    """Main entry point for LOSO cross-validation."""
    args = parse_args()
    
    # Load base configuration
    config = load_config(args.base_config)
    
    # Get all subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = get_subjects_from_config(config)
    
    # Remove subjects to skip
    subjects = [s for s in subjects if s not in args.skip_subjects]
    
    if not subjects:
        print("No subjects to process.")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_log_dir = config.get('logging', {}).get('log_dir', 'outputs/logs')
        output_dir = f"{base_log_dir}_loso"

    # Create timestamp for this LOSO run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, timestamp)
    
    print(f"Running Leave-One-Subject-Out Cross Validation with {len(subjects)} subjects")
    print(f"Base config: {args.base_config}")
    print(f"Output directory: {output_dir}")
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Transition window: {args.transition_window}s")
    
    # Extract prediction horizons from config
    prediction_horizons = config.get('dataset', {}).get('prediction_horizons', [0])
    if len(prediction_horizons) > 1:
        print(f"Multi-horizon model with horizons: {[f'{h:.1f}s' for h in prediction_horizons]}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save run metadata
    metadata = {
        'base_config': args.base_config,
        'subjects': subjects,
        'timestamp': timestamp,
        'excluded_metrics': args.excluded_metrics,
        'skipped_subjects': args.skip_subjects,
        'transition_window': args.transition_window,
        'prediction_horizons': prediction_horizons
    }
    
    with open(os.path.join(output_dir, 'loso_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy base config for reference
    with open(os.path.join(output_dir, 'base_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training for each subject
    completed = []
    failed = []
    
    for subject in subjects:
        print(f"\n{'='*80}")
        print(f"Processing subject: {subject}")
        print(f"{'='*80}")
        
        try:
            # Create config for this split
            _, config_path = create_loso_config(config, subject, output_dir)
            
            # Run training
            success = run_training(config_path, args.gpu, args.transition_window, args.dry_run)
            
            if success:
                completed.append(subject)
            else:
                failed.append(subject)
                
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(subject)
    
    # Print summary of completed and failed subjects
    print(f"\n{'='*80}")
    print(f"Cross Validation Complete")
    print(f"{'='*80}")
    print(f"Completed: {len(completed)}/{len(subjects)} subjects")
    
    if failed:
        print(f"Failed: {len(failed)}/{len(subjects)} subjects: {', '.join(failed)}")
    
    # Collect and aggregate results
    if completed and not args.dry_run:
        # Collect numeric results with horizon support
        overall_df, horizon_dfs = collect_results(output_dir, args.excluded_metrics)
        
        if overall_df is not None:
            # Save overall results
            save_aggregate_results(overall_df, output_dir, "loso_subject_results.csv")
            
            # Save horizon-specific results
            for horizon, h_df in horizon_dfs.items():
                save_aggregate_results(h_df, output_dir, f"loso_subject_results_horizon_{horizon}.csv")
            
            # Print formatted summary
            print_results_summary(overall_df, horizon_dfs)
        
        # Collect and aggregate classification reports
        class_report_df = collect_classification_reports(output_dir)
        if class_report_df is not None:
            save_aggregate_results(class_report_df, output_dir, "loso_class_performance.csv")

if __name__ == '__main__':
    main()
