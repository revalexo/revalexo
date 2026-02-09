#!/usr/bin/env python3
"""
analyze_loso_results_v3.py - Enhanced LOSO cross-validation results analyzer with overall averages
Extracts metrics directly from best_model_metrics*.csv files for each horizon

Usage:
python3 analyze_loso_results.py \
    --base-dir outputs/loso/aidwear \
    --output-dir analysis_results
"""

import os
import pandas as pd
import numpy as np
import argparse
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import re

class EnhancedLOSOAnalyzer:
    """Enhanced analyzer for LOSO cross-validation results."""
    
    def __init__(self, base_dir: str, output_dir: str, create_subject_reports: bool = True):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.create_subject_reports = create_subject_reports
        
        # Storage for metrics
        self.metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.per_class_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.loss_data = defaultdict(list)
        self.lr_data = defaultdict(list)  # NEW: Storage for learning rate data
        
        # Storage for individual subject metrics
        self.subject_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        
        # Model names and horizons
        self.model_names = []
        self.horizons = []
        self.classes = set()
        self.subjects = set()
        
    def scan_and_extract_metrics(self):
        """Scan directory structure and extract metrics from CSV files."""
        print("Scanning for model results...")
        
        # Find all model directories (exclude output directory if it's inside base_dir)
        model_dirs = [d for d in self.base_dir.iterdir()
                      if d.is_dir() and d.resolve() != self.output_dir.resolve()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"\nProcessing model: {model_name}")
            
            # Find latest timestamp directory
            timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if not timestamp_dirs:
                print(f"  No timestamp directories found for {model_name}")
                continue
                
            latest_timestamp = max(timestamp_dirs, key=lambda x: x.name)
            print(f"  Using timestamp: {latest_timestamp.name}")
            
            # Find all subject directories
            subject_dirs = [d for d in latest_timestamp.iterdir() if d.is_dir() and d.name.startswith('subject_')]
            print(f"  Found {len(subject_dirs)} subjects")

            # Skip models with 0 subjects
            if len(subject_dirs) == 0:
                print(f"  Skipping {model_name} (no subjects)")
                continue

            self.model_names.append(model_name)
            
            for subject_dir in subject_dirs:
                subject_id = subject_dir.name.replace('subject_', '')
                self.subjects.add(subject_id)
                
                # Find the run directory (there should be one per subject)
                run_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
                if not run_dirs:
                    continue
                    
                run_dir = run_dirs[0]  # Take the first (and usually only) run
                
                # Extract metrics from each horizon
                horizon_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('horizon_')])
                
                for horizon_dir in horizon_dirs:
                    horizon = horizon_dir.name.replace('horizon_', '')
                    if horizon not in self.horizons:
                        self.horizons.append(horizon)
                    
                    # Process each condition (overall, steady, transition)
                    conditions = {
                        'overall': 'best_model_metrics.csv',
                        'steady': 'best_model_metrics_steady.csv',
                        'transition': 'best_model_metrics_transition.csv'
                    }
                    
                    for condition, filename in conditions.items():
                        csv_path = horizon_dir / filename
                        if csv_path.exists():
                            metrics = self.extract_metrics_from_csv(csv_path)
                            if metrics:
                                # Store overall metrics
                                self.metrics_data[model_name][horizon][condition].append(metrics)

                                # Store subject-specific metrics
                                self.subject_metrics[subject_id][model_name][horizon][condition] = metrics

                                # Store per-class metrics (only from 'overall' condition
                                # to avoid mixing data from steady/transition)
                                if condition == 'overall' and 'per_class_f1' in metrics:
                                    for class_name, f1_score in metrics['per_class_f1'].items():
                                        self.per_class_metrics[model_name][horizon][class_name].append(f1_score)
                                        self.classes.add(class_name)
                
                # Extract loss curves and learning rates from history.log
                history_log = run_dir / 'history.log'
                if history_log.exists():
                    self.extract_loss_curves(history_log, model_name, subject_id)
        
        # Sort horizons for consistent ordering
        self.horizons = sorted(self.horizons, key=lambda x: float(x.replace('s', '')))
        print(f"\nFound horizons: {self.horizons}")
        print(f"Found models: {self.model_names}")
        
    def extract_metrics_from_csv(self, csv_path: Path) -> Dict:
        """Extract metrics from a single CSV file."""
        try:
            df = pd.read_csv(csv_path)
            
            metrics = {}
            
            # Get accuracy (from the 'accuracy' row)
            accuracy_row = df[df['class'] == 'accuracy']
            if not accuracy_row.empty:
                metrics['accuracy'] = float(accuracy_row['f1-score'].iloc[0])
            
            # Get macro avg F1
            macro_row = df[df['class'] == 'macro avg']
            if not macro_row.empty:
                metrics['macro_f1'] = float(macro_row['f1-score'].iloc[0])
                metrics['macro_precision'] = float(macro_row['precision'].iloc[0])
                metrics['macro_recall'] = float(macro_row['recall'].iloc[0])
            
            # Get per-class F1 scores
            per_class_f1 = {}
            class_rows = df[~df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])]
            for _, row in class_rows.iterrows():
                class_name = row['class']
                per_class_f1[class_name] = float(row['f1-score'])
            
            if per_class_f1:
                metrics['per_class_f1'] = per_class_f1
            
            return metrics
            
        except Exception as e:
            print(f"    Error reading {csv_path}: {e}")
            return {}
    
    def extract_loss_curves(self, history_log: Path, model_name: str, subject_id: str):
        """Extract training/validation loss and learning rates from history.log file."""
        try:
            with open(history_log, 'r') as f:
                lines = f.readlines()
            
            train_losses = []
            val_losses = []
            learning_rates = []
            
            for line in lines:
                # Look for training loss
                if 'Train: loss=' in line:
                    match = re.search(r'loss=(\d+\.\d+)', line)
                    if match:
                        train_losses.append(float(match.group(1)))
                
                # Look for validation loss
                if 'Val Horizon 0.0s - Overall: loss=' in line:
                    match = re.search(r'loss=(\d+\.\d+)', line)
                    if match:
                        val_losses.append(float(match.group(1)))
                
                # Look for learning rate - multiple possible formats
                # Format 1: lr=1e-4 or lr=0.0001
                if 'lr=' in line:
                    match = re.search(r'lr=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                    if match:
                        learning_rates.append(float(match.group(1)))
                # Format 2: Learning Rate: 0.0001
                elif 'Learning Rate:' in line or 'learning rate:' in line:
                    match = re.search(r'[Ll]earning [Rr]ate:\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                    if match:
                        learning_rates.append(float(match.group(1)))
                # Format 3: LR: 0.0001
                elif 'LR:' in line:
                    match = re.search(r'LR:\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                    if match:
                        learning_rates.append(float(match.group(1)))
            
            # Store loss data
            if train_losses and val_losses:
                self.loss_data[model_name].append({
                    'subject': subject_id,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                })
            
            # Store learning rate data
            if learning_rates:
                self.lr_data[model_name].append({
                    'subject': subject_id,
                    'learning_rates': learning_rates
                })
                
        except Exception as e:
            print(f"    Error reading history.log for {model_name}, subject {subject_id}: {e}")
    
    def calculate_statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate mean and std for all metrics."""
        # Prepare data for CSV reports
        accuracy_data = defaultdict(dict)
        macro_f1_data = defaultdict(dict)
        
        for model_name in self.model_names:
            for horizon in self.horizons:
                for condition in ['overall', 'steady', 'transition']:
                    metrics_list = self.metrics_data[model_name][horizon][condition]
                    
                    if metrics_list:
                        # Calculate accuracy statistics
                        accuracies = [m['accuracy'] for m in metrics_list if 'accuracy' in m]
                        if accuracies:
                            acc_mean = np.mean(accuracies)
                            acc_std = np.std(accuracies)
                            key = f"{condition}_{horizon}"
                            accuracy_data[model_name][key] = f"{acc_mean:.4f} ± {acc_std:.4f}"
                        
                        # Calculate macro F1 statistics
                        macro_f1s = [m['macro_f1'] for m in metrics_list if 'macro_f1' in m]
                        if macro_f1s:
                            f1_mean = np.mean(macro_f1s)
                            f1_std = np.std(macro_f1s)
                            key = f"{condition}_{horizon}"
                            macro_f1_data[model_name][key] = f"{f1_mean:.4f} ± {f1_std:.4f}"
        
        # Convert to DataFrames
        accuracy_df = pd.DataFrame(accuracy_data).T
        macro_f1_df = pd.DataFrame(macro_f1_data).T
        
        return accuracy_df, macro_f1_df
    
    def create_horizon_plots(self):
        """Create separate plots for each horizon showing all models."""
        print("\nCreating horizon-specific plots...")
        
        for horizon in self.horizons:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'LOSO Results - Horizon {horizon}', fontsize=16, fontweight='bold')
            
            conditions = ['overall', 'steady', 'transition']
            metrics_types = ['accuracy', 'macro_f1']
            
            for metric_idx, metric_type in enumerate(metrics_types):
                for cond_idx, condition in enumerate(conditions):
                    ax = axes[metric_idx, cond_idx]
                    
                    # Prepare data for plotting
                    model_names = []
                    means = []
                    stds = []
                    
                    for model_name in sorted(self.model_names):
                        metrics_list = self.metrics_data[model_name][horizon][condition]
                        
                        if metrics_list:
                            if metric_type == 'accuracy':
                                values = [m['accuracy'] for m in metrics_list if 'accuracy' in m]
                            else:  # macro_f1
                                values = [m['macro_f1'] for m in metrics_list if 'macro_f1' in m]
                            
                            if values:
                                model_names.append(model_name.replace('_loso', '').replace('_', ' '))
                                means.append(np.mean(values))
                                stds.append(np.std(values))
                    
                    if model_names:
                        x_pos = np.arange(len(model_names))
                        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                        
                        # Color bars based on performance
                        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
                        sorted_indices = np.argsort(means)
                        for idx, bar in enumerate(bars):
                            color_idx = np.where(sorted_indices == idx)[0][0]
                            bar.set_color(colors[color_idx])
                        
                        ax.set_xlabel('Model', fontsize=10)
                        ax.set_ylabel(metric_type.replace('_', ' ').title(), fontsize=10)
                        ax.set_title(f'{condition.title()} - {metric_type.replace("_", " ").title()}', fontsize=12)
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
                        ax.set_ylim(0, 1)
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels on bars
                        for i, (mean, std) in enumerate(zip(means, stds)):
                            ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=7)
            
            plt.tight_layout()
            output_path = self.output_dir / f'horizon_{horizon}_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: horizon_{horizon}_comparison.png")
    
    def create_overall_comparison(self):
        """Create simplified comparison of models averaged across all horizons."""
        print("\nCreating overall model comparison across all horizons...")
        
        # Collect data for all models
        overall_stats = {}
        
        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_loso', '').replace('_', ' ')
            overall_stats[model_display] = {}
            
            # Collect metrics for each condition
            for condition in ['overall', 'steady', 'transition']:
                all_accuracies = []
                all_f1s = []
                
                # Aggregate across all horizons
                for horizon in self.horizons:
                    metrics_list = self.metrics_data[model_name][horizon][condition]
                    if metrics_list:
                        accuracies = [m['accuracy'] for m in metrics_list if 'accuracy' in m]
                        f1s = [m['macro_f1'] for m in metrics_list if 'macro_f1' in m]
                        all_accuracies.extend(accuracies)
                        all_f1s.extend(f1s)
                
                if all_accuracies and all_f1s:
                    overall_stats[model_display][condition] = {
                        'acc_mean': np.mean(all_accuracies),
                        'acc_std': np.std(all_accuracies),
                        'f1_mean': np.mean(all_f1s),
                        'f1_std': np.std(all_f1s),
                        'n_samples': len(all_accuracies)
                    }
        
        # Create simplified figure with only 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Overall Model Performance
        ax1 = axes[0]
        
        # Prepare data for overall plot
        models = []
        acc_means = []
        acc_stds = []
        f1_means = []
        f1_stds = []
        
        for model, stats in overall_stats.items():
            if 'overall' in stats:
                models.append(model)
                acc_means.append(stats['overall']['acc_mean'])
                acc_stds.append(stats['overall']['acc_std'])
                f1_means.append(stats['overall']['f1_mean'])
                f1_stds.append(stats['overall']['f1_std'])
        
        # Sort by accuracy (worst to best, left to right)
        sorted_indices = np.argsort(acc_means)
        models = [models[i] for i in sorted_indices]
        acc_means = [acc_means[i] for i in sorted_indices]
        acc_stds = [acc_stds[i] for i in sorted_indices]
        f1_means = [f1_means[i] for i in sorted_indices]
        f1_stds = [f1_stds[i] for i in sorted_indices]
        
        x = np.arange(len(models))
        width = 0.35
        
        # Create bars for overall performance
        bars1 = ax1.bar(x - width/2, acc_means, width, yerr=acc_stds, 
                        label='Accuracy', capsize=5, color='skyblue', alpha=0.8, 
                        error_kw={'linewidth': 1.5})
        bars2 = ax1.bar(x + width/2, f1_means, width, yerr=f1_stds,
                        label='Macro F1', capsize=5, color='lightcoral', alpha=0.8,
                        error_kw={'linewidth': 1.5})
        
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Model Performance Averaged Across All Horizons', 
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars1, acc_means, acc_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, mean, std in zip(bars2, f1_means, f1_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Transition Performance
        ax2 = axes[1]
        
        # Prepare data for transition plot
        trans_models = []
        trans_acc_means = []
        trans_acc_stds = []
        trans_f1_means = []
        trans_f1_stds = []
        
        for model, stats in overall_stats.items():
            if 'transition' in stats:
                trans_models.append(model)
                trans_acc_means.append(stats['transition']['acc_mean'])
                trans_acc_stds.append(stats['transition']['acc_std'])
                trans_f1_means.append(stats['transition']['f1_mean'])
                trans_f1_stds.append(stats['transition']['f1_std'])
        
        # Sort by accuracy (worst to best, left to right)
        sorted_indices = np.argsort(trans_acc_means)
        trans_models = [trans_models[i] for i in sorted_indices]
        trans_acc_means = [trans_acc_means[i] for i in sorted_indices]
        trans_acc_stds = [trans_acc_stds[i] for i in sorted_indices]
        trans_f1_means = [trans_f1_means[i] for i in sorted_indices]
        trans_f1_stds = [trans_f1_stds[i] for i in sorted_indices]
        
        x = np.arange(len(trans_models))
        
        # Create bars for transition performance
        bars3 = ax2.bar(x - width/2, trans_acc_means, width, yerr=trans_acc_stds, 
                        label='Accuracy', capsize=5, color='skyblue', alpha=0.8,
                        error_kw={'linewidth': 1.5})
        bars4 = ax2.bar(x + width/2, trans_f1_means, width, yerr=trans_f1_stds,
                        label='Macro F1', capsize=5, color='lightcoral', alpha=0.8,
                        error_kw={'linewidth': 1.5})
        
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Transition Performance Averaged Across All Horizons', 
                      fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(trans_models, rotation=45, ha='right', fontsize=10)
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars3, trans_acc_means, trans_acc_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, mean, std in zip(bars4, trans_f1_means, trans_f1_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle('LOSO Cross-Validation: Model Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        output_path = self.output_dir / 'overall_model_comparison_all_horizons.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: overall_model_comparison_all_horizons.png")
        
        # Create CSV report
        csv_data = []
        for model, stats in overall_stats.items():
            row = {'Model': model}
            for condition in ['overall', 'steady', 'transition']:
                if condition in stats:
                    s = stats[condition]
                    row[f'{condition.title()} Accuracy'] = f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}"
                    row[f'{condition.title()} Macro F1'] = f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
                    row[f'{condition.title()} N'] = s['n_samples']
            csv_data.append(row)
        
        if csv_data:
            # Sort by overall accuracy for CSV
            df = pd.DataFrame(csv_data)
            df['_sort_key'] = df['Overall Accuracy'].str.extract(r'(\d+\.\d+)').astype(float)
            df = df.sort_values('_sort_key').drop('_sort_key', axis=1)
            
            csv_path = self.output_dir / 'overall_model_comparison_all_horizons.csv'
            df.to_csv(csv_path, index=False)
            print(f"  Saved: overall_model_comparison_all_horizons.csv")
            
            # Print summary
            print("\n" + "="*80)
            print("FINAL AVERAGES ACROSS ALL HORIZONS (Sorted by Overall Accuracy)")
            print("="*80)
            for _, row in df.iterrows():
                print(f"\n{row['Model'].upper()}")
                print("-" * 40)
                print(f"Overall Condition:")
                print(f"  Accuracy: {row['Overall Accuracy']}")
                print(f"  Macro F1: {row['Overall Macro F1']}")
                if 'Steady Accuracy' in row:
                    print(f"Steady Condition:")
                    print(f"  Accuracy: {row['Steady Accuracy']}")
                    print(f"  Macro F1: {row['Steady Macro F1']}")
                if 'Transition Accuracy' in row:
                    print(f"Transition Condition:")
                    print(f"  Accuracy: {row['Transition Accuracy']}")
                    print(f"  Macro F1: {row['Transition Macro F1']}")
            
            # Print ranking
            print("\n" + "="*80)
            print("MODEL RANKING BY OVERALL ACCURACY")
            print("="*80)
            for rank, (_, row) in enumerate(df.iterrows(), 1):
                acc_str = row['Overall Accuracy']
                f1_str = row['Overall Macro F1']
                # Handle both string "mean ± std" format and float (e.g., NaN)
                if isinstance(acc_str, str) and ' ± ' in acc_str:
                    acc_val = float(acc_str.split(' ± ')[0])
                else:
                    acc_val = float(acc_str) if not pd.isna(acc_str) else float('nan')
                if isinstance(f1_str, str) and ' ± ' in f1_str:
                    f1_val = float(f1_str.split(' ± ')[0])
                else:
                    f1_val = float(f1_str) if not pd.isna(f1_str) else float('nan')
                print(f"{rank:2d}. {row['Model']:30s} - Acc: {acc_val:.4f}, F1: {f1_val:.4f}")
        
        return overall_stats

    def create_subject_variance_report(self):
        """
        Create compact subject variance report with:
        - σ_subjects: Std of subject means (each subject's mean across horizons, then std)
        - σ_horizons: Std of horizon means (each horizon's mean across subjects, then std)
        - Compact format: mean ± σ_subj (σ_hor) for paper reporting
        """
        print("\nCreating subject variance report...")

        results = []

        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_loso', '').replace('_', ' ')

            for condition in ['overall', 'steady', 'transition']:
                # === σ_subjects: Std of subject means ===
                # For each subject, compute their mean across all horizons
                subject_means_acc = {}
                subject_means_f1 = {}

                for subject_id in self.subjects:
                    subject_accs = []
                    subject_f1s = []

                    for horizon in self.horizons:
                        if subject_id in self.subject_metrics:
                            if model_name in self.subject_metrics[subject_id]:
                                if horizon in self.subject_metrics[subject_id][model_name]:
                                    if condition in self.subject_metrics[subject_id][model_name][horizon]:
                                        m = self.subject_metrics[subject_id][model_name][horizon][condition]
                                        if 'accuracy' in m:
                                            subject_accs.append(m['accuracy'])
                                        if 'macro_f1' in m:
                                            subject_f1s.append(m['macro_f1'])

                    if subject_accs:
                        subject_means_acc[subject_id] = np.mean(subject_accs)
                    if subject_f1s:
                        subject_means_f1[subject_id] = np.mean(subject_f1s)

                # Compute std of subject means
                if subject_means_acc and subject_means_f1:
                    overall_acc_mean = np.mean(list(subject_means_acc.values()))
                    sigma_subj_acc = np.std(list(subject_means_acc.values()))
                    overall_f1_mean = np.mean(list(subject_means_f1.values()))
                    sigma_subj_f1 = np.std(list(subject_means_f1.values()))
                else:
                    continue

                # === σ_horizons: Std of horizon means ===
                # For each horizon, compute mean across subjects, then std of those means
                horizon_means_acc = []
                horizon_means_f1 = []

                for horizon in self.horizons:
                    metrics_list = self.metrics_data[model_name][horizon][condition]
                    if metrics_list:
                        accs = [m['accuracy'] for m in metrics_list if 'accuracy' in m]
                        f1s = [m['macro_f1'] for m in metrics_list if 'macro_f1' in m]

                        if accs:
                            horizon_means_acc.append(np.mean(accs))
                        if f1s:
                            horizon_means_f1.append(np.mean(f1s))

                if horizon_means_acc and horizon_means_f1:
                    sigma_hor_acc = np.std(horizon_means_acc)
                    sigma_hor_f1 = np.std(horizon_means_f1)
                else:
                    sigma_hor_acc = 0.0
                    sigma_hor_f1 = 0.0

                results.append({
                    'Model': model_display,
                    'Condition': condition.title(),
                    # Raw values for further processing
                    'Mean_Acc': overall_acc_mean,
                    'Mean_F1': overall_f1_mean,
                    'Sigma_Subj_Acc': sigma_subj_acc,
                    'Sigma_Subj_F1': sigma_subj_f1,
                    'Sigma_Hor_Acc': sigma_hor_acc,
                    'Sigma_Hor_F1': sigma_hor_f1,
                    # Formatted for paper: mean ± σ_subj (σ_hor)
                    'Acc_Compact': f"{overall_acc_mean:.3f} ± {sigma_subj_acc:.3f} ({sigma_hor_acc:.3f})",
                    'F1_Compact': f"{overall_f1_mean:.3f} ± {sigma_subj_f1:.3f} ({sigma_hor_f1:.3f})",
                    # Traditional format
                    'Acc_Traditional': f"{overall_acc_mean:.4f} ± {sigma_subj_acc:.4f}",
                    'F1_Traditional': f"{overall_f1_mean:.4f} ± {sigma_subj_f1:.4f}",
                    'N_Subjects': len(subject_means_acc),
                    'N_Horizons': len(self.horizons)
                })

        if results:
            df = pd.DataFrame(results)

            # Save full report
            csv_path = self.output_dir / 'subject_variance_report.csv'
            df.to_csv(csv_path, index=False)
            print(f"  Saved: subject_variance_report.csv")

            # Create compact summary for paper (all conditions, sorted by F1 within each condition)
            compact_cols = ['Model', 'Condition', 'F1_Compact', 'Acc_Compact', 'Sigma_Subj_F1', 'Sigma_Hor_F1']
            compact_df = df[compact_cols].copy()

            # Sort by condition then by F1
            compact_df['_sort'] = compact_df['Sigma_Subj_F1']  # For consistent ordering
            compact_df['Mean_F1'] = df['Mean_F1']
            compact_df = compact_df.sort_values(['Condition', 'Mean_F1'], ascending=[True, False])
            compact_df = compact_df.drop(['_sort', 'Mean_F1'], axis=1)

            compact_path = self.output_dir / 'subject_variance_compact.csv'
            compact_df.to_csv(compact_path, index=False)
            print(f"  Saved: subject_variance_compact.csv")

            # Create paper-ready table (pivoted by condition)
            paper_data = []
            for model in df['Model'].unique():
                model_df = df[df['Model'] == model]
                row = {'Model': model}
                for _, r in model_df.iterrows():
                    cond = r['Condition']
                    row[f'{cond}_F1'] = r['F1_Compact']
                    row[f'{cond}_Acc'] = r['Acc_Compact']
                paper_data.append(row)

            paper_df = pd.DataFrame(paper_data)
            # Sort by Overall F1 mean
            overall_f1_means = df[df['Condition'] == 'Overall'].set_index('Model')['Mean_F1']
            paper_df['_sort'] = paper_df['Model'].map(overall_f1_means)
            paper_df = paper_df.sort_values('_sort', ascending=False).drop('_sort', axis=1)

            paper_path = self.output_dir / 'subject_variance_paper.csv'
            paper_df.to_csv(paper_path, index=False)
            print(f"  Saved: subject_variance_paper.csv")

            # Print summary
            overall_df = df[df['Condition'] == 'Overall'].copy()
            overall_df = overall_df.sort_values('Mean_F1', ascending=False)

            print("\n" + "="*110)
            print("SUBJECT VARIANCE REPORT - Format: mean ± σ_subjects (σ_horizons)")
            print("="*110)
            print(f"{'Model':<40} {'Macro F1':<30} {'Accuracy':<30}")
            print("-"*110)
            for _, row in overall_df.iterrows():
                print(f"{row['Model']:<40} {row['F1_Compact']:<30} {row['Acc_Compact']:<30}")
            print("="*110)
            print("\nFormat: mean ± σ_subjects (σ_horizons)")
            print("  σ_subjects: Std of subject means (inter-subject variability)")
            print("  σ_horizons: Std of horizon means (variation across prediction horizons)")

        return results

    def create_paper_table_per_horizon(self, horizons=['0.0s', '0.5s', '1.0s']):
        """
        Create paper-ready CSV with per-horizon results.

        This generates a table suitable for LaTeX with:
        - Specific horizons (default: 0.0s, 0.5s, 1.0s)
        - Overall and Transition metrics (Acc, F1)
        - Pure inter-subject std at each horizon (no horizon variance mixed in)
        - ALL models are saved (user selects which to include in paper)
        """
        print(f"\nCreating paper table for horizons: {horizons}")

        rows = []

        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_loso', '').replace('_', ' ')

            row = {
                'Model': model_display,
            }

            for horizon in horizons:
                # Get metrics for this horizon
                for condition in ['overall', 'transition']:
                    metrics_list = self.metrics_data[model_name][horizon][condition]

                    if metrics_list:
                        accs = [m['accuracy'] for m in metrics_list if 'accuracy' in m]
                        f1s = [m['macro_f1'] for m in metrics_list if 'macro_f1' in m]

                        cond_prefix = 'Overall' if condition == 'overall' else 'Trans'

                        if accs:
                            mean_acc = np.mean(accs)
                            std_acc = np.std(accs)
                            row[f'{cond_prefix}_Acc_{horizon}'] = f"{mean_acc*100:.1f} ± {std_acc*100:.1f}"
                            row[f'{cond_prefix}_Acc_{horizon}_val'] = mean_acc
                            row[f'{cond_prefix}_Acc_{horizon}_std'] = std_acc

                        if f1s:
                            mean_f1 = np.mean(f1s)
                            std_f1 = np.std(f1s)
                            row[f'{cond_prefix}_F1_{horizon}'] = f"{mean_f1*100:.1f} ± {std_f1*100:.1f}"
                            row[f'{cond_prefix}_F1_{horizon}_val'] = mean_f1
                            row[f'{cond_prefix}_F1_{horizon}_std'] = std_f1

            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)

            # Sort by Overall F1 at first horizon (descending - best first)
            first_horizon = horizons[0]
            sort_col = f'Overall_F1_{first_horizon}_val'
            if sort_col in df.columns:
                df = df.sort_values(sort_col, ascending=False)

            # Select formatted columns for main output
            output_cols = ['Model']
            for horizon in horizons:
                output_cols.extend([
                    f'Overall_Acc_{horizon}',
                    f'Overall_F1_{horizon}',
                    f'Trans_Acc_{horizon}',
                    f'Trans_F1_{horizon}',
                ])

            # Filter to only existing columns
            output_cols = [c for c in output_cols if c in df.columns]
            df_output = df[output_cols].copy()

            # Save formatted table (ALL models)
            csv_path = self.output_dir / 'paper_table_per_horizon.csv'
            df_output.to_csv(csv_path, index=False)
            print(f"  Saved: paper_table_per_horizon.csv ({len(df_output)} models)")

            # Save with raw values for further processing
            raw_path = self.output_dir / 'paper_table_per_horizon_raw.csv'
            df.to_csv(raw_path, index=False)
            print(f"  Saved: paper_table_per_horizon_raw.csv")

            # Print preview (top 10)
            print("\n" + "="*120)
            print(f"PAPER TABLE - Per-Horizon Results (pure inter-subject std) - ALL {len(df_output)} MODELS")
            print(f"Horizons: {', '.join(horizons)}")
            print("="*120)
            print(df_output.head(10).to_string(index=False))
            if len(df_output) > 10:
                print(f"\n... and {len(df_output) - 10} more models (see CSV for full list)")

            return df_output

        return None

    def create_per_class_visualization(self):
        """Create visualization showing per-class F1 scores for each model."""
        print("\nCreating per-class F1 score visualization...")
        
        # Prepare data for visualization
        class_f1_data = defaultdict(dict)
        
        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_loso', '').replace('_', ' ')
            
            for class_name in sorted(self.classes):
                # Get F1 scores for this class across all horizons and subjects
                f1_scores = []
                for horizon in self.horizons:
                    if class_name in self.per_class_metrics[model_name][horizon]:
                        f1_scores.extend(self.per_class_metrics[model_name][horizon][class_name])
                
                if f1_scores:
                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    class_f1_data[model_display][class_name] = {
                        'mean': mean_f1,
                        'std': std_f1
                    }
        
        # Create figure with subplots for each class
        n_classes = len(self.classes)
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, class_name in enumerate(sorted(self.classes)):
            ax = axes[idx]
            
            # Prepare data for this class
            models = []
            means = []
            stds = []
            
            for model in class_f1_data.keys():
                if class_name in class_f1_data[model]:
                    models.append(model)
                    means.append(class_f1_data[model][class_name]['mean'])
                    stds.append(class_f1_data[model][class_name]['std'])
            
            if models:
                # Sort by F1 score (worst to best)
                sorted_indices = np.argsort(means)
                models = [models[i] for i in sorted_indices]
                means = [means[i] for i in sorted_indices]
                stds = [stds[i] for i in sorted_indices]
                
                x_pos = np.arange(len(models))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=3, 
                              color='steelblue', alpha=0.7, error_kw={'linewidth': 1})
                
                ax.set_xlabel('Model', fontsize=9)
                ax.set_ylabel('F1 Score', fontsize=9)
                ax.set_title(f'{class_name}', fontsize=10, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=7)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, (mean, std) in enumerate(zip(means, stds)):
                    ax.text(i, mean + std + 0.01, f'{mean:.2f}', 
                           ha='center', va='bottom', fontsize=6)
        
        # Hide unused subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Per-Class F1 Scores Averaged Across All Horizons', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        output_path = self.output_dir / 'per_class_f1_scores_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: per_class_f1_scores_visualization.png")
        
        # Note: Heatmaps are now created separately by condition and horizon
    
    def create_per_class_heatmaps_by_condition(self):
        """Create separate heatmaps for overall, steady, and transition conditions."""
        print("\nCreating per-class F1 score heatmaps by condition...")
        
        conditions = ['overall', 'steady', 'transition']
        
        for condition in conditions:
            # Prepare data for this condition
            class_f1_data = defaultdict(dict)
            
            for model_name in sorted(self.model_names):
                model_display = model_name.replace('_loso', '').replace('_', ' ')
                
                for class_name in sorted(self.classes):
                    # Get F1 scores for this class across all horizons for this condition
                    f1_scores = []
                    for horizon in self.horizons:
                        metrics_list = self.metrics_data[model_name][horizon][condition]
                        if metrics_list:
                            for metrics in metrics_list:
                                if 'per_class_f1' in metrics and class_name in metrics['per_class_f1']:
                                    f1_scores.append(metrics['per_class_f1'][class_name])
                    
                    if f1_scores:
                        mean_f1 = np.mean(f1_scores)
                        class_f1_data[model_display][class_name] = mean_f1
            
            # Create heatmap for this condition
            if class_f1_data:
                self._create_single_heatmap(
                    class_f1_data, 
                    f'Per-Class F1 Scores - {condition.title()} Condition (All Horizons)',
                    f'per_class_f1_heatmap_{condition}.png'
                )

    def create_per_class_heatmaps_by_horizon(self):
        """Create separate heatmaps for each horizon."""
        print("\nCreating per-class F1 score heatmaps by horizon...")
        
        for horizon in self.horizons:
            # Prepare data for this horizon
            class_f1_data = defaultdict(dict)
            
            for model_name in sorted(self.model_names):
                model_display = model_name.replace('_loso', '').replace('_', ' ')
                
                for class_name in sorted(self.classes):
                    # Get F1 scores for this class at this specific horizon (overall condition)
                    f1_scores = []
                    metrics_list = self.metrics_data[model_name][horizon]['overall']
                    if metrics_list:
                        for metrics in metrics_list:
                            if 'per_class_f1' in metrics and class_name in metrics['per_class_f1']:
                                f1_scores.append(metrics['per_class_f1'][class_name])
                    
                    if f1_scores:
                        mean_f1 = np.mean(f1_scores)
                        class_f1_data[model_display][class_name] = mean_f1
            
            # Create heatmap for this horizon
            if class_f1_data:
                self._create_single_heatmap(
                    class_f1_data, 
                    f'Per-Class F1 Scores - Horizon {horizon} (Overall Condition)',
                    f'per_class_f1_heatmap_horizon_{horizon.replace(".", "_")}.png'
                )

    def create_per_class_heatmaps_transition_by_horizon(self):
        """Create separate heatmaps for transition condition at each horizon."""
        print("\nCreating per-class F1 score heatmaps for transitions by horizon...")
        
        for horizon in self.horizons:
            # Prepare data for transition at this horizon
            class_f1_data = defaultdict(dict)
            
            for model_name in sorted(self.model_names):
                model_display = model_name.replace('_loso', '').replace('_', ' ')
                
                for class_name in sorted(self.classes):
                    # Get F1 scores for this class at this specific horizon (transition condition)
                    f1_scores = []
                    metrics_list = self.metrics_data[model_name][horizon]['transition']
                    if metrics_list:
                        for metrics in metrics_list:
                            if 'per_class_f1' in metrics and class_name in metrics['per_class_f1']:
                                f1_scores.append(metrics['per_class_f1'][class_name])
                    
                    if f1_scores:
                        mean_f1 = np.mean(f1_scores)
                        class_f1_data[model_display][class_name] = mean_f1
            
            # Create heatmap for this horizon's transitions
            if class_f1_data:
                self._create_single_heatmap(
                    class_f1_data, 
                    f'Per-Class F1 Scores - Horizon {horizon} (Transition Condition)',
                    f'per_class_f1_heatmap_transition_horizon_{horizon.replace(".", "_")}.png'
                )

    def _create_single_heatmap(self, class_f1_data, title, filename):
        """Helper function to create a single heatmap."""
        
        # Prepare data for heatmap
        models = sorted(class_f1_data.keys())
        classes = sorted(self.classes)
        
        # Create matrix
        f1_matrix = np.zeros((len(models), len(classes)))
        
        for i, model in enumerate(models):
            for j, class_name in enumerate(classes):
                if class_name in class_f1_data[model]:
                    f1_matrix[i, j] = class_f1_data[model][class_name]
                else:
                    f1_matrix[i, j] = np.nan
        
        # Sort models by average F1 score for better visualization
        avg_f1_per_model = np.nanmean(f1_matrix, axis=1)
        sorted_indices = np.argsort(avg_f1_per_model)
        f1_matrix = f1_matrix[sorted_indices, :]
        models = [models[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use a masked array to handle NaN values
        masked_matrix = np.ma.masked_invalid(f1_matrix)
        
        # Create heatmap
        im = ax.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(models, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1 Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(classes)):
                if not np.isnan(f1_matrix[i, j]):
                    value = f1_matrix[i, j]
                    # Choose text color based on background
                    text_color = 'black' if value < 0.5 else 'white'
                    text = ax.text(j, i, f'{value:.2f}',
                                  ha='center', va='center', color=text_color, 
                                  fontsize=7, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Activity Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model (sorted by avg F1)', fontsize=11, fontweight='bold')
        
        # Add grid for better readability
        ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

    def create_comparison_heatmap_grid(self):
        """Create a grid of smaller heatmaps comparing conditions side by side."""
        print("\nCreating comparison heatmap grid...")
        
        conditions = ['overall', 'steady', 'transition']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        for idx, condition in enumerate(conditions):
            ax = axes[idx]
            
            # Prepare data for this condition
            class_f1_data = defaultdict(dict)
            
            for model_name in sorted(self.model_names):
                model_display = model_name.replace('_loso', '').replace('_', ' ')
                
                for class_name in sorted(self.classes):
                    f1_scores = []
                    for horizon in self.horizons:
                        metrics_list = self.metrics_data[model_name][horizon][condition]
                        if metrics_list:
                            for metrics in metrics_list:
                                if 'per_class_f1' in metrics and class_name in metrics['per_class_f1']:
                                    f1_scores.append(metrics['per_class_f1'][class_name])
                    
                    if f1_scores:
                        mean_f1 = np.mean(f1_scores)
                        class_f1_data[model_display][class_name] = mean_f1
            
            # Prepare data for heatmap
            models = sorted(class_f1_data.keys())
            classes = sorted(self.classes)
            
            # Create matrix
            f1_matrix = np.zeros((len(models), len(classes)))
            
            for i, model in enumerate(models):
                for j, class_name in enumerate(classes):
                    if class_name in class_f1_data[model]:
                        f1_matrix[i, j] = class_f1_data[model][class_name]
                    else:
                        f1_matrix[i, j] = np.nan
            
            # Sort models by average F1 score
            avg_f1_per_model = np.nanmean(f1_matrix, axis=1)
            sorted_indices = np.argsort(avg_f1_per_model)
            f1_matrix = f1_matrix[sorted_indices, :]
            models = [models[i] for i in sorted_indices]
            
            # Use a masked array to handle NaN values
            masked_matrix = np.ma.masked_invalid(f1_matrix)
            
            # Create heatmap
            im = ax.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(models)))
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(models, fontsize=7)
            
            # Add text annotations (only for values, smaller font)
            for i in range(len(models)):
                for j in range(len(classes)):
                    if not np.isnan(f1_matrix[i, j]):
                        value = f1_matrix[i, j]
                        text_color = 'black' if value < 0.5 else 'white'
                        text = ax.text(j, i, f'{value:.1f}',
                                      ha='center', va='center', color=text_color, 
                                      fontsize=5)
            
            ax.set_title(f'{condition.title()} Condition', fontsize=11, fontweight='bold')
            ax.set_xlabel('Activity Class', fontsize=9)
            if idx == 0:
                ax.set_ylabel('Model (sorted by avg F1)', fontsize=9)
            
            # Add grid
            ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
            ax.tick_params(which='minor', size=0)
        
        # Add single colorbar for all subplots
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('F1 Score', rotation=270, labelpad=15)
        
        plt.suptitle('Per-Class F1 Scores - Condition Comparison', fontsize=14, fontweight='bold')
        
        # Save figure
        output_path = self.output_dir / 'per_class_f1_heatmap_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: per_class_f1_heatmap_comparison.png")
    
    def create_individual_subject_reports(self):
        """Create individual analysis reports for each subject."""
        if not self.create_subject_reports:
            return
            
        print("\n" + "="*80)
        print("CREATING INDIVIDUAL SUBJECT REPORTS")
        print("="*80)
        
        subjects_dir = self.output_dir / 'individual_subjects'
        subjects_dir.mkdir(exist_ok=True)
        
        for subject_id in sorted(self.subjects):
            print(f"\nProcessing Subject {subject_id}...")
            
            # Create subject directory
            subject_dir = subjects_dir / f'subject_{subject_id}'
            subject_dir.mkdir(exist_ok=True)
            
            # Create comparison plots for this subject
            self._create_subject_comparison_plot(subject_id, subject_dir)
            
            # Create per-class heatmap for this subject
            self._create_subject_per_class_heatmap(subject_id, subject_dir)
            
            # Create CSV report for this subject
            self._create_subject_csv_report(subject_id, subject_dir)
            
            # Create horizon-specific plots for this subject
            self._create_subject_horizon_plots(subject_id, subject_dir)
        
        print(f"\n✅ Individual subject reports created in: {subjects_dir}")
    
    def _create_subject_comparison_plot(self, subject_id: str, subject_dir: Path):
        """Create overall and transition comparison plot for a single subject."""
        
        # Collect data for this subject
        subject_stats = {}
        
        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_loso', '').replace('_', ' ')
            subject_stats[model_display] = {}
            
            for condition in ['overall', 'steady', 'transition']:
                accuracies = []
                f1s = []
                
                for horizon in self.horizons:
                    if subject_id in self.subject_metrics:
                        metrics = self.subject_metrics[subject_id].get(model_name, {}).get(horizon, {}).get(condition, {})
                        if 'accuracy' in metrics:
                            accuracies.append(metrics['accuracy'])
                        if 'macro_f1' in metrics:
                            f1s.append(metrics['macro_f1'])
                
                if accuracies and f1s:
                    subject_stats[model_display][condition] = {
                        'acc_mean': np.mean(accuracies),
                        'f1_mean': np.mean(f1s)
                    }
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Overall Performance
        ax1 = axes[0]
        models = []
        acc_means = []
        f1_means = []
        
        for model, stats in subject_stats.items():
            if 'overall' in stats:
                models.append(model)
                acc_means.append(stats['overall']['acc_mean'])
                f1_means.append(stats['overall']['f1_mean'])
        
        if models:
            # Sort by accuracy
            sorted_indices = np.argsort(acc_means)
            models = [models[i] for i in sorted_indices]
            acc_means = [acc_means[i] for i in sorted_indices]
            f1_means = [f1_means[i] for i in sorted_indices]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, acc_means, width, label='Accuracy', 
                           color='skyblue', alpha=0.8)
            bars2 = ax1.bar(x + width/2, f1_means, width, label='Macro F1', 
                           color='lightcoral', alpha=0.8)
            
            ax1.set_xlabel('Model', fontsize=11)
            ax1.set_ylabel('Score', fontsize=11)
            ax1.set_title(f'Overall Performance - Subject {subject_id}', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 1.05)
            
            # Add value labels
            for bar, val in zip(bars1, acc_means):
                ax1.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            for bar, val in zip(bars2, f1_means):
                ax1.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Transition Performance
        ax2 = axes[1]
        trans_models = []
        trans_acc_means = []
        trans_f1_means = []
        
        for model, stats in subject_stats.items():
            if 'transition' in stats:
                trans_models.append(model)
                trans_acc_means.append(stats['transition']['acc_mean'])
                trans_f1_means.append(stats['transition']['f1_mean'])
        
        if trans_models:
            # Sort by accuracy
            sorted_indices = np.argsort(trans_acc_means)
            trans_models = [trans_models[i] for i in sorted_indices]
            trans_acc_means = [trans_acc_means[i] for i in sorted_indices]
            trans_f1_means = [trans_f1_means[i] for i in sorted_indices]
            
            x = np.arange(len(trans_models))
            
            bars3 = ax2.bar(x - width/2, trans_acc_means, width, label='Accuracy',
                           color='skyblue', alpha=0.8)
            bars4 = ax2.bar(x + width/2, trans_f1_means, width, label='Macro F1',
                           color='lightcoral', alpha=0.8)
            
            ax2.set_xlabel('Model', fontsize=11)
            ax2.set_ylabel('Score', fontsize=11)
            ax2.set_title(f'Transition Performance - Subject {subject_id}', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(trans_models, rotation=45, ha='right', fontsize=9)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1.05)
            
            # Add value labels
            for bar, val in zip(bars3, trans_acc_means):
                ax2.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            for bar, val in zip(bars4, trans_f1_means):
                ax2.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'Subject {subject_id} - Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        output_path = subject_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_subject_per_class_heatmap(self, subject_id: str, subject_dir: Path):
        """Create per-class F1 heatmap for a single subject."""
        
        # Prepare data
        class_f1_data = defaultdict(dict)
        
        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_loso', '').replace('_', ' ')
            
            for class_name in sorted(self.classes):
                f1_scores = []
                
                for horizon in self.horizons:
                    metrics = self.subject_metrics[subject_id].get(model_name, {}).get(horizon, {}).get('overall', {})
                    if 'per_class_f1' in metrics and class_name in metrics['per_class_f1']:
                        f1_scores.append(metrics['per_class_f1'][class_name])
                
                if f1_scores:
                    class_f1_data[model_display][class_name] = np.mean(f1_scores)
        
        if not class_f1_data:
            return
        
        # Create heatmap
        models = sorted(class_f1_data.keys())
        classes = sorted(self.classes)
        
        f1_matrix = np.zeros((len(models), len(classes)))
        
        for i, model in enumerate(models):
            for j, class_name in enumerate(classes):
                if class_name in class_f1_data[model]:
                    f1_matrix[i, j] = class_f1_data[model][class_name]
                else:
                    f1_matrix[i, j] = np.nan
        
        # Sort models by average F1
        avg_f1_per_model = np.nanmean(f1_matrix, axis=1)
        sorted_indices = np.argsort(avg_f1_per_model)
        f1_matrix = f1_matrix[sorted_indices, :]
        models = [models[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        masked_matrix = np.ma.masked_invalid(f1_matrix)
        im = ax.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(models, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1 Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(classes)):
                if not np.isnan(f1_matrix[i, j]):
                    value = f1_matrix[i, j]
                    text_color = 'black' if value < 0.5 else 'white'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontsize=6, fontweight='bold')
        
        ax.set_title(f'Per-Class F1 Scores - Subject {subject_id}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Activity Class', fontsize=10)
        ax.set_ylabel('Model (sorted by avg F1)', fontsize=10)
        
        # Add grid
        ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
        plt.tight_layout()
        
        output_path = subject_dir / 'per_class_f1_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_subject_csv_report(self, subject_id: str, subject_dir: Path):
        """Create CSV report for a single subject."""
        
        csv_data = []
        
        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_loso', '').replace('_', ' ')
            row = {'Model': model_display}
            
            for condition in ['overall', 'steady', 'transition']:
                accuracies = []
                f1s = []
                
                for horizon in self.horizons:
                    metrics = self.subject_metrics[subject_id].get(model_name, {}).get(horizon, {}).get(condition, {})
                    if 'accuracy' in metrics:
                        accuracies.append(metrics['accuracy'])
                    if 'macro_f1' in metrics:
                        f1s.append(metrics['macro_f1'])
                
                if accuracies and f1s:
                    row[f'{condition.title()} Accuracy'] = f"{np.mean(accuracies):.4f}"
                    row[f'{condition.title()} Macro F1'] = f"{np.mean(f1s):.4f}"
            
            csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = subject_dir / 'metrics_summary.csv'
            df.to_csv(csv_path, index=False)
    
    def _create_subject_horizon_plots(self, subject_id: str, subject_dir: Path):
        """Create horizon-specific plots for a single subject."""
        
        horizons_dir = subject_dir / 'by_horizon'
        horizons_dir.mkdir(exist_ok=True)
        
        for horizon in self.horizons[:3]:  # Create plots for first 3 horizons
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            conditions = ['overall', 'steady', 'transition']
            
            for idx, condition in enumerate(conditions):
                ax = axes[idx]
                
                models = []
                accuracies = []
                f1s = []
                
                for model_name in sorted(self.model_names):
                    model_display = model_name.replace('_loso', '').replace('_', ' ')
                    metrics = self.subject_metrics[subject_id].get(model_name, {}).get(horizon, {}).get(condition, {})
                    
                    if 'accuracy' in metrics and 'macro_f1' in metrics:
                        models.append(model_display)
                        accuracies.append(metrics['accuracy'])
                        f1s.append(metrics['macro_f1'])
                
                if models:
                    # Sort by accuracy
                    sorted_indices = np.argsort(accuracies)
                    models = [models[i] for i in sorted_indices]
                    accuracies = [accuracies[i] for i in sorted_indices]
                    f1s = [f1s[i] for i in sorted_indices]
                    
                    x = np.arange(len(models))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                                  color='skyblue', alpha=0.8)
                    bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1',
                                  color='lightcoral', alpha=0.8)
                    
                    ax.set_xlabel('Model', fontsize=9)
                    ax.set_ylabel('Score', fontsize=9)
                    ax.set_title(f'{condition.title()}', fontsize=10, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=7)
                    if idx == 0:
                        ax.legend(loc='upper left', fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_ylim(0, 1.05)
            
            plt.suptitle(f'Subject {subject_id} - Horizon {horizon}', fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            output_path = horizons_dir / f'horizon_{horizon.replace(".", "_")}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_loss_curves(self):
        """Create loss curve plots for each model."""
        print("\nCreating loss curve plots...")
        
        loss_dir = self.output_dir / 'loss_curves'
        loss_dir.mkdir(exist_ok=True)
        
        for model_name in self.model_names:
            if model_name not in self.loss_data:
                continue
                
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Training vs Validation Loss - {model_name}', fontsize=14, fontweight='bold')
            axes = axes.flatten()
            
            for idx, subject_data in enumerate(self.loss_data[model_name][:8]):  # Show up to 8 subjects
                ax = axes[idx]
                
                epochs = range(1, len(subject_data['train_losses']) + 1)
                ax.plot(epochs, subject_data['train_losses'], 'b-', label='Training Loss', linewidth=2)
                ax.plot(epochs, subject_data['val_losses'], 'r-', label='Validation Loss', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f"Subject {subject_data['subject']}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Highlight overfitting region if validation loss increases
                val_losses = subject_data['val_losses']
                min_val_idx = np.argmin(val_losses)
                if min_val_idx < len(val_losses) - 1:
                    ax.axvspan(min_val_idx + 1, len(val_losses), alpha=0.2, color='red')
            
            # Hide unused subplots
            for idx in range(len(self.loss_data[model_name]), 8):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            output_path = loss_dir / f'{model_name}_loss_curves.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {model_name}_loss_curves.png")
    
    def create_learning_rate_curves(self):
        """Create learning rate plots for each model."""
        print("\nCreating learning rate plots...")
        
        lr_dir = self.output_dir / 'learning_rate_curves'
        lr_dir.mkdir(exist_ok=True)
        
        for model_name in self.model_names:
            if model_name not in self.lr_data:
                print(f"  No learning rate data found for {model_name}")
                continue
                
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Learning Rate Schedule - {model_name}', fontsize=14, fontweight='bold')
            axes = axes.flatten()
            
            # Track if we have any data to plot
            has_data = False
            
            for idx, subject_data in enumerate(self.lr_data[model_name][:8]):  # Show up to 8 subjects
                ax = axes[idx]
                
                learning_rates = subject_data['learning_rates']
                if learning_rates:
                    has_data = True
                    epochs = range(1, len(learning_rates) + 1)
                    
                    # Plot learning rate curve
                    ax.plot(epochs, learning_rates, 'g-', linewidth=2)
                    
                    # Use log scale if learning rates vary over multiple orders of magnitude
                    min_lr_val = min(learning_rates)
                    max_lr_val = max(learning_rates)
                    if min_lr_val > 0 and max_lr_val / min_lr_val > 10:
                        ax.set_yscale('log')
                    
                    ax.set_xlabel('Epoch', fontsize=9)
                    ax.set_ylabel('Learning Rate', fontsize=9)
                    ax.set_title(f"Subject {subject_data['subject']}", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add annotations for key points
                    if len(set(learning_rates)) > 1:  # If LR changes
                        # Mark the point where LR changes significantly
                        for i in range(1, len(learning_rates)):
                            if abs(learning_rates[i] - learning_rates[i-1]) > learning_rates[i-1] * 0.1:
                                ax.axvline(x=i+1, color='r', linestyle='--', alpha=0.5, linewidth=1)
                    
                    # Add min/max annotations
                    min_lr = min(learning_rates)
                    max_lr = max(learning_rates)
                    min_idx = learning_rates.index(min_lr)
                    max_idx = learning_rates.index(max_lr)
                    
                    ax.annotate(f'Max: {max_lr:.2e}', 
                               xy=(max_idx+1, max_lr),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=7, color='blue')
                    ax.annotate(f'Min: {min_lr:.2e}',
                               xy=(min_idx+1, min_lr),
                               xytext=(5, -15), textcoords='offset points',
                               fontsize=7, color='red')
                else:
                    ax.text(0.5, 0.5, 'No LR data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_visible(False)
            
            # Hide unused subplots
            for idx in range(len(self.lr_data[model_name]), 8):
                axes[idx].set_visible(False)
            
            if has_data:
                plt.tight_layout()
                output_path = lr_dir / f'{model_name}_learning_rate_curves.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"  Saved: {model_name}_learning_rate_curves.png")
            else:
                print(f"  No learning rate data to plot for {model_name}")
            
            plt.close()
    
    def create_combined_training_curves(self):
        """Create combined plots showing loss and learning rate together."""
        print("\nCreating combined training curves (loss + LR)...")
        
        combined_dir = self.output_dir / 'combined_training_curves'
        combined_dir.mkdir(exist_ok=True)
        
        for model_name in self.model_names:
            if model_name not in self.loss_data:
                continue
                
            # Create figure with 2 rows per subject (loss and LR)
            n_subjects = min(4, len(self.loss_data[model_name]))  # Show up to 4 subjects
            if n_subjects == 0:
                continue
                
            fig, axes = plt.subplots(n_subjects, 2, figsize=(14, 3*n_subjects))
            if n_subjects == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Training Dynamics - {model_name}', fontsize=14, fontweight='bold')
            
            for idx in range(n_subjects):
                # Loss plot
                ax_loss = axes[idx, 0]
                loss_data = self.loss_data[model_name][idx]
                
                epochs = range(1, len(loss_data['train_losses']) + 1)
                ax_loss.plot(epochs, loss_data['train_losses'], 'b-', 
                           label='Training Loss', linewidth=2)
                ax_loss.plot(epochs, loss_data['val_losses'], 'r-', 
                           label='Validation Loss', linewidth=2)
                
                ax_loss.set_xlabel('Epoch', fontsize=9)
                ax_loss.set_ylabel('Loss', fontsize=9)
                ax_loss.set_title(f"Subject {loss_data['subject']} - Loss", fontsize=10)
                ax_loss.legend(fontsize=8)
                ax_loss.grid(True, alpha=0.3)
                
                # Highlight overfitting region
                val_losses = loss_data['val_losses']
                min_val_idx = np.argmin(val_losses)
                if min_val_idx < len(val_losses) - 1:
                    ax_loss.axvspan(min_val_idx + 1, len(val_losses), 
                                   alpha=0.2, color='red')
                    ax_loss.axvline(x=min_val_idx + 1, color='red', 
                                  linestyle='--', alpha=0.5, linewidth=1,
                                  label='Best Val')
                
                # Learning rate plot
                ax_lr = axes[idx, 1]
                
                # Find matching LR data for this subject
                lr_subject_data = None
                for lr_data in self.lr_data.get(model_name, []):
                    if lr_data['subject'] == loss_data['subject']:
                        lr_subject_data = lr_data
                        break
                
                if lr_subject_data and lr_subject_data['learning_rates']:
                    learning_rates = lr_subject_data['learning_rates']
                    lr_epochs = range(1, len(learning_rates) + 1)
                    
                    ax_lr.plot(lr_epochs, learning_rates, 'g-', linewidth=2)

                    min_lr_val = min(learning_rates)
                    max_lr_val = max(learning_rates)
                    if min_lr_val > 0 and max_lr_val / min_lr_val > 10:
                        ax_lr.set_yscale('log')
                    
                    ax_lr.set_xlabel('Epoch', fontsize=9)
                    ax_lr.set_ylabel('Learning Rate', fontsize=9)
                    ax_lr.set_title(f"Subject {loss_data['subject']} - Learning Rate", fontsize=10)
                    ax_lr.grid(True, alpha=0.3)
                    
                    # Mark LR changes
                    for i in range(1, len(learning_rates)):
                        if abs(learning_rates[i] - learning_rates[i-1]) > learning_rates[i-1] * 0.1:
                            ax_lr.axvline(x=i+1, color='orange', linestyle='--', 
                                        alpha=0.5, linewidth=1)
                else:
                    ax_lr.text(0.5, 0.5, 'No LR data available', 
                             ha='center', va='center', transform=ax_lr.transAxes)
                    ax_lr.set_visible(False)
            
            plt.tight_layout()
            output_path = combined_dir / f'{model_name}_combined_training_curves.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {model_name}_combined_training_curves.png")
    
    def create_csv_reports(self):
        """Create CSV reports for accuracy and macro F1 scores."""
        print("\nCreating CSV reports...")
        
        # Prepare data structures for CSV
        for condition in ['overall', 'steady', 'transition']:
            accuracy_data = {}
            macro_f1_data = {}
            
            for model_name in sorted(self.model_names):
                accuracy_row = {}
                macro_f1_row = {}
                
                for horizon in self.horizons:
                    metrics_list = self.metrics_data[model_name][horizon][condition]
                    
                    if metrics_list:
                        # Accuracy
                        accuracies = [m['accuracy'] for m in metrics_list if 'accuracy' in m]
                        if accuracies:
                            acc_mean = np.mean(accuracies)
                            acc_std = np.std(accuracies)
                            accuracy_row[horizon] = f"{acc_mean:.4f} ± {acc_std:.4f}"
                        
                        # Macro F1
                        macro_f1s = [m['macro_f1'] for m in metrics_list if 'macro_f1' in m]
                        if macro_f1s:
                            f1_mean = np.mean(macro_f1s)
                            f1_std = np.std(macro_f1s)
                            macro_f1_row[horizon] = f"{f1_mean:.4f} ± {f1_std:.4f}"
                
                if accuracy_row:
                    model_display_name = model_name.replace('_loso', '').replace('_', ' ')
                    accuracy_data[model_display_name] = accuracy_row
                    macro_f1_data[model_display_name] = macro_f1_row
            
            # Save accuracy CSV
            if accuracy_data:
                accuracy_df = pd.DataFrame(accuracy_data).T
                accuracy_df.index.name = 'Model'
                accuracy_path = self.output_dir / f'{condition}_accuracy_results.csv'
                accuracy_df.to_csv(accuracy_path)
                print(f"  Saved: {condition}_accuracy_results.csv")
            
            # Save macro F1 CSV
            if macro_f1_data:
                macro_f1_df = pd.DataFrame(macro_f1_data).T
                macro_f1_df.index.name = 'Model'
                macro_f1_path = self.output_dir / f'{condition}_macro_f1_results.csv'
                macro_f1_df.to_csv(macro_f1_path)
                print(f"  Saved: {condition}_macro_f1_results.csv")
    
    def create_per_class_report(self):
        """Create a report showing per-class F1 scores averaged across subjects.

        Now generates separate CSVs for each condition (overall, steady, transition)
        using the correct data source (metrics_data) that tracks conditions properly.
        """
        print("\nCreating per-class F1 reports by condition...")

        conditions = ['overall', 'steady', 'transition']

        for condition in conditions:
            # Organize data by class for this condition
            class_data = defaultdict(dict)

            for model_name in sorted(self.model_names):
                model_display_name = model_name.replace('_loso', '').replace('_', ' ')

                for class_name in sorted(self.classes):
                    # Get F1 scores for this class across all horizons for THIS condition
                    f1_scores = []
                    for horizon in self.horizons:
                        metrics_list = self.metrics_data[model_name][horizon][condition]
                        if metrics_list:
                            for metrics in metrics_list:
                                if 'per_class_f1' in metrics and class_name in metrics['per_class_f1']:
                                    f1_scores.append(metrics['per_class_f1'][class_name])

                    if f1_scores:
                        mean_f1 = np.mean(f1_scores)
                        std_f1 = np.std(f1_scores)
                        class_data[class_name][model_display_name] = f"{mean_f1:.4f} ± {std_f1:.4f}"

            # Create DataFrame and save for this condition
            if class_data:
                class_df = pd.DataFrame(class_data).T
                class_df.index.name = 'Activity Class'
                class_path = self.output_dir / f'per_class_f1_scores_{condition}.csv'
                class_df.to_csv(class_path)
                print(f"  Saved: per_class_f1_scores_{condition}.csv")

        # Generate per-horizon per-class CSVs for overall and transition conditions
        print("\n  Creating per-horizon per-class F1 CSVs...")
        for condition in ['overall', 'transition']:
            for horizon in self.horizons:
                class_data = defaultdict(dict)

                for model_name in sorted(self.model_names):
                    model_display_name = model_name.replace('_loso', '').replace('_', ' ')

                    for class_name in sorted(self.classes):
                        f1_scores = []
                        metrics_list = self.metrics_data[model_name][horizon][condition]
                        if metrics_list:
                            for metrics in metrics_list:
                                if 'per_class_f1' in metrics and class_name in metrics['per_class_f1']:
                                    f1_scores.append(metrics['per_class_f1'][class_name])

                        if f1_scores:
                            mean_f1 = np.mean(f1_scores)
                            std_f1 = np.std(f1_scores)
                            class_data[class_name][model_display_name] = f"{mean_f1:.4f} ± {std_f1:.4f}"

                if class_data:
                    class_df = pd.DataFrame(class_data).T
                    class_df.index.name = 'Activity Class'
                    horizon_label = horizon.replace('.', '_')
                    class_path = self.output_dir / f'per_class_f1_scores_{condition}_horizon_{horizon_label}.csv'
                    class_df.to_csv(class_path)
                    print(f"  Saved: per_class_f1_scores_{condition}_horizon_{horizon_label}.csv")

        # Also create the legacy combined file for backward compatibility (marked as deprecated)
        print("\n  Note: Creating legacy per_class_f1_scores.csv (deprecated - uses mixed conditions)")
        class_data = defaultdict(dict)
        for model_name in sorted(self.model_names):
            for class_name in sorted(self.classes):
                f1_scores = []
                for horizon in self.horizons:
                    if class_name in self.per_class_metrics[model_name][horizon]:
                        f1_scores.extend(self.per_class_metrics[model_name][horizon][class_name])

                if f1_scores:
                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    model_display_name = model_name.replace('_loso', '').replace('_', ' ')
                    class_data[class_name][model_display_name] = f"{mean_f1:.4f} ± {std_f1:.4f}"

        if class_data:
            class_df = pd.DataFrame(class_data).T
            class_df.index.name = 'Activity Class'
            class_path = self.output_dir / 'per_class_f1_scores.csv'
            class_df.to_csv(class_path)
            print(f"  Saved: per_class_f1_scores.csv (deprecated)")
    
    def print_summary(self):
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("LOSO CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        for model_name in sorted(self.model_names):
            print(f"\n{model_name.replace('_', ' ').upper()}")
            print("-" * 40)
            
            for horizon in self.horizons[:3]:  # Show first 3 horizons as example
                print(f"\nHorizon {horizon}:")
                
                for condition in ['overall', 'steady', 'transition']:
                    metrics_list = self.metrics_data[model_name][horizon][condition]
                    
                    if metrics_list:
                        accuracies = [m['accuracy'] for m in metrics_list if 'accuracy' in m]
                        macro_f1s = [m['macro_f1'] for m in metrics_list if 'macro_f1' in m]
                        
                        if accuracies and macro_f1s:
                            print(f"  {condition.title():12} - Acc: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}, "
                                  f"F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced LOSO cross-validation results analyzer')
    parser.add_argument('--base-dir', type=str, default='outputs/loso/aidwear',
                       help='Base directory containing LOSO results')
    parser.add_argument('--output-dir', type=str, default='analysis_results_v3',
                       help='Output directory for analysis results')
    parser.add_argument('--no-subject-reports', action='store_true',
                       help='Skip creating individual subject reports')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = EnhancedLOSOAnalyzer(
        args.base_dir, 
        args.output_dir,
        create_subject_reports=not args.no_subject_reports
    )
    
    # Extract metrics
    analyzer.scan_and_extract_metrics()
    
    # Create visualizations and reports
    analyzer.create_horizon_plots()
    analyzer.create_loss_curves()
    analyzer.create_learning_rate_curves()  # NEW
    analyzer.create_combined_training_curves()  # NEW
    analyzer.create_csv_reports()
    analyzer.create_per_class_report()
    
    # Create overall comparison across all horizons
    analyzer.create_overall_comparison()

    # Create subject variance report (Options B and C)
    analyzer.create_subject_variance_report()

    # Create paper table with per-horizon results (pure inter-subject std)
    analyzer.create_paper_table_per_horizon(horizons=['0.0s', '0.5s', '1.0s'])

    # Create per-class visualizations
    analyzer.create_per_class_visualization()
    
    # Create multiple heatmaps by condition and horizon
    analyzer.create_per_class_heatmaps_by_condition()
    analyzer.create_per_class_heatmaps_by_horizon()
    analyzer.create_per_class_heatmaps_transition_by_horizon()
    analyzer.create_comparison_heatmap_grid()
    
    # Create individual subject reports
    analyzer.create_individual_subject_reports()
    
    # Print summary
    analyzer.print_summary()
    
    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")
    print(f"\n📊 Key outputs:")
    print(f"  - overall_model_comparison_all_horizons.png: Overall and transition comparison")
    print(f"  - per_class_f1_scores_visualization.png: Individual class performance")
    print(f"\n🔥 Heatmap outputs:")
    print(f"  By Condition (all horizons averaged):")
    print(f"    - per_class_f1_heatmap_overall.png")
    print(f"    - per_class_f1_heatmap_steady.png")
    print(f"    - per_class_f1_heatmap_transition.png")
    print(f"  By Horizon (overall condition):")
    for horizon in analyzer.horizons:
        print(f"    - per_class_f1_heatmap_horizon_{horizon.replace('.', '_')}.png")
    print(f"  By Horizon (transition condition):")
    for horizon in analyzer.horizons:
        print(f"    - per_class_f1_heatmap_transition_horizon_{horizon.replace('.', '_')}.png")
    print(f"  Comparison Grid:")
    print(f"    - per_class_f1_heatmap_comparison.png: Side-by-side condition comparison")
    
    print(f"\n📈 Training Dynamics:")
    print(f"  - loss_curves/: Training and validation loss curves")
    print(f"  - learning_rate_curves/: Learning rate schedules over training")
    print(f"  - combined_training_curves/: Loss and LR side-by-side for each subject")
    
    if not args.no_subject_reports:
        print(f"\n👥 Individual Subject Reports:")
        print(f"  Location: {args.output_dir}/individual_subjects/")
        print(f"  Each subject folder contains:")
        print(f"    - model_comparison.png: Overall and transition performance")
        print(f"    - per_class_f1_heatmap.png: Per-class F1 scores")
        print(f"    - metrics_summary.csv: Numerical results")
        print(f"    - by_horizon/: Horizon-specific plots")
        print(f"  Total subjects analyzed: {len(analyzer.subjects)}")
    
    print(f"\n📝 CSV outputs:")
    print(f"  - overall_model_comparison_all_horizons.csv: Detailed metrics table")
    print(f"  - subject_variance_report.csv: Full variance decomposition")
    print(f"  - subject_variance_compact.csv: Compact by condition")
    print(f"  - subject_variance_paper.csv: Paper-ready table with all conditions")
    print(f"    Format: mean ± σ_subjects (σ_horizons)")
    print(f"  - paper_table_per_horizon.csv: Per-horizon table for paper (0.0s, 0.5s, 1.0s)")
    print(f"    Format: mean ± σ_subjects (pure inter-subject std, no horizon variance)")
    print(f"  - per_class_f1_scores.csv: Per-class metrics table")
    print(f"  - Per-horizon plots and CSVs for detailed analysis")
    print(f"  - Loss curves showing training dynamics")
    print(f"  - Learning rate schedules for each model")


if __name__ == '__main__':
    main()