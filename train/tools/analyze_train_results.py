#!/usr/bin/env python3
"""
analyze_train_results.py - Train-only results analyzer with visualization
Extracts metrics directly from best_model_metrics*.csv files for each horizon
Creates comparison visualizations across different models

Usage:
python3 analyze_train_results.py \
    --base-dir outputs/revalexo \
    --output-dir analysis_results_train
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

class TrainOnlyAnalyzer:
    """Analyzer for train-only results (no cross-validation)."""

    def __init__(self, base_dir: str, output_dir: str, filter_pattern: str = None,
                 create_subject_reports: bool = True):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filter_pattern = filter_pattern
        self.create_subject_reports = create_subject_reports

        # Storage for metrics
        self.metrics_data = defaultdict(lambda: defaultdict(dict))
        self.per_class_metrics = defaultdict(lambda: defaultdict(dict))
        self.loss_data = defaultdict(dict)
        self.lr_data = defaultdict(dict)

        # Per-subject metrics storage
        self.per_subject_metrics = defaultdict(lambda: defaultdict(dict))
        self.subjects = set()

        # Model names and horizons
        self.model_names = []
        self.horizons = []
        self.classes = set()
        
    def scan_and_extract_metrics(self):
        """Scan directory structure and extract metrics from CSV files."""
        print("Scanning for model results...")

        # Find all model directories
        model_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        # Apply filter if specified
        if self.filter_pattern:
            import fnmatch
            model_dirs = [d for d in model_dirs if fnmatch.fnmatch(d.name, self.filter_pattern)]
            print(f"Filtered to {len(model_dirs)} models matching '{self.filter_pattern}'")
        
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
            
            self.model_names.append(model_name)
            
            # Extract metrics from each horizon directory
            horizon_dirs = sorted([d for d in latest_timestamp.iterdir() 
                                 if d.is_dir() and d.name.startswith('horizon_')])
            
            print(f"  Found {len(horizon_dirs)} horizons")
            
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
                            self.metrics_data[model_name][horizon][condition] = metrics

                            # Store per-class metrics (only from 'overall' condition
                            # to avoid overwriting with steady/transition data)
                            if condition == 'overall' and 'per_class_f1' in metrics:
                                for class_name, f1_score in metrics['per_class_f1'].items():
                                    self.per_class_metrics[model_name][horizon][class_name] = f1_score
                                    self.classes.add(class_name)
            
            # Extract loss curves and learning rates from history.log
            history_log = latest_timestamp / 'history.log'
            if history_log.exists():
                self.extract_training_curves(history_log, model_name)

            # Extract per-subject metrics if available
            # First check for per_subject_results folder (new structure from train.py)
            per_subject_dir = latest_timestamp / 'per_subject_results'
            if per_subject_dir.exists():
                self.extract_per_subject_metrics_from_summary(per_subject_dir, model_name)
            else:
                # Fallback: Look for subject_* directories directly (LOSO-like structure)
                subject_dirs = [d for d in latest_timestamp.iterdir()
                               if d.is_dir() and d.name.startswith('subject_')]
                if subject_dirs:
                    self.extract_per_subject_metrics(latest_timestamp, model_name)

        # Sort horizons for consistent ordering
        self.horizons = sorted(self.horizons, key=lambda x: float(x.replace('s', '')))
        print(f"\nFound horizons: {self.horizons}")
        print(f"Found models: {self.model_names}")
        if self.subjects:
            print(f"Found subjects: {sorted(self.subjects)}")
        
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
            
            # Get weighted avg
            weighted_row = df[df['class'] == 'weighted avg']
            if not weighted_row.empty:
                metrics['weighted_f1'] = float(weighted_row['f1-score'].iloc[0])
                metrics['weighted_precision'] = float(weighted_row['precision'].iloc[0])
                metrics['weighted_recall'] = float(weighted_row['recall'].iloc[0])
            
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
    
    def extract_training_curves(self, history_log: Path, model_name: str):
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
                if 'Val: loss=' in line:
                    match = re.search(r'loss=(\d+\.\d+)', line)
                    if match:
                        val_losses.append(float(match.group(1)))

                # Look for learning rate
                if 'lr=' in line:
                    match = re.search(r'lr=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                    if match:
                        learning_rates.append(float(match.group(1)))
            
            self.loss_data[model_name] = {
                'train': train_losses,
                'val': val_losses
            }

            if learning_rates:
                self.lr_data[model_name] = learning_rates

        except Exception as e:
            print(f"    Error reading history log: {e}")

    def extract_per_subject_metrics(self, timestamp_dir: Path, model_name: str):
        """
        Extract per-subject metrics from LOSO-like structure.

        Expected structure:
            <timestamp_dir>/
            ├── subject_Subject01/
            │   ├── horizon_0.0s/
            │   │   ├── best_model_metrics.csv
            │   │   ├── best_model_metrics_steady.csv
            │   │   └── best_model_metrics_transition.csv
            │   └── horizon_0.1s/
            │       └── ...
            └── subject_Subject02/
                └── ...
        """
        try:
            # Look for subject directories
            subject_dirs = [d for d in timestamp_dir.iterdir()
                           if d.is_dir() and d.name.startswith('subject_')]

            if not subject_dirs:
                return

            print(f"  Found {len(subject_dirs)} subject directories")

            for subject_dir in subject_dirs:
                # Extract subject ID from directory name
                subject_id = subject_dir.name.replace('subject_', '')
                self.subjects.add(subject_id)

                subject_metrics = {'horizons': {}}

                # Find horizon directories
                horizon_dirs = [d for d in subject_dir.iterdir()
                               if d.is_dir() and d.name.startswith('horizon_')]

                for horizon_dir in horizon_dirs:
                    horizon_name = horizon_dir.name.replace('horizon_', '')
                    horizon_metrics = {}

                    # Read overall metrics
                    metrics_file = horizon_dir / 'best_model_metrics.csv'
                    if metrics_file.exists():
                        metrics = self._extract_metrics_from_classification_report(metrics_file)
                        horizon_metrics['overall'] = metrics

                    # Read steady metrics
                    steady_file = horizon_dir / 'best_model_metrics_steady.csv'
                    if steady_file.exists():
                        metrics = self._extract_metrics_from_classification_report(steady_file)
                        horizon_metrics['steady'] = metrics

                    # Read transition metrics
                    transition_file = horizon_dir / 'best_model_metrics_transition.csv'
                    if transition_file.exists():
                        metrics = self._extract_metrics_from_classification_report(transition_file)
                        horizon_metrics['transition'] = metrics

                    if horizon_metrics:
                        subject_metrics['horizons'][horizon_name] = horizon_metrics

                        # Also store flattened metrics for backward compatibility
                        for condition, cond_metrics in horizon_metrics.items():
                            if 'accuracy' in cond_metrics:
                                key = f"accuracy_{horizon_name}" if condition == 'overall' else f"{condition}_accuracy_{horizon_name}"
                                subject_metrics[key] = cond_metrics['accuracy']
                            if 'macro_f1' in cond_metrics:
                                key = f"macro_f1_{horizon_name}" if condition == 'overall' else f"{condition}_macro_f1_{horizon_name}"
                                subject_metrics[key] = cond_metrics['macro_f1']

                self.per_subject_metrics[model_name][subject_id] = subject_metrics

        except Exception as e:
            print(f"    Error extracting per-subject metrics: {e}")
            import traceback
            traceback.print_exc()

    def extract_per_subject_metrics_from_summary(self, per_subject_dir: Path, model_name: str):
        """
        Extract per-subject metrics from the new per_subject_results structure.

        Expected structure:
            per_subject_results/
            ├── test_subject_results.csv (or best_val_subject_results.csv)
            ├── test_subject_results_horizon_0.0s.csv
            ├── subject_Subject01/
            │   └── horizon_0.0s/
            │       ├── best_model_metrics.csv
            │       └── ...
            └── ...
        """
        try:
            # First, try to read the summary CSV files
            summary_files = list(per_subject_dir.glob('*_subject_results.csv'))
            if not summary_files:
                print(f"  No summary CSV files found in {per_subject_dir}")
                return

            # Use the main summary file (without horizon suffix)
            main_summary = None
            for f in summary_files:
                if 'horizon' not in f.name:
                    main_summary = f
                    break

            if main_summary is None:
                main_summary = summary_files[0]

            print(f"  Reading per-subject summary from: {main_summary.name}")

            df = pd.read_csv(main_summary)

            # Extract subject metrics (skip MEAN and STD rows)
            for _, row in df.iterrows():
                subject_id = str(row['subject'])
                if subject_id in ['MEAN', 'STD']:
                    continue

                self.subjects.add(subject_id)
                subject_metrics = {'horizons': {}}

                # Extract all numeric columns as metrics
                for col in df.columns:
                    if col == 'subject':
                        continue
                    try:
                        value = float(row[col])
                        subject_metrics[col] = value

                        # Also extract horizon-specific metrics
                        # Parse column names like "accuracy_0.0s", "macro_f1_0.1s"
                        if '_' in col:
                            parts = col.rsplit('_', 1)
                            if len(parts) == 2 and parts[1].endswith('s'):
                                metric_name = parts[0]
                                horizon_name = parts[1]
                                if horizon_name not in subject_metrics['horizons']:
                                    subject_metrics['horizons'][horizon_name] = {}
                                subject_metrics['horizons'][horizon_name][metric_name] = value
                    except (ValueError, TypeError):
                        pass

                self.per_subject_metrics[model_name][subject_id] = subject_metrics

            # Also extract detailed per-subject metrics from individual folders
            subject_dirs = [d for d in per_subject_dir.iterdir()
                           if d.is_dir() and d.name.startswith('subject_')]

            for subject_dir in subject_dirs:
                subject_id = subject_dir.name.replace('subject_', '')
                if subject_id not in self.per_subject_metrics[model_name]:
                    self.per_subject_metrics[model_name][subject_id] = {'horizons': {}}

                # Read horizon-specific detailed metrics
                horizon_dirs = [d for d in subject_dir.iterdir()
                               if d.is_dir() and d.name.startswith('horizon_')]

                for horizon_dir in horizon_dirs:
                    horizon_name = horizon_dir.name.replace('horizon_', '')

                    # Read metrics files for each condition
                    for condition, filename in [('overall', 'best_model_metrics.csv'),
                                                ('steady', 'best_model_metrics_steady.csv'),
                                                ('transition', 'best_model_metrics_transition.csv')]:
                        metrics_file = horizon_dir / filename
                        if metrics_file.exists():
                            metrics = self._extract_metrics_from_classification_report(metrics_file)
                            if metrics:
                                if horizon_name not in self.per_subject_metrics[model_name][subject_id]['horizons']:
                                    self.per_subject_metrics[model_name][subject_id]['horizons'][horizon_name] = {}
                                self.per_subject_metrics[model_name][subject_id]['horizons'][horizon_name][condition] = metrics

            print(f"  Found {len(self.per_subject_metrics[model_name])} subjects for {model_name}")

        except Exception as e:
            print(f"    Error extracting per-subject metrics from summary: {e}")
            import traceback
            traceback.print_exc()

    def _extract_metrics_from_classification_report(self, csv_path: Path) -> Dict:
        """Extract key metrics from a classification report CSV."""
        try:
            df = pd.read_csv(csv_path)
            metrics = {}

            # Get accuracy
            accuracy_row = df[df['class'] == 'accuracy']
            if not accuracy_row.empty:
                metrics['accuracy'] = float(accuracy_row['f1-score'].iloc[0])

            # Get macro avg
            macro_row = df[df['class'] == 'macro avg']
            if not macro_row.empty:
                metrics['macro_f1'] = float(macro_row['f1-score'].iloc[0])
                metrics['macro_precision'] = float(macro_row['precision'].iloc[0])
                metrics['macro_recall'] = float(macro_row['recall'].iloc[0])

            # Get weighted avg
            weighted_row = df[df['class'] == 'weighted avg']
            if not weighted_row.empty:
                metrics['weighted_f1'] = float(weighted_row['f1-score'].iloc[0])

            return metrics
        except Exception as e:
            print(f"    Warning: Error reading {csv_path}: {e}")
            return {}

    def create_horizon_plots(self):
        """Create plots showing metrics across horizons for each model."""
        print("\nCreating horizon comparison plots...")
        
        conditions = ['overall', 'steady', 'transition']
        
        for condition in conditions:
            # Skip if no data for this condition
            has_data = any(
                condition in self.metrics_data[model][horizon]
                for model in self.model_names
                for horizon in self.horizons
            )
            if not has_data:
                continue
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            for model_name in sorted(self.model_names):
                accuracies = []
                macro_f1s = []
                horizon_labels = []
                
                for horizon in self.horizons:
                    if condition in self.metrics_data[model_name][horizon]:
                        metrics = self.metrics_data[model_name][horizon][condition]
                        if 'accuracy' in metrics:
                            accuracies.append(metrics['accuracy'])
                        if 'macro_f1' in metrics:
                            macro_f1s.append(metrics['macro_f1'])
                        horizon_labels.append(horizon)
                
                model_display = model_name.replace('_', ' ').title()
                
                if accuracies:
                    ax1.plot(range(len(accuracies)), accuracies, marker='o', 
                            label=model_display, linewidth=2, markersize=8)
                
                if macro_f1s:
                    ax2.plot(range(len(macro_f1s)), macro_f1s, marker='s', 
                            label=model_display, linewidth=2, markersize=8)
            
            # Accuracy plot
            ax1.set_xlabel('Horizon', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax1.set_title(f'{condition.title()} - Accuracy by Horizon', 
                         fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(self.horizons)))
            ax1.set_xticklabels(self.horizons, rotation=45)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # Macro F1 plot
            ax2.set_xlabel('Horizon', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
            ax2.set_title(f'{condition.title()} - Macro F1 by Horizon', 
                         fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(self.horizons)))
            ax2.set_xticklabels(self.horizons, rotation=45)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            output_path = self.output_dir / f'horizon_comparison_{condition}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: horizon_comparison_{condition}.png")
    
    def create_individual_horizon_plots(self):
        """Create separate plots for each individual horizon."""
        print("\nCreating individual horizon plots...")
        
        horizon_dir = self.output_dir / 'by_horizon'
        horizon_dir.mkdir(exist_ok=True)
        
        for horizon in self.horizons:
            # Create a figure for this horizon
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Performance at Horizon {horizon}', fontsize=16, fontweight='bold')
            
            conditions = ['overall', 'steady', 'transition']
            metrics = ['accuracy', 'macro_f1']
            
            for col, condition in enumerate(conditions):
                for row, metric in enumerate(metrics):
                    ax = axes[row, col]
                    
                    # Collect data for this horizon and condition
                    models = []
                    values = []
                    
                    for model_name in sorted(self.model_names):
                        if condition in self.metrics_data[model_name][horizon]:
                            model_metrics = self.metrics_data[model_name][horizon][condition]
                            if metric in model_metrics:
                                models.append(model_name.replace('_', ' ').title())
                                values.append(model_metrics[metric])
                    
                    if values:
                        # Create bar plot
                        x_pos = np.arange(len(models))
                        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
                        bars = ax.bar(x_pos, values, color=colors)
                        
                        # Add value labels
                        for i, (bar, val) in enumerate(zip(bars, values)):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                        
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
                        ax.set_ylim([0, 1])
                        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
                        ax.set_title(f'{condition.title()}', fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            output_path = horizon_dir / f'horizon_{horizon.replace(".", "_")}_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved individual horizon plots to: {horizon_dir}")
    
    def create_overall_comparison(self):
        """Create overall comparison averaged across all horizons."""
        print("\nCreating overall model comparison...")
        
        # Calculate average metrics across all horizons
        model_metrics = defaultdict(lambda: defaultdict(list))
        
        for model_name in self.model_names:
            for horizon in self.horizons:
                for condition in ['overall', 'transition']:
                    if condition in self.metrics_data[model_name][horizon]:
                        metrics = self.metrics_data[model_name][horizon][condition]
                        if 'accuracy' in metrics:
                            model_metrics[model_name][f'{condition}_accuracy'].append(metrics['accuracy'])
                        if 'macro_f1' in metrics:
                            model_metrics[model_name][f'{condition}_macro_f1'].append(metrics['macro_f1'])
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        models = sorted(self.model_names)
        model_labels = [m.replace('_', ' ').title() for m in models]
        x_pos = np.arange(len(models))
        width = 0.35
        
        # Overall Accuracy
        overall_acc = [np.mean(model_metrics[m]['overall_accuracy']) if model_metrics[m]['overall_accuracy'] else 0 
                      for m in models]
        ax1.bar(x_pos, overall_acc, width, label='Overall', color='steelblue')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Average Accuracy (All Horizons)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_labels, rotation=45, ha='right')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(overall_acc):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Overall Macro F1
        overall_f1 = [np.mean(model_metrics[m]['overall_macro_f1']) if model_metrics[m]['overall_macro_f1'] else 0 
                     for m in models]
        ax2.bar(x_pos, overall_f1, width, label='Overall', color='coral')
        ax2.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
        ax2.set_title('Average Macro F1 (All Horizons)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_labels, rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(overall_f1):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Transition Accuracy
        trans_acc = [np.mean(model_metrics[m]['transition_accuracy']) if model_metrics[m]['transition_accuracy'] else 0 
                    for m in models]
        ax3.bar(x_pos, trans_acc, width, label='Transition', color='lightseagreen')
        ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Average Transition Accuracy (All Horizons)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_labels, rotation=45, ha='right')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(trans_acc):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Transition Macro F1
        trans_f1 = [np.mean(model_metrics[m]['transition_macro_f1']) if model_metrics[m]['transition_macro_f1'] else 0 
                   for m in models]
        ax4.bar(x_pos, trans_f1, width, label='Transition', color='mediumpurple')
        ax4.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
        ax4.set_title('Average Transition Macro F1 (All Horizons)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_labels, rotation=45, ha='right')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(trans_f1):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'overall_model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: overall_model_comparison.png")
        
        # Save CSV summary
        summary_data = {
            'Model': model_labels,
            'Overall_Accuracy': overall_acc,
            'Overall_Macro_F1': overall_f1,
            'Transition_Accuracy': trans_acc,
            'Transition_Macro_F1': trans_f1
        }
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'overall_model_comparison.csv'
        summary_df.to_csv(summary_path, index=False)
        print("  Saved: overall_model_comparison.csv")
    
    def create_per_class_visualization(self):
        """Create visualization of per-class F1 scores."""
        print("\nCreating per-class F1 visualization...")
        
        if not self.classes:
            print("  No per-class data available")
            return
        
        # Average per-class F1 across all horizons
        class_data = defaultdict(lambda: defaultdict(list))
        
        for model_name in self.model_names:
            for horizon in self.horizons:
                for class_name in self.classes:
                    if class_name in self.per_class_metrics[model_name][horizon]:
                        f1_score = self.per_class_metrics[model_name][horizon][class_name]
                        class_data[model_name][class_name].append(f1_score)
        
        # Calculate averages
        avg_class_data = {}
        for model_name in self.model_names:
            avg_class_data[model_name] = {}
            for class_name in self.classes:
                if class_data[model_name][class_name]:
                    avg_class_data[model_name][class_name] = np.mean(class_data[model_name][class_name])
                else:
                    avg_class_data[model_name][class_name] = 0
        
        # Create grouped bar plot
        classes_sorted = sorted(self.classes)
        models_sorted = sorted(self.model_names)
        
        fig, ax = plt.subplots(figsize=(max(16, len(classes_sorted) * 1.2), 8))
        
        x = np.arange(len(classes_sorted))
        width = 0.8 / len(models_sorted)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_sorted)))
        
        for i, model_name in enumerate(models_sorted):
            f1_scores = [avg_class_data[model_name][cls] for cls in classes_sorted]
            offset = (i - len(models_sorted)/2 + 0.5) * width
            model_display = model_name.replace('_', ' ').title()
            ax.bar(x + offset, f1_scores, width, label=model_display, color=colors[i])
        
        ax.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class F1 Scores (Averaged Across All Horizons)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes_sorted, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        output_path = self.output_dir / 'per_class_f1_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: per_class_f1_comparison.png")
    
    def create_per_class_heatmap(self):
        """Create heatmap of per-class F1 scores (averaged across horizons)."""
        print("\nCreating per-class F1 heatmap (averaged across horizons)...")
        
        if not self.classes:
            print("  No per-class data available")
            return
        
        # Prepare data for heatmap (averaged across horizons)
        classes_sorted = sorted(self.classes)
        models_sorted = sorted(self.model_names)
        
        heatmap_data = []
        for model_name in models_sorted:
            row = []
            for class_name in classes_sorted:
                f1_scores = []
                for horizon in self.horizons:
                    if class_name in self.per_class_metrics[model_name][horizon]:
                        f1_scores.append(self.per_class_metrics[model_name][horizon][class_name])
                row.append(np.mean(f1_scores) if f1_scores else 0)
            heatmap_data.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(14, len(classes_sorted) * 0.8), 
                                       max(8, len(models_sorted) * 0.6)))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(classes_sorted)))
        ax.set_yticks(np.arange(len(models_sorted)))
        ax.set_xticklabels(classes_sorted, rotation=45, ha='right')
        ax.set_yticklabels([m.replace('_', ' ').title() for m in models_sorted])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1-Score', rotation=270, labelpad=20, fontweight='bold')
        
        # Add text annotations
        for i in range(len(models_sorted)):
            for j in range(len(classes_sorted)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Per-Class F1 Scores Heatmap (Averaged Across All Horizons)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'per_class_f1_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: per_class_f1_heatmap.png")
    
    def create_per_class_heatmaps_by_horizon(self):
        """Create separate heatmaps for each horizon (overall condition)."""
        print("\nCreating per-horizon per-class F1 heatmaps (overall)...")
        
        if not self.classes:
            print("  No per-class data available")
            return
        
        heatmap_dir = self.output_dir / 'per_class_heatmaps_by_horizon'
        heatmap_dir.mkdir(exist_ok=True)
        
        classes_sorted = sorted(self.classes)
        models_sorted = sorted(self.model_names)
        
        for horizon in self.horizons:
            # Prepare data for this horizon
            heatmap_data = []
            for model_name in models_sorted:
                row = []
                for class_name in classes_sorted:
                    if class_name in self.per_class_metrics[model_name][horizon]:
                        row.append(self.per_class_metrics[model_name][horizon][class_name])
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(max(14, len(classes_sorted) * 0.8), 
                                           max(8, len(models_sorted) * 0.6)))
            
            im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks
            ax.set_xticks(np.arange(len(classes_sorted)))
            ax.set_yticks(np.arange(len(models_sorted)))
            ax.set_xticklabels(classes_sorted, rotation=45, ha='right')
            ax.set_yticklabels([m.replace('_', ' ').title() for m in models_sorted])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('F1-Score', rotation=270, labelpad=20, fontweight='bold')
            
            # Add text annotations
            for i in range(len(models_sorted)):
                for j in range(len(classes_sorted)):
                    text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(f'Per-Class F1 Scores - Horizon {horizon} (Overall)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            output_path = heatmap_dir / f'heatmap_overall_horizon_{horizon.replace(".", "_")}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved per-horizon heatmaps to: {heatmap_dir}")
    
    def create_per_class_heatmaps_transition_by_horizon(self):
        """Create separate heatmaps for each horizon (transition condition)."""
        print("\nCreating per-horizon per-class F1 heatmaps (transition)...")
        
        if not self.classes:
            print("  No per-class data available")
            return
        
        heatmap_dir = self.output_dir / 'per_class_heatmaps_by_horizon_transition'
        heatmap_dir.mkdir(exist_ok=True)
        
        classes_sorted = sorted(self.classes)
        models_sorted = sorted(self.model_names)
        
        # First, we need to extract per-class metrics from transition CSVs
        # Re-scan for transition-specific per-class data
        transition_per_class = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        for model_name in self.model_names:
            model_dir = self.base_dir / model_name
            timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if not timestamp_dirs:
                continue
            latest_timestamp = max(timestamp_dirs, key=lambda x: x.name)
            
            for horizon in self.horizons:
                horizon_dir = latest_timestamp / f'horizon_{horizon}'
                if horizon_dir.exists():
                    csv_path = horizon_dir / 'best_model_metrics_transition.csv'
                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path)
                            class_rows = df[~df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])]
                            for _, row in class_rows.iterrows():
                                class_name = row['class']
                                f1_score = float(row['f1-score'])
                                transition_per_class[model_name][horizon][class_name] = f1_score
                        except Exception as e:
                            pass
        
        for horizon in self.horizons:
            # Prepare data for this horizon
            heatmap_data = []
            has_data = False
            for model_name in models_sorted:
                row = []
                for class_name in classes_sorted:
                    if class_name in transition_per_class[model_name][horizon]:
                        val = transition_per_class[model_name][horizon][class_name]
                        row.append(val)
                        has_data = True
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            if not has_data:
                continue
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(max(14, len(classes_sorted) * 0.8), 
                                           max(8, len(models_sorted) * 0.6)))
            
            im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks
            ax.set_xticks(np.arange(len(classes_sorted)))
            ax.set_yticks(np.arange(len(models_sorted)))
            ax.set_xticklabels(classes_sorted, rotation=45, ha='right')
            ax.set_yticklabels([m.replace('_', ' ').title() for m in models_sorted])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('F1-Score', rotation=270, labelpad=20, fontweight='bold')
            
            # Add text annotations
            for i in range(len(models_sorted)):
                for j in range(len(classes_sorted)):
                    text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(f'Per-Class F1 Scores - Horizon {horizon} (Transition)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            output_path = heatmap_dir / f'heatmap_transition_horizon_{horizon.replace(".", "_")}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved per-horizon transition heatmaps to: {heatmap_dir}")
    
    def create_comparison_heatmap_grid(self):
        """Create a grid comparing overall vs transition per-class F1 (averaged across horizons)."""
        print("\nCreating comparison heatmap grid...")
        
        if not self.classes:
            print("  No per-class data available")
            return
        
        classes_sorted = sorted(self.classes)
        models_sorted = sorted(self.model_names)
        
        # Prepare overall data (averaged across horizons)
        overall_data = []
        for model_name in models_sorted:
            row = []
            for class_name in classes_sorted:
                f1_scores = []
                for horizon in self.horizons:
                    if class_name in self.per_class_metrics[model_name][horizon]:
                        f1_scores.append(self.per_class_metrics[model_name][horizon][class_name])
                row.append(np.mean(f1_scores) if f1_scores else 0)
            overall_data.append(row)
        
        # Prepare transition data
        transition_data = []
        transition_per_class = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        for model_name in self.model_names:
            model_dir = self.base_dir / model_name
            timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if not timestamp_dirs:
                continue
            latest_timestamp = max(timestamp_dirs, key=lambda x: x.name)
            
            for horizon in self.horizons:
                horizon_dir = latest_timestamp / f'horizon_{horizon}'
                if horizon_dir.exists():
                    csv_path = horizon_dir / 'best_model_metrics_transition.csv'
                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path)
                            class_rows = df[~df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])]
                            for _, row in class_rows.iterrows():
                                class_name = row['class']
                                f1_score = float(row['f1-score'])
                                if class_name not in transition_per_class[model_name][horizon]:
                                    transition_per_class[model_name][horizon][class_name] = f1_score
                        except Exception as e:
                            pass
        
        for model_name in models_sorted:
            row = []
            for class_name in classes_sorted:
                f1_scores = []
                for horizon in self.horizons:
                    if class_name in transition_per_class[model_name][horizon]:
                        f1_scores.append(transition_per_class[model_name][horizon][class_name])
                row.append(np.mean(f1_scores) if f1_scores else 0)
            transition_data.append(row)
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(24, len(classes_sorted) * 1.4), 
                                                       max(10, len(models_sorted) * 0.7)))
        
        # Overall heatmap
        im1 = ax1.imshow(overall_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(np.arange(len(classes_sorted)))
        ax1.set_yticks(np.arange(len(models_sorted)))
        ax1.set_xticklabels(classes_sorted, rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels([m.replace('_', ' ').title() for m in models_sorted], fontsize=10)
        
        for i in range(len(models_sorted)):
            for j in range(len(classes_sorted)):
                ax1.text(j, i, f'{overall_data[i][j]:.2f}',
                        ha="center", va="center", color="black", fontsize=7)
        
        ax1.set_title('Overall Condition', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Activity Class', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=11, fontweight='bold')
        
        # Transition heatmap
        im2 = ax2.imshow(transition_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(np.arange(len(classes_sorted)))
        ax2.set_yticks(np.arange(len(models_sorted)))
        ax2.set_xticklabels(classes_sorted, rotation=45, ha='right', fontsize=9)
        ax2.set_yticklabels([m.replace('_', ' ').title() for m in models_sorted], fontsize=10)
        
        for i in range(len(models_sorted)):
            for j in range(len(classes_sorted)):
                ax2.text(j, i, f'{transition_data[i][j]:.2f}',
                        ha="center", va="center", color="black", fontsize=7)
        
        ax2.set_title('Transition Condition', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Activity Class', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Model', fontsize=11, fontweight='bold')
        
        # Add colorbars
        fig.colorbar(im1, ax=ax1, label='F1-Score')
        fig.colorbar(im2, ax=ax2, label='F1-Score')
        
        fig.suptitle('Per-Class F1 Comparison: Overall vs Transition (Averaged Across Horizons)', 
                    fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'per_class_f1_heatmap_comparison_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: per_class_f1_heatmap_comparison_grid.png")
    
    def create_loss_curves(self):
        """Create training and validation loss curves with overfitting analysis."""
        print("\nCreating loss curves...")
        
        if not self.loss_data:
            print("  No loss data available")
            return
        
        loss_dir = self.output_dir / 'loss_curves'
        loss_dir.mkdir(exist_ok=True)
        
        for model_name, losses in self.loss_data.items():
            if not losses['train'] and not losses['val']:
                continue
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            train_losses = losses['train']
            val_losses = losses['val']
            
            if train_losses:
                epochs = range(1, len(train_losses) + 1)
                ax.plot(epochs, train_losses, label='Training Loss', 
                       linewidth=2.5, color='steelblue', alpha=0.8)
            
            if val_losses:
                epochs = range(1, len(val_losses) + 1)
                ax.plot(epochs, val_losses, label='Validation Loss', 
                       linewidth=2.5, color='coral', alpha=0.8)
                
                # Find potential overfitting point (when val loss starts increasing)
                if len(val_losses) > 5:
                    # Find minimum validation loss
                    min_val_idx = np.argmin(val_losses)
                    min_val_loss = val_losses[min_val_idx]
                    
                    # Check if there's significant increase after minimum
                    if min_val_idx < len(val_losses) - 3:
                        recent_avg = np.mean(val_losses[min_val_idx+1:])
                        if recent_avg > min_val_loss * 1.05:  # 5% increase threshold
                            ax.axvline(x=min_val_idx + 1, color='red', linestyle='--', 
                                      linewidth=2, alpha=0.6, 
                                      label=f'Potential Overfitting (Epoch {min_val_idx + 1})')
                            ax.plot(min_val_idx + 1, min_val_loss, 'r*', 
                                   markersize=20, label='Best Val Loss')
            
            ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
            model_display = model_name.replace('_', ' ').title()
            ax.set_title(f'Training Curves - {model_display}', 
                        fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add text box with summary stats if both losses available
            if train_losses and val_losses and len(train_losses) > 0 and len(val_losses) > 0:
                final_train = train_losses[-1]
                final_val = val_losses[-1]
                min_val = min(val_losses)
                textstr = f'Final Train Loss: {final_train:.4f}\n'
                textstr += f'Final Val Loss: {final_val:.4f}\n'
                textstr += f'Best Val Loss: {min_val:.4f}\n'
                textstr += f'Gap: {final_val - final_train:.4f}'
                
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            output_path = loss_dir / f'{model_name}_loss.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved loss curves to: {loss_dir}")
    
    def create_learning_rate_curves(self):
        """Create learning rate curves."""
        print("\nCreating learning rate curves...")
        
        if not self.lr_data:
            print("  No learning rate data available")
            return
        
        lr_dir = self.output_dir / 'learning_rate_curves'
        lr_dir.mkdir(exist_ok=True)
        
        for model_name, lr_values in self.lr_data.items():
            if not lr_values:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            epochs = range(1, len(lr_values) + 1)
            ax.plot(epochs, lr_values, linewidth=2, color='purple')
            
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            model_display = model_name.replace('_', ' ').title()
            ax.set_title(f'Learning Rate Schedule - {model_display}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            output_path = lr_dir / f'{model_name}_lr.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved learning rate curves to: {lr_dir}")
    
    def create_csv_reports(self):
        """Create CSV reports for each condition."""
        print("\nCreating CSV reports...")
        
        for condition in ['overall', 'steady', 'transition']:
            # Check if we have data for this condition
            has_data = any(
                condition in self.metrics_data[model][horizon]
                for model in self.model_names
                for horizon in self.horizons
            )
            if not has_data:
                continue
            
            # Accuracy table
            accuracy_data = {}
            macro_f1_data = {}
            
            for model_name in sorted(self.model_names):
                accuracy_row = {}
                macro_f1_row = {}
                
                for horizon in self.horizons:
                    if condition in self.metrics_data[model_name][horizon]:
                        metrics = self.metrics_data[model_name][horizon][condition]
                        
                        if 'accuracy' in metrics:
                            accuracy_row[horizon] = f"{metrics['accuracy']:.4f}"
                        
                        if 'macro_f1' in metrics:
                            macro_f1_row[horizon] = f"{metrics['macro_f1']:.4f}"
                
                if accuracy_row:
                    model_display_name = model_name.replace('_', ' ').title()
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
        """Create a report showing per-class F1 scores."""
        print("\nCreating per-class F1 report...")
        
        if not self.classes:
            print("  No per-class data available")
            return
        
        # Organize data by class (averaged across horizons)
        class_data = defaultdict(dict)
        
        for model_name in sorted(self.model_names):
            for class_name in sorted(self.classes):
                f1_scores = []
                for horizon in self.horizons:
                    if class_name in self.per_class_metrics[model_name][horizon]:
                        f1_scores.append(self.per_class_metrics[model_name][horizon][class_name])
                
                if f1_scores:
                    mean_f1 = np.mean(f1_scores)
                    model_display_name = model_name.replace('_', ' ').title()
                    class_data[class_name][model_display_name] = f"{mean_f1:.4f}"
        
        # Create DataFrame and save
        if class_data:
            class_df = pd.DataFrame(class_data).T
            class_df.index.name = 'Activity Class'
            class_path = self.output_dir / 'per_class_f1_scores.csv'
            class_df.to_csv(class_path)
            print(f"  Saved: per_class_f1_scores.csv")
    
    def create_individual_subject_reports(self):
        """Create individual analysis reports for each subject."""
        if not self.create_subject_reports or not self.subjects:
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

            # Create CSV report for this subject
            self._create_subject_csv_report(subject_id, subject_dir)

        print(f"\n✅ Individual subject reports created in: {subjects_dir}")

    def _create_subject_comparison_plot(self, subject_id: str, subject_dir: Path):
        """Create overall and transition comparison plot for a single subject."""

        # Collect data for this subject across all models
        model_accuracies = []
        model_names_display = []

        for model_name in sorted(self.model_names):
            if subject_id in self.per_subject_metrics[model_name]:
                subject_data = self.per_subject_metrics[model_name][subject_id]

                # Collect accuracies across horizons
                acc_values = []
                for key, value in subject_data.items():
                    if key.startswith('accuracy_') and isinstance(value, (int, float)):
                        acc_values.append(value)

                if acc_values:
                    model_accuracies.append(np.mean(acc_values))
                    model_names_display.append(model_name.replace('_', ' ').title())

        if not model_accuracies:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(model_names_display) * 0.8), 6))

        # Sort by accuracy
        sorted_indices = np.argsort(model_accuracies)
        model_names_display = [model_names_display[i] for i in sorted_indices]
        model_accuracies = [model_accuracies[i] for i in sorted_indices]

        x = np.arange(len(model_names_display))
        bars = ax.bar(x, model_accuracies, color='steelblue', alpha=0.8)

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Average Accuracy', fontsize=11)
        ax.set_title(f'Model Performance - Subject {subject_id}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names_display, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)

        # Add value labels
        for bar, val in zip(bars, model_accuracies):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        output_path = subject_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_subject_csv_report(self, subject_id: str, subject_dir: Path):
        """Create CSV report for a single subject."""

        csv_data = []

        for model_name in sorted(self.model_names):
            if subject_id in self.per_subject_metrics[model_name]:
                subject_data = self.per_subject_metrics[model_name][subject_id]
                row = {'Model': model_name.replace('_', ' ').title()}

                # Add all metrics
                for key, value in subject_data.items():
                    if key != 'detailed_horizons' and isinstance(value, (int, float)):
                        row[key] = value

                csv_data.append(row)

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = subject_dir / 'metrics_summary.csv'
            df.to_csv(csv_path, index=False)

    def create_per_subject_statistics_comparison(self):
        """Create comparison with mean ± std across subjects (like LOSO format)."""
        if not self.subjects or not self.per_subject_metrics:
            print("\nSkipping per-subject statistics (no per-subject data available)")
            return

        print("\nCreating per-subject statistics comparison (mean ± std)...")

        # Collect per-subject metrics for each model
        overall_stats = {}

        for model_name in sorted(self.model_names):
            model_display = model_name.replace('_', ' ')
            overall_stats[model_display] = {}

            if model_name not in self.per_subject_metrics:
                continue

            subjects_data = self.per_subject_metrics[model_name]

            # Collect metrics for each condition across all subjects and horizons
            for condition in ['overall', 'steady', 'transition']:
                all_accuracies = []
                all_f1s = []

                for subject_id, subject_metrics in subjects_data.items():
                    found_from_horizons = False
                    # Get metrics from the horizons dict
                    if 'horizons' in subject_metrics:
                        for horizon_name, horizon_data in subject_metrics['horizons'].items():
                            if condition == 'overall':
                                # Overall metrics are stored without prefix
                                if 'overall' in horizon_data:
                                    cond_data = horizon_data['overall']
                                    if 'accuracy' in cond_data:
                                        all_accuracies.append(cond_data['accuracy'])
                                        found_from_horizons = True
                                    if 'macro_f1' in cond_data:
                                        all_f1s.append(cond_data['macro_f1'])
                                        found_from_horizons = True
                            else:
                                # Steady/transition metrics
                                if condition in horizon_data:
                                    cond_data = horizon_data[condition]
                                    if 'accuracy' in cond_data:
                                        all_accuracies.append(cond_data['accuracy'])
                                        found_from_horizons = True
                                    if 'macro_f1' in cond_data:
                                        all_f1s.append(cond_data['macro_f1'])
                                        found_from_horizons = True

                    # Only try flat keys if horizons dict didn't provide data
                    # (avoid double-counting when both sources are populated)
                    if not found_from_horizons:
                        for key, value in subject_metrics.items():
                            if key == 'horizons':
                                continue
                            if isinstance(value, (int, float)):
                                if condition == 'overall':
                                    if key.startswith('accuracy_') and 'steady' not in key and 'transition' not in key:
                                        all_accuracies.append(value)
                                    elif key.startswith('macro_f1_') and 'steady' not in key and 'transition' not in key:
                                        all_f1s.append(value)
                                elif condition == 'steady':
                                    if key.startswith('steady_accuracy_'):
                                        all_accuracies.append(value)
                                    elif key.startswith('steady_macro_f1_'):
                                        all_f1s.append(value)
                                elif condition == 'transition':
                                    if key.startswith('transition_accuracy_'):
                                        all_accuracies.append(value)
                                    elif key.startswith('transition_macro_f1_'):
                                        all_f1s.append(value)

                if all_accuracies or all_f1s:
                    overall_stats[model_display][condition] = {
                        'acc_mean': np.mean(all_accuracies) if all_accuracies else 0,
                        'acc_std': np.std(all_accuracies) if all_accuracies else 0,
                        'f1_mean': np.mean(all_f1s) if all_f1s else 0,
                        'f1_std': np.std(all_f1s) if all_f1s else 0,
                        'n_samples': len(all_accuracies) if all_accuracies else len(all_f1s)
                    }

        # Create figure with error bars
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # Plot 1: Overall Model Performance
        ax1 = axes[0]
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

        if models:
            # Sort by accuracy
            sorted_indices = np.argsort(acc_means)
            models = [models[i] for i in sorted_indices]
            acc_means = [acc_means[i] for i in sorted_indices]
            acc_stds = [acc_stds[i] for i in sorted_indices]
            f1_means = [f1_means[i] for i in sorted_indices]
            f1_stds = [f1_stds[i] for i in sorted_indices]

            x = np.arange(len(models))
            width = 0.35

            bars1 = ax1.bar(x - width/2, acc_means, width, yerr=acc_stds,
                            label='Accuracy', capsize=5, color='skyblue', alpha=0.8)
            bars2 = ax1.bar(x + width/2, f1_means, width, yerr=f1_stds,
                            label='Macro F1', capsize=5, color='lightcoral', alpha=0.8)

            ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax1.set_title('Overall Performance (Per-Subject Mean ± Std)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 1.15)

            for bar, mean, std in zip(bars1, acc_means, acc_stds):
                ax1.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            for bar, mean, std in zip(bars2, f1_means, f1_stds):
                ax1.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Transition Performance
        ax2 = axes[1]
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

        if trans_models:
            sorted_indices = np.argsort(trans_acc_means)
            trans_models = [trans_models[i] for i in sorted_indices]
            trans_acc_means = [trans_acc_means[i] for i in sorted_indices]
            trans_acc_stds = [trans_acc_stds[i] for i in sorted_indices]
            trans_f1_means = [trans_f1_means[i] for i in sorted_indices]
            trans_f1_stds = [trans_f1_stds[i] for i in sorted_indices]

            x = np.arange(len(trans_models))

            bars3 = ax2.bar(x - width/2, trans_acc_means, width, yerr=trans_acc_stds,
                            label='Accuracy', capsize=5, color='skyblue', alpha=0.8)
            bars4 = ax2.bar(x + width/2, trans_f1_means, width, yerr=trans_f1_stds,
                            label='Macro F1', capsize=5, color='lightcoral', alpha=0.8)

            ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax2.set_title('Transition Performance (Per-Subject Mean ± Std)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(trans_models, rotation=45, ha='right', fontsize=9)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1.15)

            for bar, mean, std in zip(bars3, trans_acc_means, trans_acc_stds):
                ax2.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            for bar, mean, std in zip(bars4, trans_f1_means, trans_f1_stds):
                ax2.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle(f'Model Comparison: Per-Subject Statistics ({len(self.subjects)} subjects)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        output_path = self.output_dir / 'overall_model_comparison_per_subject.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: overall_model_comparison_per_subject.png")

        # Create CSV report with mean ± std format (like LOSO)
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
            df = pd.DataFrame(csv_data)
            # Sort by overall accuracy
            if 'Overall Accuracy' in df.columns:
                df['_sort_key'] = df['Overall Accuracy'].str.extract(r'(\d+\.\d+)').astype(float)
                df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)

            csv_path = self.output_dir / 'overall_model_comparison_per_subject.csv'
            df.to_csv(csv_path, index=False)
            print(f"  Saved: overall_model_comparison_per_subject.csv")

            # Print summary
            print("\n" + "="*80)
            print(f"PER-SUBJECT STATISTICS (Mean ± Std across {len(self.subjects)} subjects)")
            print("="*80)
            for _, row in df.iterrows():
                print(f"\n{row['Model'].upper()}")
                print("-" * 50)
                if 'Overall Accuracy' in row:
                    print(f"  Overall:    Acc = {row['Overall Accuracy']}, F1 = {row['Overall Macro F1']}")
                if 'Steady Accuracy' in row and pd.notna(row.get('Steady Accuracy')):
                    print(f"  Steady:     Acc = {row['Steady Accuracy']}, F1 = {row['Steady Macro F1']}")
                if 'Transition Accuracy' in row and pd.notna(row.get('Transition Accuracy')):
                    print(f"  Transition: Acc = {row['Transition Accuracy']}, F1 = {row['Transition Macro F1']}")

    def create_subject_variance_report(self):
        """
        Create paper-ready CSV with pure inter-subject variance.

        This method properly separates subject variance from horizon variance:
        1. For each subject: compute mean performance across all horizons
        2. Then compute mean and SD across subjects (pure inter-subject variance)

        Output format: mean ± σ_subjects (σ_horizons)
        - σ_subjects: SD across subject-level means (pure inter-subject variability)
        - σ_horizons: SD of individual horizon values after removing subject mean
        """
        if not self.subjects or not self.per_subject_metrics:
            print("\nSkipping subject variance report (no per-subject data available)")
            return

        print("\nCreating subject variance report (pure inter-subject SD)...")
        print(f"  Number of subjects: {len(self.subjects)}")
        print(f"  Number of horizons: {len(self.horizons)}")

        rows = []

        for model_name in sorted(self.model_names):
            if model_name not in self.per_subject_metrics:
                continue

            model_display = model_name.replace('_', ' ')
            subjects_data = self.per_subject_metrics[model_name]

            row = {'Model': model_display}

            for condition in ['overall', 'steady', 'transition']:
                for metric_type in ['accuracy', 'macro_f1']:
                    # Step 1: For each subject, compute mean across horizons
                    subject_means = []
                    all_horizon_values = []  # For computing horizon variance

                    for subject_id in sorted(self.subjects):
                        if subject_id not in subjects_data:
                            continue

                        subject_data = subjects_data[subject_id]
                        horizon_values = []

                        # Extract values from horizons dict
                        if 'horizons' in subject_data:
                            for horizon_name, horizon_data in subject_data['horizons'].items():
                                cond_key = 'overall' if condition == 'overall' else condition
                                if cond_key in horizon_data:
                                    cond_data = horizon_data[cond_key]
                                    if metric_type in cond_data:
                                        horizon_values.append(cond_data[metric_type])
                                        all_horizon_values.append(cond_data[metric_type])

                        # Also try flat keys (backward compatibility)
                        if not horizon_values:
                            for key, value in subject_data.items():
                                if key == 'horizons' or not isinstance(value, (int, float)):
                                    continue
                                if condition == 'overall':
                                    if key.startswith(f'{metric_type}_') and 'steady' not in key and 'transition' not in key:
                                        horizon_values.append(value)
                                        all_horizon_values.append(value)
                                else:
                                    if key.startswith(f'{condition}_{metric_type}_'):
                                        horizon_values.append(value)
                                        all_horizon_values.append(value)

                        if horizon_values:
                            subject_means.append(np.mean(horizon_values))

                    if len(subject_means) >= 2:
                        # Compute mean and SD across subjects (pure inter-subject variance)
                        overall_mean = np.mean(subject_means)
                        sigma_subjects = np.std(subject_means, ddof=0)

                        # Compute horizon variance (residual after removing subject means)
                        # This represents within-subject variability across horizons
                        if len(all_horizon_values) > len(subject_means):
                            sigma_horizons = np.std(all_horizon_values, ddof=0)
                        else:
                            sigma_horizons = 0.0

                        # Format as "mean ± σ_subjects (σ_horizons)"
                        col_prefix = condition.title() if condition != 'overall' else 'Overall'
                        metric_suffix = 'F1' if metric_type == 'macro_f1' else 'Acc'
                        col_name = f'{col_prefix}_{metric_suffix}'

                        row[col_name] = f"{overall_mean:.3f} ± {sigma_subjects:.3f} ({sigma_horizons:.3f})"

            if len(row) > 1:  # Has more than just Model
                rows.append(row)

        if not rows:
            print("  No data to create report")
            return

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by Overall F1 (extract mean value for sorting)
        if 'Overall_F1' in df.columns:
            df['_sort_key'] = df['Overall_F1'].str.extract(r'([\d.]+)').astype(float)
            df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)

        # Reorder columns
        col_order = ['Model', 'Overall_F1', 'Overall_Acc', 'Steady_F1', 'Steady_Acc',
                     'Transition_F1', 'Transition_Acc']
        cols = [c for c in col_order if c in df.columns]
        cols.extend([c for c in df.columns if c not in cols])
        df = df[cols]

        # Save CSV
        output_path = self.output_dir / 'subject_variance_paper.csv'
        df.to_csv(output_path, index=False)
        print(f"  Saved: subject_variance_paper.csv")

        # Print summary
        print(f"\n  Format: mean ± σ_subjects (σ_horizons)")
        print(f"  - σ_subjects: SD across {len(self.subjects)} subject-level means (pure inter-subject variability)")
        print(f"  - σ_horizons: SD of all {len(self.subjects) * len(self.horizons)} horizon values (pooled)")
        print(f"\n  For paper table, use mean ± σ_subjects and ignore (σ_horizons)")

        return df

    def create_paper_table_per_horizon(self, horizons=None):
        """
        Create paper-ready CSV with per-horizon results using pure inter-subject SD.

        For each horizon: collect each subject's metric, compute mean ± SD across subjects.
        """
        if not self.subjects or not self.per_subject_metrics:
            print("\nSkipping per-horizon paper table (no per-subject data available)")
            return

        if horizons is None:
            horizons = ['0.0s', '0.5s', '1.0s']

        print(f"\nCreating per-horizon paper table for horizons: {horizons}")
        print(f"  Subjects: {len(self.subjects)}, Horizons requested: {horizons}")

        rows = []

        for model_name in sorted(self.model_names):
            if model_name not in self.per_subject_metrics:
                continue

            model_display = model_name.replace('_', ' ')
            subjects_data = self.per_subject_metrics[model_name]

            row = {'Model': model_display}

            for horizon in horizons:
                for condition in ['overall', 'transition']:
                    for metric_type in ['accuracy', 'macro_f1']:
                        # Collect this metric for each subject at this specific horizon
                        subject_values = []

                        for subject_id in sorted(self.subjects):
                            if subject_id not in subjects_data:
                                continue

                            subject_data = subjects_data[subject_id]
                            value = None

                            # Try horizons dict first
                            if 'horizons' in subject_data:
                                cond_key = 'overall' if condition == 'overall' else condition
                                if horizon in subject_data['horizons']:
                                    horizon_data = subject_data['horizons'][horizon]
                                    if cond_key in horizon_data:
                                        cond_data = horizon_data[cond_key]
                                        if metric_type in cond_data:
                                            value = cond_data[metric_type]

                            if value is not None:
                                subject_values.append(value)

                        if len(subject_values) >= 2:
                            mean_val = np.mean(subject_values)
                            std_val = np.std(subject_values, ddof=0)

                            cond_prefix = 'Overall' if condition == 'overall' else 'Trans'
                            metric_suffix = 'F1' if metric_type == 'macro_f1' else 'Acc'
                            col_name = f'{cond_prefix}_{metric_suffix}_{horizon}'

                            row[col_name] = f"{mean_val*100:.1f} ± {std_val*100:.1f}"

            if len(row) > 1:
                rows.append(row)

        if not rows:
            print("  No data to create per-horizon table")
            return

        df = pd.DataFrame(rows)

        # Sort by Overall F1 at first horizon (descending)
        sort_col = f'Overall_F1_{horizons[0]}'
        if sort_col in df.columns:
            df['_sort_key'] = df[sort_col].str.extract(r'([\d.]+)').astype(float)
            df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)

        output_path = self.output_dir / 'paper_table_per_horizon.csv'
        df.to_csv(output_path, index=False)
        print(f"  Saved: paper_table_per_horizon.csv")

        return df

    def create_per_subject_comparison(self):
        """Create comparison plots showing all subjects for each model."""
        if not self.subjects or not self.per_subject_metrics:
            return

        print("\nCreating per-subject comparison plots...")

        per_subject_dir = self.output_dir / 'per_subject_comparison'
        per_subject_dir.mkdir(exist_ok=True)

        for model_name in self.model_names:
            if model_name not in self.per_subject_metrics:
                continue

            subjects_data = self.per_subject_metrics[model_name]
            if not subjects_data:
                continue

            # Collect accuracy data for each subject
            subject_ids = sorted(subjects_data.keys())
            accuracies = []

            for subject_id in subject_ids:
                subject_data = subjects_data[subject_id]
                # Average accuracy across horizons
                acc_values = []
                for key, value in subject_data.items():
                    if key.startswith('accuracy_') and isinstance(value, (int, float)):
                        acc_values.append(value)
                accuracies.append(np.mean(acc_values) if acc_values else 0)

            if not accuracies:
                continue

            # Create bar plot
            fig, ax = plt.subplots(figsize=(max(10, len(subject_ids) * 0.6), 6))

            x = np.arange(len(subject_ids))
            bars = ax.bar(x, accuracies, color='steelblue', alpha=0.8)

            # Color bars based on performance
            colors = plt.cm.RdYlGn(np.array(accuracies))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel('Subject', fontsize=11)
            ax.set_ylabel('Average Accuracy', fontsize=11)
            model_display = model_name.replace('_', ' ').title()
            ax.set_title(f'Per-Subject Performance - {model_display}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(subject_ids, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, accuracies)):
                ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            # Add mean line
            mean_acc = np.mean(accuracies)
            ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_acc:.3f}')
            ax.legend(loc='lower right')

            plt.tight_layout()
            output_path = per_subject_dir / f'{model_name}_per_subject.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  Saved per-subject comparison plots to: {per_subject_dir}")

    def create_improvement_analysis(self, baseline_pattern: str = None):
        """
        Create improvement analysis showing delta (gain) over baseline for each method.

        This calculates per-subject improvement and reports mean ± std of deltas,
        which directly shows:
        - Mean improvement: Does the method help on average?
        - Std of improvement: Is the improvement consistent across subjects?

        Args:
            baseline_pattern: Pattern to match baseline model (e.g., "deepconvlstm_acc_gyro").
                             If None, uses the model without "kd" or "contrastive" in name.
        """
        if not self.subjects or len(self.subjects) < 2:
            print("\nSkipping improvement analysis (need at least 2 subjects)")
            return

        print("\nCreating improvement analysis (delta over baseline)...")

        # Find baseline model
        baseline_model = None
        for model_name in sorted(self.model_names):
            model_lower = model_name.lower()
            if baseline_pattern:
                if baseline_pattern.lower() in model_lower:
                    baseline_model = model_name
                    break
            else:
                # Default: model without kd, contrastive, pretrained
                if 'kd' not in model_lower and 'contrastive' not in model_lower and 'pretrained' not in model_lower:
                    baseline_model = model_name
                    break

        if not baseline_model:
            baseline_model = sorted(self.model_names)[0]

        print(f"  Baseline model: {baseline_model}")
        print(f"  Number of subjects: {len(self.subjects)}")

        # Get baseline per-subject metrics
        baseline_subjects = self.per_subject_metrics.get(baseline_model, {})
        if not baseline_subjects:
            print("  No per-subject data for baseline model")
            return

        # Calculate deltas for each model
        improvement_data = {}

        for model_name in sorted(self.model_names):
            if model_name == baseline_model:
                continue

            model_subjects = self.per_subject_metrics.get(model_name, {})
            if not model_subjects:
                continue

            # Calculate per-subject deltas for F1 and accuracy
            deltas_f1 = []
            deltas_acc = []
            subject_details = []

            for subject_id in sorted(self.subjects):
                if subject_id not in model_subjects or subject_id not in baseline_subjects:
                    continue

                model_data = model_subjects[subject_id]
                baseline_data = baseline_subjects[subject_id]

                # Get average F1 and accuracy across horizons for each subject
                model_f1s = []
                model_accs = []
                baseline_f1s = []
                baseline_accs = []

                # Extract from horizons dict
                if 'horizons' in model_data:
                    for horizon_name, horizon_data in model_data['horizons'].items():
                        if 'overall' in horizon_data:
                            if 'macro_f1' in horizon_data['overall']:
                                model_f1s.append(horizon_data['overall']['macro_f1'])
                            if 'accuracy' in horizon_data['overall']:
                                model_accs.append(horizon_data['overall']['accuracy'])

                if 'horizons' in baseline_data:
                    for horizon_name, horizon_data in baseline_data['horizons'].items():
                        if 'overall' in horizon_data:
                            if 'macro_f1' in horizon_data['overall']:
                                baseline_f1s.append(horizon_data['overall']['macro_f1'])
                            if 'accuracy' in horizon_data['overall']:
                                baseline_accs.append(horizon_data['overall']['accuracy'])

                # Only try flat keys if horizons dict didn't provide data
                # (avoid double-counting when both sources are populated)
                if not model_f1s and not model_accs:
                    for key, value in model_data.items():
                        if key == 'horizons' or not isinstance(value, (int, float)):
                            continue
                        if key.startswith('macro_f1_') and 'steady' not in key and 'transition' not in key:
                            model_f1s.append(value)
                        elif key.startswith('accuracy_') and 'steady' not in key and 'transition' not in key:
                            model_accs.append(value)

                if not baseline_f1s and not baseline_accs:
                    for key, value in baseline_data.items():
                        if key == 'horizons' or not isinstance(value, (int, float)):
                            continue
                        if key.startswith('macro_f1_') and 'steady' not in key and 'transition' not in key:
                            baseline_f1s.append(value)
                        elif key.startswith('accuracy_') and 'steady' not in key and 'transition' not in key:
                            baseline_accs.append(value)

                # Calculate delta for this subject
                if model_f1s and baseline_f1s:
                    delta_f1 = np.mean(model_f1s) - np.mean(baseline_f1s)
                    deltas_f1.append(delta_f1)

                if model_accs and baseline_accs:
                    delta_acc = np.mean(model_accs) - np.mean(baseline_accs)
                    deltas_acc.append(delta_acc)
                    subject_details.append({
                        'subject': subject_id,
                        'baseline_acc': np.mean(baseline_accs),
                        'model_acc': np.mean(model_accs),
                        'delta_acc': delta_acc,
                        'baseline_f1': np.mean(baseline_f1s) if baseline_f1s else 0,
                        'model_f1': np.mean(model_f1s) if model_f1s else 0,
                        'delta_f1': np.mean(model_f1s) - np.mean(baseline_f1s) if model_f1s and baseline_f1s else 0
                    })

            if deltas_f1 or deltas_acc:
                improvement_data[model_name] = {
                    'deltas_f1': deltas_f1,
                    'deltas_acc': deltas_acc,
                    'mean_delta_f1': np.mean(deltas_f1) if deltas_f1 else 0,
                    'std_delta_f1': np.std(deltas_f1, ddof=0) if len(deltas_f1) > 1 else 0,
                    'mean_delta_acc': np.mean(deltas_acc) if deltas_acc else 0,
                    'std_delta_acc': np.std(deltas_acc, ddof=0) if len(deltas_acc) > 1 else 0,
                    'n_subjects': len(deltas_f1) if deltas_f1 else len(deltas_acc),
                    'subject_details': subject_details
                }

        if not improvement_data:
            print("  No improvement data calculated")
            return

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        models = []
        mean_deltas_f1 = []
        std_deltas_f1 = []
        mean_deltas_acc = []
        std_deltas_acc = []

        for model_name, data in sorted(improvement_data.items()):
            short_name = model_name.replace(baseline_model + '_', '').replace('_', ' ').title()
            models.append(short_name)
            mean_deltas_f1.append(data['mean_delta_f1'] * 100)  # Convert to percentage
            std_deltas_f1.append(data['std_delta_f1'] * 100)
            mean_deltas_acc.append(data['mean_delta_acc'] * 100)
            std_deltas_acc.append(data['std_delta_acc'] * 100)

        x = np.arange(len(models))
        width = 0.6

        # F1 improvement plot
        colors_f1 = ['green' if v > 0 else 'red' for v in mean_deltas_f1]
        bars1 = ax1.bar(x, mean_deltas_f1, width, yerr=std_deltas_f1, capsize=8,
                        color=colors_f1, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Δ Macro F1 (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Macro F1 Improvement over Baseline\n(N={improvement_data[list(improvement_data.keys())[0]]["n_subjects"]} subjects)',
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (mean, std) in enumerate(zip(mean_deltas_f1, std_deltas_f1)):
            y_pos = mean + std + 0.5 if mean >= 0 else mean - std - 0.5
            va = 'bottom' if mean >= 0 else 'top'
            ax1.text(i, y_pos, f'{mean:+.2f}±{std:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')

        # Accuracy improvement plot
        colors_acc = ['green' if v > 0 else 'red' for v in mean_deltas_acc]
        bars2 = ax2.bar(x, mean_deltas_acc, width, yerr=std_deltas_acc, capsize=8,
                        color=colors_acc, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Δ Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Accuracy Improvement over Baseline\n(N={improvement_data[list(improvement_data.keys())[0]]["n_subjects"]} subjects)',
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        for i, (mean, std) in enumerate(zip(mean_deltas_acc, std_deltas_acc)):
            y_pos = mean + std + 0.5 if mean >= 0 else mean - std - 0.5
            va = 'bottom' if mean >= 0 else 'top'
            ax2.text(i, y_pos, f'{mean:+.2f}±{std:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')

        baseline_display = baseline_model.replace('_', ' ').title()
        plt.suptitle(f'Improvement Analysis: Delta over "{baseline_display}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = self.output_dir / 'improvement_over_baseline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: improvement_over_baseline.png")

        # Create CSV report
        csv_data = []
        for model_name, data in sorted(improvement_data.items()):
            short_name = model_name.replace('_', ' ').title()
            csv_data.append({
                'Method': short_name,
                'Baseline': baseline_model.replace('_', ' ').title(),
                'N Subjects': data['n_subjects'],
                'Δ Macro F1 (Mean)': f"{data['mean_delta_f1']*100:+.2f}%",
                'Δ Macro F1 (Std)': f"{data['std_delta_f1']*100:.2f}%",
                'Δ Macro F1 (Format)': f"{data['mean_delta_f1']*100:+.2f} ± {data['std_delta_f1']*100:.2f}",
                'Δ Accuracy (Mean)': f"{data['mean_delta_acc']*100:+.2f}%",
                'Δ Accuracy (Std)': f"{data['std_delta_acc']*100:.2f}%",
                'Δ Accuracy (Format)': f"{data['mean_delta_acc']*100:+.2f} ± {data['std_delta_acc']*100:.2f}",
            })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / 'improvement_over_baseline.csv'
            df.to_csv(csv_path, index=False)
            print(f"  Saved: improvement_over_baseline.csv")

        # Create per-subject detail CSV
        detail_rows = []
        for model_name, data in sorted(improvement_data.items()):
            short_name = model_name.replace('_', ' ').title()
            for subj in data['subject_details']:
                detail_rows.append({
                    'Method': short_name,
                    'Subject': subj['subject'],
                    'Baseline Acc': f"{subj['baseline_acc']:.4f}",
                    'Method Acc': f"{subj['model_acc']:.4f}",
                    'Δ Acc': f"{subj['delta_acc']*100:+.2f}%",
                    'Baseline F1': f"{subj['baseline_f1']:.4f}",
                    'Method F1': f"{subj['model_f1']:.4f}",
                    'Δ F1': f"{subj['delta_f1']*100:+.2f}%",
                })

        if detail_rows:
            detail_df = pd.DataFrame(detail_rows)
            detail_path = self.output_dir / 'improvement_per_subject_detail.csv'
            detail_df.to_csv(detail_path, index=False)
            print(f"  Saved: improvement_per_subject_detail.csv")

        # Print summary
        print("\n" + "="*80)
        print(f"IMPROVEMENT ANALYSIS: Delta over {baseline_display}")
        print(f"N = {improvement_data[list(improvement_data.keys())[0]]['n_subjects']} subjects")
        print("="*80)
        print(f"\n{'Method':<45} {'Δ Macro F1':>20} {'Δ Accuracy':>20}")
        print("-"*85)
        for model_name, data in sorted(improvement_data.items()):
            short_name = model_name.replace('_', ' ').title()
            f1_str = f"{data['mean_delta_f1']*100:+.2f} ± {data['std_delta_f1']*100:.2f}%"
            acc_str = f"{data['mean_delta_acc']*100:+.2f} ± {data['std_delta_acc']*100:.2f}%"
            print(f"{short_name:<45} {f1_str:>20} {acc_str:>20}")

    def create_subject_heatmap(self):
        """Create heatmap showing all subjects vs all models."""
        if not self.subjects or not self.per_subject_metrics:
            return

        print("\nCreating subject vs model heatmap...")

        subjects_sorted = sorted(self.subjects)
        models_sorted = sorted(self.model_names)

        # Build matrix of accuracies
        heatmap_data = []
        for model_name in models_sorted:
            row = []
            for subject_id in subjects_sorted:
                if subject_id in self.per_subject_metrics[model_name]:
                    subject_data = self.per_subject_metrics[model_name][subject_id]
                    acc_values = []
                    for key, value in subject_data.items():
                        if key.startswith('accuracy_') and isinstance(value, (int, float)):
                            acc_values.append(value)
                    row.append(np.mean(acc_values) if acc_values else 0)
                else:
                    row.append(0)
            heatmap_data.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(subjects_sorted) * 0.8),
                                        max(8, len(models_sorted) * 0.5)))

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(subjects_sorted)))
        ax.set_yticks(np.arange(len(models_sorted)))
        ax.set_xticklabels(subjects_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([m.replace('_', ' ').title() for m in models_sorted], fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Accuracy', rotation=270, labelpad=20, fontweight='bold')

        # Add text annotations
        for i in range(len(models_sorted)):
            for j in range(len(subjects_sorted)):
                value = heatmap_data[i][j]
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}',
                       ha="center", va="center", color=text_color, fontsize=7)

        ax.set_title('Model Performance by Subject (Average Accuracy)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'subject_model_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: subject_model_heatmap.png")

        # Also save as CSV
        df = pd.DataFrame(heatmap_data,
                         index=[m.replace('_', ' ').title() for m in models_sorted],
                         columns=subjects_sorted)
        df.index.name = 'Model'
        csv_path = self.output_dir / 'subject_model_accuracy.csv'
        df.to_csv(csv_path)
        print("  Saved: subject_model_accuracy.csv")

    def print_summary(self):
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("TRAIN-ONLY RESULTS SUMMARY")
        print("="*80)

        for model_name in sorted(self.model_names):
            print(f"\n{model_name.replace('_', ' ').upper()}")
            print("-" * 40)

            for horizon in self.horizons[:3]:  # Show first 3 horizons as example
                print(f"\nHorizon {horizon}:")

                for condition in ['overall', 'steady', 'transition']:
                    if condition in self.metrics_data[model_name][horizon]:
                        metrics = self.metrics_data[model_name][horizon][condition]

                        if 'accuracy' in metrics and 'macro_f1' in metrics:
                            print(f"  {condition.title():12} - Acc: {metrics['accuracy']:.4f}, "
                                  f"F1: {metrics['macro_f1']:.4f}")

        # Print per-subject summary if available
        if self.subjects:
            print("\n" + "="*80)
            print("PER-SUBJECT SUMMARY")
            print("="*80)
            print(f"\nTotal subjects: {len(self.subjects)}")
            print(f"Subjects: {', '.join(sorted(self.subjects))}")


def main():
    parser = argparse.ArgumentParser(description='Train-only results analyzer')
    parser.add_argument('--base-dir', type=str, default='outputs/revalexo',
                       help='Base directory containing train results')
    parser.add_argument('--output-dir', type=str, default='analysis_results_train',
                       help='Output directory for analysis results')
    parser.add_argument('--filter', type=str, default=None,
                       help='Filter model directories by glob pattern (e.g., "deepconvlstm*")')
    parser.add_argument('--no-subject-reports', action='store_true',
                       help='Skip creating individual subject reports')

    args = parser.parse_args()

    # Create analyzer and run analysis
    analyzer = TrainOnlyAnalyzer(
        args.base_dir,
        args.output_dir,
        filter_pattern=args.filter,
        create_subject_reports=not args.no_subject_reports
    )
    
    # Extract metrics
    analyzer.scan_and_extract_metrics()
    
    # Create visualizations and reports
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Horizon plots (combined across horizons)
    analyzer.create_horizon_plots()
    
    # Individual horizon plots (separate for each horizon)
    analyzer.create_individual_horizon_plots()
    
    # Training dynamics
    analyzer.create_loss_curves()
    analyzer.create_learning_rate_curves()
    
    # Overall comparison
    analyzer.create_overall_comparison()
    
    # Per-class analysis
    analyzer.create_per_class_visualization()
    analyzer.create_per_class_heatmap()
    
    # Per-horizon heatmaps
    analyzer.create_per_class_heatmaps_by_horizon()
    analyzer.create_per_class_heatmaps_transition_by_horizon()
    
    # Comparison grid
    analyzer.create_comparison_heatmap_grid()

    # Per-subject analysis (if per-subject metrics available)
    analyzer.create_per_subject_statistics_comparison()  # New: mean ± std format like LOSO
    analyzer.create_subject_variance_report()  # Paper-ready: pure inter-subject SD
    analyzer.create_paper_table_per_horizon()  # Per-horizon with per-subject SD
    analyzer.create_improvement_analysis()  # New: delta over baseline analysis
    analyzer.create_per_subject_comparison()
    analyzer.create_subject_heatmap()
    analyzer.create_individual_subject_reports()

    # CSV reports
    analyzer.create_csv_reports()
    analyzer.create_per_class_report()

    # Print summary
    analyzer.print_summary()
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📂 Results saved to: {args.output_dir}")
    print(f"\n📊 Main Outputs:")
    print(f"  - overall_model_comparison.png: Overall model comparison")
    print(f"  - per_class_f1_comparison.png: Per-class F1 comparison")
    print(f"  - per_class_f1_heatmap.png: Per-class F1 heatmap (averaged)")
    print(f"  - per_class_f1_heatmap_comparison_grid.png: Overall vs Transition comparison")
    print(f"  - horizon_comparison_*.png: Performance across horizons (3 files)")
    
    print(f"\n📈 Training Dynamics:")
    print(f"  - loss_curves/: Training and validation loss curves")
    print(f"    * Shows potential overfitting points")
    print(f"    * Summary statistics in text boxes")
    print(f"  - learning_rate_curves/: Learning rate schedules")
    
    print(f"\n🔥 Per-Horizon Analysis:")
    print(f"  - by_horizon/: Individual plots for each horizon ({len(analyzer.horizons)} horizons)")
    print(f"  - per_class_heatmaps_by_horizon/: Per-class F1 heatmaps (overall)")
    print(f"  - per_class_heatmaps_by_horizon_transition/: Per-class F1 heatmaps (transition)")
    
    print(f"\n📝 CSV Reports:")
    print(f"  - overall_model_comparison.csv: Overall metrics summary")
    print(f"  - per_class_f1_scores.csv: Per-class metrics table")
    print(f"  - *_accuracy_results.csv: Accuracy by condition and horizon")
    print(f"  - *_macro_f1_results.csv: Macro F1 by condition and horizon")

    if analyzer.subjects:
        print(f"\n👥 Per-Subject Analysis:")
        print(f"  - subject_variance_paper.csv: PAPER-READY with pure inter-subject SD")
        print(f"    Format: mean ± σ_subjects (σ_horizons)")
        print(f"  - overall_model_comparison_per_subject.csv: Mean ± Std format (like LOSO)")
        print(f"  - overall_model_comparison_per_subject.png: Bar plot with error bars")
        print(f"  - improvement_over_baseline.csv: Delta (gain) over baseline (mean ± std)")
        print(f"  - improvement_over_baseline.png: Improvement visualization")
        print(f"  - improvement_per_subject_detail.csv: Per-subject delta breakdown")
        print(f"  - subject_model_heatmap.png: Model x Subject performance heatmap")
        print(f"  - subject_model_accuracy.csv: Model x Subject accuracy table")
        print(f"  - per_subject_comparison/: Per-subject performance for each model")
        if not args.no_subject_reports:
            print(f"  - individual_subjects/: Individual reports per subject")
            print(f"    * Each subject folder contains:")
            print(f"      - model_comparison.png: Model comparison for that subject")
            print(f"      - metrics_summary.csv: Detailed metrics")
            print(f"    Total subjects analyzed: {len(analyzer.subjects)}")

    print(f"\n💡 Quick Start:")
    print(f"  1. Check overall_model_comparison.png for high-level comparison")
    print(f"  2. Review loss_curves/ to identify overfitting")
    print(f"  3. Explore by_horizon/ for horizon-specific insights")
    print(f"  4. Dive into per-class heatmaps for detailed analysis")
    if analyzer.subjects:
        print(f"  5. Check subject_model_heatmap.png for per-subject performance")
        print(f"  6. Explore individual_subjects/ for subject-level analysis")


if __name__ == '__main__':
    main()