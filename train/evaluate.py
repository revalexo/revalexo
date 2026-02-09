#!/usr/bin/env python
# evaluate.py - Standalone evaluation script for trained models
#
# Evaluate a trained model checkpoint on the test set defined in the config.
# Optionally override test subjects to evaluate on a different population (can also be directly defined in thw config)
#
# Usage:
#   # Evaluate using test set from config
#   python evaluate.py --config configs/train/experiment.yaml --checkpoint path/to/best_model.pt
#
#   # Evaluate on different subjects (override test set)
#   python evaluate.py --config configs/train/experiment.yaml --checkpoint path/to/best_model.pt \
#       --test-subjects Subject05 Subject06 Subject07

import os
import sys
import argparse
import time
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

import torch
import torch.nn as nn
import yaml

from utils import (
    load_config, handle_scientific_notation, setup_seed, resolve_paths, get_device,
    setup_datasets_and_loaders, build_models, get_main_model,
    setup_training_components, setup_metrics, evaluate
)
from utils.training_utils import MultiHorizonLoss, save_per_subject_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model checkpoint on a test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation using test set from config
  python evaluate.py --config configs/train/experiment.yaml --checkpoint outputs/train/model/best_model.pt

  # Evaluate on different test subjects (cross-population)
  python evaluate.py --config configs/train/healthy_stroke.yaml --checkpoint model.pt \\
      --test-subjects Subject04 Subject13 Subject14

  # Specify output directory
  python evaluate.py --config config.yaml --checkpoint model.pt --output-dir outputs/eval/my_eval
        """
    )

    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')

    # Optional arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results. Default: creates "eval" folder next to checkpoint')
    parser.add_argument('--test-subjects', type=str, nargs='+', default=None,
                        help='Override test subjects (space-separated list)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of workers for data loading')
    parser.add_argument('--eval-name', type=str, default=None,
                        help='Name for this evaluation run (used in output folder name)')

    return parser.parse_args()


def override_config_values(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override configuration values with command line arguments."""
    # Override batch size
    if args.batch_size is not None:
        if 'dataloaders' not in config:
            config['dataloaders'] = {}
        for split in ['train', 'validation', 'test']:
            if split not in config['dataloaders']:
                config['dataloaders'][split] = {}
            config['dataloaders'][split]['batch_size'] = args.batch_size

    # Override num_workers
    if args.num_workers is not None:
        if 'dataloaders' not in config:
            config['dataloaders'] = {}
        for split in ['train', 'validation', 'test']:
            if split not in config['dataloaders']:
                config['dataloaders'][split] = {}
            config['dataloaders'][split]['num_workers'] = args.num_workers

    # Override test subjects if specified
    if args.test_subjects is not None:
        if 'dataset' not in config:
            config['dataset'] = {}
        if 'splits' not in config['dataset']:
            config['dataset']['splits'] = {}
        config['dataset']['splits']['test_subjects'] = args.test_subjects
        # Also set val_subjects to same for consistency (evaluation only uses test)
        config['dataset']['splits']['val_subjects'] = args.test_subjects
        print(f"Overriding test subjects: {args.test_subjects}")

    return config


def setup_output_directory(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """Set up output directory for evaluation results."""
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        # Create eval folder next to checkpoint
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.eval_name:
            folder_name = f"eval_{timestamp}_{args.eval_name}"
        else:
            # Use test subjects in folder name if overridden
            test_subjects = config.get('dataset', {}).get('splits', {}).get('test_subjects', [])
            if test_subjects and len(test_subjects) <= 3:
                subjects_str = "_".join(test_subjects)
            elif test_subjects:
                subjects_str = f"{test_subjects[0]}_to_{test_subjects[-1]}"
            else:
                subjects_str = "test"
            folder_name = f"eval_{timestamp}_{subjects_str}"

        output_dir = os.path.join(checkpoint_dir, folder_name)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_evaluation_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    prediction_horizons: List[float],
    epoch: int = 0
) -> None:
    """Save evaluation metrics in the same format as training."""
    num_horizons = len(prediction_horizons)
    categories = ['overall', 'steady', 'transition']

    for h in range(num_horizons):
        h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
        horizon_time = prediction_horizons[h]
        horizon_str = f"{horizon_time:.1f}s"

        # Create horizon-specific directory
        horizon_dir = os.path.join(output_dir, f"horizon_{horizon_str}")
        os.makedirs(horizon_dir, exist_ok=True)

        for category in categories:
            if category == 'overall':
                category_prefix = ''
                category_suffix = ''
            else:
                category_prefix = f'{category}_'
                category_suffix = f'_{category}'

            # Save classification report CSV
            classreport_key = f'{category_prefix}classificationreport_dataframe{h_suffix}'
            if classreport_key in metrics:
                try:
                    csv_filename = f'metrics_report{category_suffix}_epoch_{epoch}.csv'
                    csv_path = os.path.join(horizon_dir, csv_filename)
                    metrics[classreport_key].to_csv(csv_path, index=False)

                    # Also save as best_model_metrics for compatibility
                    best_csv_path = os.path.join(horizon_dir, f'best_model_metrics{category_suffix}.csv')
                    metrics[classreport_key].to_csv(best_csv_path, index=False)
                    logging.info(f"Saved {category} horizon {horizon_str} metrics: {best_csv_path}")
                except Exception as e:
                    logging.warning(f"Error saving {category} metrics: {e}")

            # Save confusion matrix CSV
            confmat_key = f'{category_prefix}confusionmatrix_dataframe{h_suffix}'
            if confmat_key in metrics:
                try:
                    cm_path = os.path.join(horizon_dir, f'best_model_confusion_matrix{category_suffix}.csv')
                    metrics[confmat_key].to_csv(cm_path, index=False)
                except Exception as e:
                    logging.warning(f"Error saving confusion matrix: {e}")


def save_evaluation_summary(
    metrics: Dict[str, Any],
    output_dir: str,
    prediction_horizons: List[float],
    config: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_info: Dict[str, Any]
) -> None:
    """Save evaluation summary as JSON."""
    num_horizons = len(prediction_horizons)

    # Build results.json (compatible with analyze scripts)
    results = {
        'test_loss': metrics.get('loss', 0.0),
        'evaluation_epoch': checkpoint_info.get('epoch', 0),
        'prediction_horizons': prediction_horizons,
        'checkpoint': args.checkpoint,
        'config': args.config,
        'test_subjects': config.get('dataset', {}).get('splits', {}).get('test_subjects', []),
    }

    if num_horizons > 1:
        for h in range(num_horizons):
            h_suffix = f"_horizon_{h}"
            horizon_time = prediction_horizons[h]
            horizon_str = f"{horizon_time:.1f}s"

            results[f'test_accuracy_horizon_{horizon_str}'] = metrics.get(f'accuracy{h_suffix}', 0.0)
            results[f'test_steady_accuracy_horizon_{horizon_str}'] = metrics.get(f'steady_accuracy{h_suffix}', 0.0)
            results[f'test_transition_accuracy_horizon_{horizon_str}'] = metrics.get(f'transition_accuracy{h_suffix}', 0.0)
    else:
        results.update({
            'test_accuracy': metrics.get('accuracy', 0.0),
            'test_steady_accuracy': metrics.get('steady_accuracy', 0.0),
            'test_transition_accuracy': metrics.get('transition_accuracy', 0.0),
        })

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results: {results_path}")


def load_model_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Load model weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        checkpoint = {'model_state_dict': state_dict}

    model.load_state_dict(state_dict)

    print(f"Loaded checkpoint: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")

    return checkpoint


def print_evaluation_results(metrics: Dict[str, Any], prediction_horizons: List[float]) -> None:
    """Print evaluation results to console."""
    num_horizons = len(prediction_horizons)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nLoss: {metrics.get('loss', 0):.4f}")

    for h in range(num_horizons):
        h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
        horizon_time = prediction_horizons[h]

        print(f"\n--- Horizon {horizon_time:.1f}s ---")

        acc = metrics.get(f'accuracy{h_suffix}', 0)
        macro_f1 = metrics.get(f'classificationreport_macro_f1-score{h_suffix}', 0)
        weighted_f1 = metrics.get(f'classificationreport_weighted_f1-score{h_suffix}', 0)
        print(f"  Overall:    Acc={acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}")

        steady_acc = metrics.get(f'steady_accuracy{h_suffix}')
        if steady_acc is not None:
            steady_f1 = metrics.get(f'steady_classificationreport_macro_f1-score{h_suffix}', 0)
            print(f"  Steady:     Acc={steady_acc:.4f}, Macro F1={steady_f1:.4f}")

        trans_acc = metrics.get(f'transition_accuracy{h_suffix}')
        if trans_acc is not None:
            trans_f1 = metrics.get(f'transition_classificationreport_macro_f1-score{h_suffix}', 0)
            print(f"  Transition: Acc={trans_acc:.4f}, Macro F1={trans_f1:.4f}")

    print("\n" + "=" * 60)


def main():
    """Main evaluation function."""
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    if args.test_subjects:
        print(f"Test subjects (override): {args.test_subjects}")
    print("=" * 60 + "\n")

    # Load and process configuration
    config = load_config(args.config)
    config = handle_scientific_notation(config)
    config = override_config_values(config, args)
    config = resolve_paths(config)

    setup_seed(args.seed)
    device = get_device(args.no_cuda)
    print(f"Device: {device}")

    # Set up output directory
    output_dir = setup_output_directory(args, config)
    print(f"Output: {output_dir}")

    # Set up dataset
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['base_seed'] = args.seed

    data_components = setup_datasets_and_loaders(config, distributed=False, rank=0, world_size=1)
    test_loader = data_components['test_loader']
    test_dataset = data_components['test_dataset']

    test_subjects = config.get('dataset', {}).get('splits', {}).get('test_subjects', [])
    print(f"\nTest set: {len(test_dataset)} samples from {len(test_subjects)} subjects")
    print(f"Subjects: {test_subjects}")

    # Build model
    models = build_models(config, device)
    model = get_main_model(config, models)
    print(f"\nModel: {type(model).__name__} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Get prediction horizons
    if hasattr(model, 'get_prediction_horizons'):
        prediction_horizons = model.get_prediction_horizons()
    else:
        prediction_horizons = config['dataset'].get('prediction_horizons', [0])
    print(f"Horizons: {prediction_horizons}")

    # Load checkpoint
    checkpoint_info = load_model_checkpoint(args.checkpoint, model, device)

    # Set up loss and metrics
    training_components = setup_training_components(config, model, device)
    loss_fn = training_components['loss_fn']
    metrics_fns = setup_metrics(config)

    # Run evaluation
    print("\nEvaluating...")
    start_time = time.time()

    eval_metrics = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        metrics=metrics_fns,
        compute_per_subject=True
    )

    print(f"Completed in {time.time() - start_time:.2f}s")

    # Save per-subject results if available
    if 'per_subject_metrics' in eval_metrics:
        per_subject_metrics = eval_metrics.pop('per_subject_metrics')
        save_per_subject_results(
            output_dir=output_dir,
            per_subject_metrics=per_subject_metrics,
            prediction_horizons=prediction_horizons,
            prefix="test"
        )

    # Print and save results
    print_evaluation_results(eval_metrics, prediction_horizons)

    epoch = checkpoint_info.get('epoch', 0)
    save_evaluation_metrics(eval_metrics, output_dir, prediction_horizons, epoch)
    save_evaluation_summary(eval_metrics, output_dir, prediction_horizons, config, args, checkpoint_info)

    # Save config copy
    with open(os.path.join(output_dir, 'eval_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
