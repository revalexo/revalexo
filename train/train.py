#!/usr/bin/env python
# train.py - Main training script for multimodal sensor models

import os
import sys
import argparse
import time
from typing import Dict, Any, List
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json

from utils import (
    load_config, handle_scientific_notation, setup_seed, resolve_paths, get_device,
    setup_datasets_and_loaders, build_models, get_main_model,
    setup_training_components, setup_metrics, setup_logging,
    train_epoch, evaluate, save_checkpoint, load_checkpoint,
    debug_data_shape, finalize_scheduler, WarmupWrapper, load_teacher_model,
    save_per_subject_results
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multimodal sensor models')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for logs and checkpoints')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of processes for distributed training')
    parser.add_argument('--rank', type=int, default=0,
                        help='Process rank for distributed training')
    parser.add_argument('--dist-url', type=str, default='tcp://localhost:23456',
                        help='URL for distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='Backend for distributed training')
    
    # Override arguments
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for all data loaders')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of workers for all data loaders')
    
    parser.add_argument('--transition-window', type=float, default=0.5,
                        help='Transition window size in seconds (default: 0.5)')
    
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable fully deterministic mode (may impact performance)') # Dropout?
    
    return parser.parse_args()


def override_config_values(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Override configuration values with command line arguments.
    
    Args:
        config: Original configuration dictionary
        args: Command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Override batch size if specified
    if args.batch_size is not None:
        if 'dataloaders' not in config:
            config['dataloaders'] = {}
        
        for split in ['train', 'validation', 'test']:
            if split not in config['dataloaders']:
                config['dataloaders'][split] = {}
            config['dataloaders'][split]['batch_size'] = args.batch_size
    
    # Override num_workers if specified
    if args.num_workers is not None:
        if 'dataloaders' not in config:
            config['dataloaders'] = {}
        
        for split in ['train', 'validation', 'test']:
            if split not in config['dataloaders']:
                config['dataloaders'][split] = {}
            config['dataloaders'][split]['num_workers'] = args.num_workers

    return config


def log_enhanced_metrics(val_metrics: Dict[str, Any], epoch: int, num_epochs: int, 
                        train_metrics: Dict[str, Any], logging_config: Dict[str, Any],
                        prediction_horizons: List[float] = None):
    """
    Log enhanced metrics with separate transition and steady-state reporting.
    Now supports multi-horizon models with time-based horizon labels.
    """
    if not logging_config.get('enabled', False):
        return
        
    writer = logging_config.get('writer')
    
    # Log to TensorBoard
    if writer:
        for name, value in val_metrics.items():
            # Skip non-scalar values that can't be logged to TensorBoard
            if isinstance(value, (int, float)):
                writer.add_scalar(f'val/{name}', value, epoch)
            elif name.endswith('_dataframe') or name.endswith('_report'):
                # Skip dataframes/complex objects
                continue
    
    # Determine number of horizons and horizon values
    num_horizons = 1
    horizon_values = [0.0]  # Default for single horizon
    
    if prediction_horizons is not None:
        horizon_values = prediction_horizons
        num_horizons = len(horizon_values)
    else:
        # Try to infer from metrics
        for key in val_metrics.keys():
            if "_horizon_" in key:
                try:
                    h_num = int(key.split("_horizon_")[1].split("_")[0]) + 1
                    num_horizons = max(num_horizons, h_num)
                except:
                    pass
        if num_horizons > 1:
            horizon_values = list(range(num_horizons))
    
    # Enhanced console logging
    logging.info(f"Epoch {epoch+1}/{num_epochs}:")
    
    # Training metrics
    train_loss = train_metrics['loss']
    if num_horizons > 1:
        # Multi-horizon training metrics
        train_acc_overall = train_metrics.get('accuracy', 0)
        logging.info(f"  Train: loss={train_loss:.4f}, accuracy_overall={train_acc_overall:.4f}")
        for h in range(num_horizons):
            horizon_time = horizon_values[h]
            horizon_str = f"{horizon_time:.1f}s"
            # Check both standard and distillation metric names
            train_acc_h = train_metrics.get(f'accuracy_horizon_{h}', train_metrics.get(f'student_accuracy_horizon_{h}', 0))
            logging.info(f"    Horizon {horizon_str}: accuracy={train_acc_h:.4f}")
    else:
        # Single horizon training metrics
        train_acc = train_metrics.get('accuracy', 0)
        logging.info(f"  Train: loss={train_loss:.4f}, accuracy={train_acc:.4f}")
    
    # Validation metrics for each horizon
    for h in range(num_horizons):
        h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
        horizon_time = horizon_values[h]
        horizon_str = f"{horizon_time:.1f}s"
        horizon_prefix = f"Horizon {horizon_str} - " if num_horizons > 1 else ""
        
        # Overall metrics
        overall_acc = val_metrics.get(f'accuracy{h_suffix}', 0)
        overall_loss = val_metrics.get('loss', 0)  # Loss is shared across horizons
        if h == 0:  # Only log loss once
            logging.info(f"  Val {horizon_prefix}Overall: loss={overall_loss:.4f}, accuracy={overall_acc:.4f}")
        else:
            logging.info(f"  Val {horizon_prefix}Overall: accuracy={overall_acc:.4f}")
        
        # F1 scores - overall
        if f'f1score_f1{h_suffix}' in val_metrics:
            logging.info(f"    {horizon_prefix}Overall F1: {val_metrics[f'f1score_f1{h_suffix}']:.4f}, "
                        f"Precision: {val_metrics[f'f1score_precision{h_suffix}']:.4f}, "
                        f"Recall: {val_metrics[f'f1score_recall{h_suffix}']:.4f}")
        elif f'f1score{h_suffix}' in val_metrics:
            logging.info(f"    {horizon_prefix}Overall F1: {val_metrics[f'f1score{h_suffix}']:.4f}")
        
        # Classification report - overall
        if f'classificationreport_macro_f1-score{h_suffix}' in val_metrics:
            logging.info(f"    {horizon_prefix}Overall Classification Report:")
            logging.info(f"      Macro F1: {val_metrics[f'classificationreport_macro_f1-score{h_suffix}']:.4f}")
            logging.info(f"      Weighted F1: {val_metrics[f'classificationreport_weighted_f1-score{h_suffix}']:.4f}")
        
        # Steady-state metrics
        steady_acc = val_metrics.get(f'steady_accuracy{h_suffix}')
        if steady_acc is not None:
            logging.info(f"  Val {horizon_prefix}Steady-State: accuracy={steady_acc:.4f}")
            
            # Steady-state F1 scores
            if f'steady_f1score_f1{h_suffix}' in val_metrics:
                logging.info(f"    {horizon_prefix}Steady F1: {val_metrics[f'steady_f1score_f1{h_suffix}']:.4f}, "
                            f"Precision: {val_metrics[f'steady_f1score_precision{h_suffix}']:.4f}, "
                            f"Recall: {val_metrics[f'steady_f1score_recall{h_suffix}']:.4f}")
            elif f'steady_f1score{h_suffix}' in val_metrics:
                logging.info(f"    {horizon_prefix}Steady F1: {val_metrics[f'steady_f1score{h_suffix}']:.4f}")
            
            # Steady-state classification report
            if f'steady_classificationreport_macro_f1-score{h_suffix}' in val_metrics:
                logging.info(f"    {horizon_prefix}Steady Classification Report:")
                logging.info(f"      Macro F1: {val_metrics[f'steady_classificationreport_macro_f1-score{h_suffix}']:.4f}")
                logging.info(f"      Weighted F1: {val_metrics[f'steady_classificationreport_weighted_f1-score{h_suffix}']:.4f}")
        
        # Transition-state metrics
        transition_acc = val_metrics.get(f'transition_accuracy{h_suffix}')
        if transition_acc is not None:
            logging.info(f"  Val {horizon_prefix}Transition: accuracy={transition_acc:.4f}")
            
            # Transition F1 scores
            if f'transition_f1score_f1{h_suffix}' in val_metrics:
                logging.info(f"    {horizon_prefix}Transition F1: {val_metrics[f'transition_f1score_f1{h_suffix}']:.4f}, "
                            f"Precision: {val_metrics[f'transition_f1score_precision{h_suffix}']:.4f}, "
                            f"Recall: {val_metrics[f'transition_f1score_recall{h_suffix}']:.4f}")
            elif f'transition_f1score{h_suffix}' in val_metrics:
                logging.info(f"    {horizon_prefix}Transition F1: {val_metrics[f'transition_f1score{h_suffix}']:.4f}")
            
            # Transition classification report
            if f'transition_classificationreport_macro_f1-score{h_suffix}' in val_metrics:
                logging.info(f"    {horizon_prefix}Transition Classification Report:")
                logging.info(f"      Macro F1: {val_metrics[f'transition_classificationreport_macro_f1-score{h_suffix}']:.4f}")
                logging.info(f"      Weighted F1: {val_metrics[f'transition_classificationreport_weighted_f1-score{h_suffix}']:.4f}")
        
        # Add separator between horizons for clarity
        if num_horizons > 1 and h < num_horizons - 1:
            logging.info("")


def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    """
    Main training worker function with reproducible training support.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Set up distributed training
    if args.distributed:
        # Set environment variables if not already set
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '23456'
            
        # Initialize process group
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        print(f"Initialized process group: rank {rank}/{world_size}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle scientific notation in config (for instance, 1e-5 is treated as string otherwise)
    config = handle_scientific_notation(config)

    # Override config values with command line arguments
    config = override_config_values(config, args)
    
    # Resolve relative paths between training and dataset configs
    config = resolve_paths(config)
    
    # Set manual seed for reproducibility
    if args.seed is not None:
        seed = setup_seed(args.seed)
    else:
        seed = setup_seed(config.get('training', {}).get('seed', 42))
    
    # Enable deterministic mode if requested
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print("Deterministic mode enabled - performance may be impacted")
    
    # Get device
    device = get_device(args.no_cuda, rank)
    
    # Set up logging
    if args.log_dir is not None:
        # Override config log directory
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['log_dir'] = args.log_dir
    
    logging_config = setup_logging(config, args.distributed, rank)
    
    # Print training information
    if logging_config.get('enabled', False):
        print(f"Training with configuration: {args.config}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Deterministic mode: {args.deterministic}")
        print(f"Distributed: {args.distributed}")
        print(f"Transition window size: {args.transition_window}s")
        if args.distributed:
            print(f"World size: {world_size}, Rank: {rank}")
        
        # Print modalities being used
        modalities = list(config.get('modalities', {}).keys())
        print(f"Using modalities: {modalities}")
    
    # Set up datasets and data loaders
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['base_seed'] = seed  # Pass seed to dataset config
    
    data_components = setup_datasets_and_loaders(
        config, 
        distributed=args.distributed,
        rank=rank,
        world_size=world_size,
    )
    
    train_loader = data_components['train_loader']
    val_loader = data_components['val_loader']
    test_loader = data_components['test_loader']

    # Log dataset information
    if logging_config.get('enabled', False):
        train_size = len(data_components['train_dataset'])
        val_size = len(data_components['val_dataset'])
        test_size = len(data_components['test_dataset'])

        logging.info(f"Loaded {train_size} training samples")
        logging.info(f"Loaded {val_size} validation samples")
        logging.info(f"Loaded {test_size} test samples")
        logging.info(f"Random seed: {seed}")

    print("\nDebugging training data loader:")
    debug_data_shape(train_loader, num_batches=1)

    # Build models
    models = build_models(config, device)
    
    # Get the main model for training
    model = get_main_model(config, models)
    
    # Print model structure
    if logging_config.get('enabled', False):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {type(model).__name__}")
        print(model)
        print(f"Total parameters: {num_params:,}")
    
    prediction_horizons = None
    if hasattr(model, 'get_prediction_horizons'):
        prediction_horizons = model.get_prediction_horizons()
        print(f"Model prediction horizons: {prediction_horizons}")
    else:
        prediction_horizons = config['dataset'].get('prediction_horizons', [0])
        print(f"Using config prediction horizons: {prediction_horizons}")

    # Set up training components
    training_components = setup_training_components(config, model, device)
    
    optimizer = training_components['optimizer']
    loss_fn = training_components['loss_fn']
    scheduler = training_components['scheduler']
    early_stopping = training_components['early_stopping']
    checkpointing = training_components['checkpointing']
    
    # Finalize scheduler once we know steps_per_epoch
    steps_per_epoch = len(train_loader)
    scheduler = finalize_scheduler(scheduler, optimizer, steps_per_epoch)
    training_components['scheduler'] = scheduler
    print(f"Learning rate scheduler: {type(scheduler).__name__}")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    # Check if scheduler requires batch-level updates
    is_batch_level_scheduler = isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR,))

    # Check for knowledge distillation mode
    distillation_config = config.get('distillation', {})
    distillation_enabled = distillation_config.get('enabled', False)
    teacher = None
    distillation_loss = training_components.get('distillation_loss')

    if distillation_enabled:
        teacher_checkpoint = distillation_config.get('teacher_checkpoint')
        teacher_config_path = distillation_config.get('teacher_config')

        if teacher_checkpoint is None:
            raise ValueError("Distillation enabled but teacher_checkpoint not specified in config")

        print(f"\n=== Knowledge Distillation Mode ===")
        print(f"Loading teacher model from: {teacher_checkpoint}")

        # Load teacher config if provided, otherwise use current config
        if teacher_config_path:
            teacher_model_config = load_config(teacher_config_path)
            teacher_model_config = handle_scientific_notation(teacher_model_config)
            teacher_model_config = resolve_paths(teacher_model_config)
        else:
            teacher_model_config = config

        teacher = load_teacher_model(teacher_checkpoint, teacher_model_config, device, freeze=True)

        # Initialize feature-based distillation loss if needed
        if isinstance(distillation_loss, dict) and distillation_loss.get('type') == 'feature_based':
            from utils.training_utils import initialize_feature_distillation_loss
            distillation_loss = initialize_feature_distillation_loss(
                distillation_loss, model, teacher, device
            )
            print(f"Initialized {distillation_config.get('method', 'unknown')} distillation loss")

        if logging_config.get('enabled', False):
            teacher_params = sum(p.numel() for p in teacher.parameters())
            student_params = sum(p.numel() for p in model.parameters())
            print(f"Teacher model: {type(teacher).__name__} ({teacher_params:,} params, frozen)")
            print(f"Student model: {type(model).__name__} ({student_params:,} params)")
            kd_method = distillation_config.get('method', 'vanilla')
            print(f"KD Method: {kd_method}")
            print(f"Temperature: {distillation_config.get('temperature', 4.0)}")
            print(f"Alpha (task weight): {distillation_config.get('alpha', 0.5)}")
        print("=" * 40)

    # Set up metrics
    metrics = setup_metrics(config)
    
    # Load checkpoint if provided
    start_epoch = 0
    best_metric_value = float('inf') if checkpointing and checkpointing.get('mode', 'min') == 'min' else float('-inf')
    
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            checkpoint = load_checkpoint(args.checkpoint, model, optimizer, scheduler, device)
            start_epoch = checkpoint.get('epoch', 0) + 1
            
            # Update best metric value
            if checkpointing and 'metrics' in checkpoint:
                metric_name = checkpointing.get('metric', 'val_loss')
                if metric_name in checkpoint['metrics']:
                    best_metric_value = checkpoint['metrics'][metric_name]
            
            if logging_config.get('enabled', False):
                print(f"Resumed from checkpoint: {args.checkpoint}")
                print(f"Starting from epoch: {start_epoch}")
        else:
            print(f"Warning: Checkpoint not found at {args.checkpoint}")
    
    # Get training configuration
    num_epochs = config.get('training', {}).get('epochs', 50)
    
    # Get validation frequency
    val_frequency = config.get('training', {}).get('val_every', 1)
        
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Sets training seed different per epoch
        if hasattr(data_components['train_dataset'], 'set_epoch'):
            data_components['train_dataset'].set_epoch(epoch)
        
        # In train.py, for each epoch:
        if data_components.get('train_sampler') is not None:
            if hasattr(data_components['train_sampler'], 'set_epoch'):
                data_components['train_sampler'].set_epoch(epoch)

        # Log learning rate at the beginning of each epoch
        if logging_config.get('enabled', False) and logging_config.get('writer'):
            writer = logging_config.get('writer')
            writer.add_scalar('train/epoch_learning_rate', optimizer.param_groups[0]['lr'], epoch)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train for one epoch (handles both standard and distillation training)
        # Get modality order from config for distillation routing
        modality_order = list(config.get('modalities', {}).keys()) if distillation_enabled else None

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            logging_config=logging_config,
            scheduler=scheduler if is_batch_level_scheduler else None,
            teacher=teacher if distillation_enabled else None,
            distillation_loss=distillation_loss if distillation_enabled else None,
            modality_order=modality_order
        )
        
        # Only evaluate if it's a validation epoch (either by frequency or final epoch)
        is_val_epoch = (epoch + 1) % val_frequency == 0 or epoch == num_epochs - 1
        
        if is_val_epoch:
            # Evaluate on validation set with per-subject metrics
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                metrics=metrics,
                compute_per_subject=True
            )
            
            log_enhanced_metrics(val_metrics, epoch, num_epochs, train_metrics, logging_config, prediction_horizons)
                            
            # Update learning rate scheduler if needed
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
                
            # Save checkpoint if needed
            if checkpointing and checkpointing.get('enabled', False):
                checkpoint_dir = logging_config.get('log_dir')
                
                # Determine if this is the best model
                metric_name = checkpointing.get('metric', 'val_accuracy')
                
                if len(prediction_horizons) > 1:
                    # Multi-horizon model - use horizon-specific metric
                    horizon_index = checkpointing.get('horizon_index', 0)
                    horizon_time = prediction_horizons[horizon_index]
                    if metric_name == 'loss':
                        # Keep loss as-is, don't append horizon suffix
                        pass  # metric_name stays as 'loss'
                    elif metric_name.startswith('val_'):
                        base_metric = metric_name[4:]  # Remove 'val_' prefix
                        metric_name = f"{base_metric}_horizon_{horizon_index}"
                    else:
                        metric_name = f"{metric_name}_horizon_{horizon_index}"
                    
                    print(f"Using horizon {horizon_time:.1f}s (index {horizon_index}) metric '{metric_name}' for checkpointing")
                
                current_metric = val_metrics.get(metric_name)
                
                is_best = False
                if current_metric is not None:
                    is_better = checkpointing['is_better']
                    if is_better(current_metric, best_metric_value):
                        best_metric_value = current_metric
                        is_best = True
                        horizon_str = f"{prediction_horizons[checkpointing.get('horizon_index', 0)]:.1f}s" if len(prediction_horizons) > 1 else ""
                        print(f"New best model! {metric_name}: {current_metric:.4f} (horizon {horizon_str})")
                else:
                    print(f"Warning: Metric '{metric_name}' not found in validation metrics")
                    print(f"Available metrics: {list(val_metrics.keys())}")
                
                # Save checkpoint
                save_checkpoint_interval = checkpointing.get('frequency', 1)
                save_best_only = checkpointing.get('save_best_only', True)
                
                if (epoch + 1) % save_checkpoint_interval == 0:
                    if save_best_only and not is_best:
                        print("Not the best model, skipping checkpoint")
                        pass
                    else:
                        print("Saving checkpoint")
                        # Extract per-subject metrics before saving (they shouldn't go in the checkpoint)
                        per_subject_metrics = val_metrics.pop('per_subject_metrics', None)

                        # Save checkpoint with prediction horizons
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=(epoch + 1),
                            metrics=val_metrics,
                            prediction_horizons=prediction_horizons,
                            is_best=is_best
                        )

                        # Save per-subject results only when best model is found
                        if is_best and per_subject_metrics:
                            save_per_subject_results(
                                output_dir=checkpoint_dir,
                                per_subject_metrics=per_subject_metrics,
                                prediction_horizons=prediction_horizons,
                                prefix="best_val"
                            )
            
            # Check early stopping
            if early_stopping and early_stopping.get('enabled', False):
                early_stop_metric = 'loss'  # Default to loss
                current_value = val_metrics.get(early_stop_metric, float('inf'))
                
                if early_stopping['is_better'](current_value, early_stopping['best_value']):
                    # Reset counter if improved
                    early_stopping['counter'] = 0
                    early_stopping['best_value'] = current_value
                else:
                    # Increment counter if not improved
                    early_stopping['counter'] += 1
                
                # Check if we should stop
                if early_stopping['counter'] >= early_stopping.get('patience', 10):
                    if logging_config.get('enabled', False):
                        logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            # For non-validation epochs, just print training metrics
            if logging_config.get('enabled', False):
                logging.info(f"Epoch {epoch+1}/{num_epochs}:")
                logging.info(f"  Train: loss={train_metrics['loss']:.4f}, accuracy={train_metrics.get('accuracy', 0):.4f}")
            
        # Step epoch-level schedulers that are not batch-level or validation-based
        if (scheduler is not None and 
            not is_batch_level_scheduler and 
            not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and
            not isinstance(scheduler, WarmupWrapper)):
            scheduler.step()
        elif (scheduler is not None and 
            isinstance(scheduler, WarmupWrapper) and 
            not isinstance(scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
            scheduler.step()
    
    # Final evaluation on test set with per-subject metrics
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        metrics=metrics,
        compute_per_subject=True
    )

    # Save per-subject results if available
    if 'per_subject_metrics' in test_metrics and logging_config.get('enabled', False):
        per_subject_metrics = test_metrics.pop('per_subject_metrics')
        save_per_subject_results(
            output_dir=logging_config.get('log_dir', './'),
            per_subject_metrics=per_subject_metrics,
            prediction_horizons=prediction_horizons,
            prefix="test"
        )

    # Log test metrics with enhanced format for multi-horizon
    if logging_config.get('enabled', False) and test_metrics:
        logging.info("\nTest set evaluation:")
        
        # Use the prediction horizons we extracted earlier
        num_horizons = len(prediction_horizons)
        horizon_values = prediction_horizons
        
        # Test metrics for each horizon
        overall_test_loss = test_metrics.get('loss', 0)
        
        for h in range(num_horizons):
            h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
            horizon_time = horizon_values[h]
            horizon_str = f"{horizon_time:.1f}s"
            horizon_prefix = f"Horizon {horizon_str} - " if num_horizons > 1 else ""
            
            # Overall test metrics
            overall_test_acc = test_metrics.get(f'accuracy{h_suffix}', 0)
            if h == 0:  # Only log loss once
                logging.info(f"  Test {horizon_prefix}Overall: loss={overall_test_loss:.4f}, accuracy={overall_test_acc:.4f}")
            else:
                logging.info(f"  Test {horizon_prefix}Overall: accuracy={overall_test_acc:.4f}")
            
            # Steady-state test metrics
            steady_test_acc = test_metrics.get(f'steady_accuracy{h_suffix}')
            if steady_test_acc is not None:
                logging.info(f"  Test {horizon_prefix}Steady-State: accuracy={steady_test_acc:.4f}")
                
            # Transition test metrics
            transition_test_acc = test_metrics.get(f'transition_accuracy{h_suffix}')
            if transition_test_acc is not None:
                logging.info(f"  Test {horizon_prefix}Transition: accuracy={transition_test_acc:.4f}")
        
        # Log other metrics if available (first horizon only to avoid spam)
        for name, value in test_metrics.items():
            if (isinstance(value, (int, float)) and 
                not name.endswith('accuracy') and 
                not name.endswith('loss') and
                '_horizon_' not in name):  # Only log non-horizon-specific metrics once
                logging.info(f"  {name}: {value:.4f}")
    
    # Save results for hyperparameter search
    if logging_config.get('enabled', False):
        results = {
            'test_loss': test_metrics.get('loss', 0.0),
            'final_epoch': epoch,
            'prediction_horizons': prediction_horizons,
            'seed_used': seed,  # Save the seed for reference
            'deterministic_mode': args.deterministic
        }
        
        # Add results for each horizon
        if len(prediction_horizons) > 1:
            # Multi-horizon results
            overall_accuracies = []
            steady_accuracies = []
            transition_accuracies = []
            
            for h in range(len(prediction_horizons)):
                h_suffix = f"_horizon_{h}"
                horizon_time = prediction_horizons[h]
                horizon_str = f"{horizon_time:.1f}s"
                
                test_acc = test_metrics.get(f'accuracy{h_suffix}', 0.0)
                steady_acc = test_metrics.get(f'steady_accuracy{h_suffix}', 0.0)
                transition_acc = test_metrics.get(f'transition_accuracy{h_suffix}', 0.0)
                
                results[f'test_accuracy_horizon_{horizon_str}'] = test_acc
                results[f'test_steady_accuracy_horizon_{horizon_str}'] = steady_acc
                results[f'test_transition_accuracy_horizon_{horizon_str}'] = transition_acc
                
                overall_accuracies.append(test_acc)
                steady_accuracies.append(steady_acc)
                transition_accuracies.append(transition_acc)
            
            # Also save averages across horizons
            results['test_accuracy_avg'] = sum(overall_accuracies) / len(overall_accuracies)
            results['test_steady_accuracy_avg'] = sum(steady_accuracies) / len(steady_accuracies)
            results['test_transition_accuracy_avg'] = sum(transition_accuracies) / len(transition_accuracies)
            
            # Use specified horizon for best validation metric
            horizon_index = checkpointing.get('horizon_index', 0) if checkpointing else 0
            horizon_time = prediction_horizons[horizon_index]
            horizon_str = f"{horizon_time:.1f}s"
            metric_name = checkpointing.get('metric', 'accuracy') if checkpointing else 'accuracy'
            if metric_name.startswith('val_'):
                base_metric = metric_name[4:]
            else:
                base_metric = metric_name
            best_val_metric_name = f"{base_metric}_horizon_{horizon_index}"
            results['best_val_accuracy'] = best_metric_value if checkpointing and checkpointing.get('metric') == 'val_accuracy' else val_metrics.get(best_val_metric_name, 0.0)
            results['best_val_horizon'] = horizon_str
        else:
            # Single horizon results
            horizon_str = f"{prediction_horizons[0]:.1f}s"
            results.update({
                'best_val_accuracy': best_metric_value if checkpointing and checkpointing.get('metric', 'val_accuracy') == 'val_accuracy' else val_metrics.get('accuracy', 0.0),
                'test_accuracy': test_metrics.get('accuracy', 0.0),
                'test_steady_accuracy': test_metrics.get('steady_accuracy', 0.0),
                'test_transition_accuracy': test_metrics.get('transition_accuracy', 0.0),
                'horizon': horizon_str
            })
        
        save_results(results, logging_config.get('log_dir', './'))

    # Clean up
    if logging_config.get('enabled', False) and logging_config.get('writer') is not None:
        logging_config.get('writer').close()
    
    # Clean up distributed
    if args.distributed:
        dist.destroy_process_group()


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save training results to a JSON file."""
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes
        world_size = args.world_size
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process training
        main_worker(0, 1, args)


if __name__ == '__main__':
    main()