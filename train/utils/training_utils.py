# utils/training_utils.py

import os
import time
import importlib
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np


def get_model_feature_dim(model: nn.Module) -> int:
    """
    Get the feature dimension of a model for knowledge distillation.

    Args:
        model: Model to get feature dimension from

    Returns:
        Feature dimension (int)
    """
    # Try different methods to get feature dimension
    if hasattr(model, 'get_feature_dim'):
        return model.get_feature_dim()
    elif hasattr(model, 'encode_features'):
        # Try to infer from model structure
        if hasattr(model, 'lstm_units'):
            return model.lstm_units
        elif hasattr(model, 'feature_dim'):
            return model.feature_dim
        elif hasattr(model, 'classifier'):
            # Check classifier input dimension
            if hasattr(model.classifier, 'shared_fc'):
                return model.classifier.shared_fc[0].in_features
            elif hasattr(model.classifier, 'fc'):
                return model.classifier.fc[0].in_features if hasattr(model.classifier.fc, '__getitem__') else model.classifier.fc.in_features
    # Default fallback
    return 512


def initialize_feature_distillation_loss(
    distillation_config: Dict[str, Any],
    student: nn.Module,
    teacher: nn.Module,
    device: torch.device
) -> nn.Module:
    """
    Initialize feature-based distillation loss with actual model dimensions.

    Args:
        distillation_config: Distillation configuration dict
        student: Student model
        teacher: Teacher model
        device: Device to use

    Returns:
        Initialized FeatureDistillationLoss module
    """
    from losses.distillation import FeatureDistillationLoss

    # Get feature dimensions from models
    student_dim = distillation_config.get('student_dim') or get_model_feature_dim(student)
    teacher_dim = distillation_config.get('teacher_dim') or get_model_feature_dim(teacher)

    print(f"Feature dimensions - Student: {student_dim}, Teacher: {teacher_dim}")

    # Extract parameters
    method = distillation_config['method']
    task_loss_fn = distillation_config['task_loss_fn']
    alpha = distillation_config['alpha']
    temperature = distillation_config['temperature']
    prediction_horizons = distillation_config['prediction_horizons']
    horizon_loss_weights = distillation_config['horizon_loss_weights']
    include_vanilla_kd = distillation_config.get('include_vanilla_kd', False)
    vanilla_kd_weight = distillation_config.get('vanilla_kd_weight', 0.5)

    # Method-specific kwargs
    method_kwargs = {}
    if method == 'fitnets':
        method_kwargs['beta'] = distillation_config.get('beta', 100.0)
        method_kwargs['normalize_features'] = distillation_config.get('normalize_features', True)
    elif method == 'rkd':
        method_kwargs['distance_weight'] = distillation_config.get('distance_weight', 25.0)
        method_kwargs['angle_weight'] = distillation_config.get('angle_weight', 50.0)
    elif method == 'crd':
        method_kwargs['embed_dim'] = distillation_config.get('embed_dim', 128)
        method_kwargs['crd_temperature'] = distillation_config.get('crd_temperature', 0.07)
        method_kwargs['use_memory_bank'] = distillation_config.get('use_memory_bank', False)
        method_kwargs['memory_size'] = distillation_config.get('memory_size', 16384)

    # Create the loss
    loss = FeatureDistillationLoss(
        task_loss_fn=task_loss_fn,
        method=method,
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        alpha=alpha,
        include_vanilla_kd=include_vanilla_kd,
        vanilla_kd_weight=vanilla_kd_weight,
        temperature=temperature,
        prediction_horizons=prediction_horizons,
        horizon_loss_weights=horizon_loss_weights,
        **method_kwargs
    ).to(device)

    return loss


def load_teacher_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
    freeze: bool = True
) -> nn.Module:
    """
    Load a pre-trained teacher model from checkpoint for knowledge distillation.

    Args:
        checkpoint_path: Path to teacher model checkpoint
        config: Configuration dictionary for model building
        device: Device to load model on
        freeze: Whether to freeze teacher parameters (default: True)

    Returns:
        Loaded teacher model in evaluation mode
    """
    from utils.model_utils import build_models, get_main_model

    # Build model from config
    models = build_models(config, device)
    teacher = get_main_model(config, models)

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher.load_state_dict(checkpoint)

    # Freeze parameters if requested
    if freeze:
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        print(f"Teacher model loaded and frozen from: {checkpoint_path}")
    else:
        print(f"Teacher model loaded (trainable) from: {checkpoint_path}")

    return teacher

class MultiHorizonLoss(nn.Module):
    """
    Loss function for multi-horizon prediction.
    Computes loss for each prediction horizon and combines them.
    """
    def __init__(self, base_loss_fn, prediction_horizons, loss_weights=None):
        """
        Initialize multi-horizon loss.
        
        Args:
            base_loss_fn: Base loss function (e.g., CrossEntropyLoss)
            prediction_horizons: List of prediction horizons
            loss_weights: Optional weights for each horizon's loss (defaults to equal weights)
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.prediction_horizons = prediction_horizons
        self.num_heads = len(prediction_horizons)
        
        # Set loss weights (equal by default)
        if loss_weights is None:
            self.loss_weights = torch.ones(self.num_heads) / self.num_heads
        else:
            if len(loss_weights) != self.num_heads:
                raise ValueError(f"Number of loss weights ({len(loss_weights)}) must match number of horizons ({self.num_heads})")
            self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
            
        print(f"Multi-horizon loss initialized with weights: {self.loss_weights.tolist()}")
    
    def forward(self, outputs, targets):
        """
        Compute multi-horizon loss.
        
        Args:
            outputs: List of model outputs for each horizon [batch_size, num_classes]
            targets: List of target labels for each horizon [batch_size]
            
        Returns:
            dict: Dictionary containing individual and total losses
        """
        if len(outputs) != self.num_heads:
            raise ValueError(f"Number of outputs ({len(outputs)}) must match number of horizons ({self.num_heads})")
        
        if len(targets) != self.num_heads:
            raise ValueError(f"Number of target sets ({len(targets)}) must match number of horizons ({self.num_heads})")
        
        # Move loss weights to same device as outputs
        device = outputs[0].device
        loss_weights = self.loss_weights.to(device)
        
        # Compute loss for each horizon
        individual_losses = []
        for i, (output, target) in enumerate(zip(outputs, targets)):
            loss = self.base_loss_fn(output, target)
            individual_losses.append(loss)
        
        # Combine losses with weights
        total_loss = sum(weight * loss for weight, loss in zip(loss_weights, individual_losses))
        
        return {
            'total_loss': total_loss,
            'individual_losses': individual_losses,
            'horizon_losses': {f'horizon_{h}': loss.item() for h, loss in zip(self.prediction_horizons, individual_losses)}
        }

def setup_training_components(config: Dict[str, Any], model: nn.Module, 
                             device: torch.device) -> Dict[str, Any]:
    """
    Set up training components with support for differential learning rates.
    """
    from utils.lr_schedulers import create_scheduler
    
    training_config = config.get('training', {})
    
    # Check if this is a fusion model (has modality_encoders attribute)
    is_fusion_model = hasattr(model, 'modality_encoders')
    use_differential_lr = training_config.get('use_differential_lr', False) and is_fusion_model
    
    # Set up optimizer
    optimizer_name = training_config.get('optimizer', {}).get('name', 'Adam')
    optimizer_params = training_config.get('optimizer', {}).get('params', {}).copy()
    base_lr = optimizer_params.pop('lr', 1e-3)  # Remove lr from params

    # FOR EVI-MAE:
    head_lr_multiplier = training_config.get('head_lr_multiplier', None)
    base_lr_multiplier = training_config.get('base_lr_multiplier', 1.0)
    if head_lr_multiplier is not None:
        # Create parameter groups for head vs base
        head_params = []
        base_params = []
        
        for name, param in model.named_parameters():
            # Check if this is a classification head parameter
            if 'mlp_head' in name or 'classifier' in name or 'another_mlp_head' in name:
                head_params.append(param)
            else:
                base_params.append(param)
        
        param_groups = [
            {'params': base_params, 'lr': base_lr * base_lr_multiplier},
            {'params': head_params, 'lr': base_lr * head_lr_multiplier}
        ]
        
        print(f"Base layers LR: {base_lr * base_lr_multiplier}")
        print(f"Head layers LR: {base_lr * head_lr_multiplier}")
        
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(param_groups, **optimizer_params)

    elif use_differential_lr:
        # UNUSED FOR NOW, FOR FUTURE EXPERIMENTS
        # Create parameter groups with different learning rates
        param_groups = []
        
        # Get modality-specific learning rate multipliers
        lr_multipliers = training_config.get('lr_multipliers', {})
        
        # Group parameters by modality
        for modality, encoder in model.modality_encoders.items():
            modality_lr_mult = lr_multipliers.get(modality, 1.0)
            
            # Check if this is a pretrained model (simple heuristic)
            has_pretrained = any('pretrained' in str(param) for param in encoder.state_dict().keys())
            if has_pretrained and modality not in lr_multipliers:
                # Default: lower LR for pretrained models
                modality_lr_mult = 0.1
                print(f"Auto-detected pretrained model for {modality}, using lr multiplier: {modality_lr_mult}")
            
            param_groups.append({
                'params': list(encoder.parameters()),
                'lr': base_lr * modality_lr_mult,
                'name': f'{modality}_encoder'
            })
            print(f"Setting {modality} encoder LR to {base_lr * modality_lr_mult}")
        
        # Fusion layers get base learning rate
        fusion_params = []
        fusion_param_names = []
        for name, param in model.named_parameters():
            # Skip encoder parameters
            if not any(f'modality_encoders.{m}' in name for m in model.modalities):
                fusion_params.append(param)
                fusion_param_names.append(name)
        
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': base_lr,
                'name': 'fusion_layers'
            })
            print(f"Setting fusion layers LR to {base_lr}")
            print(f"Fusion layer parameters: {len(fusion_params)} params")
        
        # Create optimizer with parameter groups
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(param_groups, **optimizer_params)
        
    else:
        # Standard optimizer setup
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(model.parameters(), lr=base_lr, **optimizer_params)
    
    # Set up loss function - handles both single and multi-horizon
    task_config = config.get('task', {})
    loss_name = task_config.get('loss', {}).get('name', 'CrossEntropyLoss')
    loss_params = task_config.get('loss', {}).get('params', {})

    if hasattr(nn, loss_name):
        base_loss_class = getattr(nn, loss_name)
        base_loss_fn = base_loss_class(**loss_params)
    else:
        try:
            loss_module = importlib.import_module('losses')
            if hasattr(loss_module, loss_name):
                base_loss_class = getattr(loss_module, loss_name)
                base_loss_fn = base_loss_class(**loss_params)
            else:
                raise ValueError(f"Loss function {loss_name} not found")
        except ImportError:
            raise ValueError(f"Loss function {loss_name} not found and no custom losses module available")
    
    # Check if model has multiple prediction heads
    if hasattr(model, 'get_num_prediction_heads') and model.get_num_prediction_heads() > 1:
        prediction_horizons = model.get_prediction_horizons()
        loss_weights = task_config.get('horizon_loss_weights', None)
        loss_fn = MultiHorizonLoss(base_loss_fn, prediction_horizons, loss_weights)
        print(f"Using multi-horizon loss for horizons: {prediction_horizons}")
    else:
        loss_fn = base_loss_fn
        print("Using single-horizon loss")
    

    # Check for knowledge distillation configuration
    distillation_config = config.get('distillation', {})
    distillation_loss = None
    distillation_method = distillation_config.get('method', 'vanilla')

    if distillation_config.get('enabled', False):
        temperature = distillation_config.get('temperature', 4.0)
        alpha = distillation_config.get('alpha', 0.5)
        loss_weights = task_config.get('horizon_loss_weights', None)

        if distillation_method == 'vanilla':
            # Standard vanilla KD (logit-based)
            from losses.distillation import DistillationLoss

            distillation_loss = DistillationLoss(
                task_loss_fn=base_loss_fn,
                temperature=temperature,
                alpha=alpha,
                prediction_horizons=prediction_horizons,
                horizon_loss_weights=loss_weights
            )
            print(f"Vanilla KD enabled: temperature={temperature}, alpha={alpha}")

        elif distillation_method == 'nkd':
            # NKD: Normalized Knowledge Distillation (Yang et al., ICCV 2023)
            from losses.distillation import NKDDistillationLoss

            nkd_config = distillation_config.get('nkd', {})
            nkd_temperature = nkd_config.get('temperature', 1.0)
            nkd_gamma = nkd_config.get('gamma', 1.5)

            distillation_loss = NKDDistillationLoss(
                task_loss_fn=base_loss_fn,
                temperature=nkd_temperature,
                gamma=nkd_gamma,
                alpha=alpha,
                prediction_horizons=prediction_horizons,
                horizon_loss_weights=loss_weights
            )
            print(f"NKD enabled: alpha={alpha}, temperature={nkd_temperature}, gamma={nkd_gamma}")

        elif distillation_method in ['fitnets', 'rkd', 'crd']:
            # Feature-based KD methods
            from losses.distillation import FeatureDistillationLoss

            # Get student and teacher feature dimensions from config
            # These will be set during training when models are available
            student_dim = distillation_config.get('student_dim', None)
            teacher_dim = distillation_config.get('teacher_dim', None)

            # Method-specific parameters
            method_kwargs = {}

            if distillation_method == 'fitnets':
                fitnets_config = distillation_config.get('fitnets', {})
                method_kwargs['beta'] = fitnets_config.get('beta', 100.0)
                method_kwargs['normalize_features'] = fitnets_config.get('normalize_features', True)
                print(f"FitNets KD enabled: alpha={alpha}, beta={method_kwargs['beta']}")

            elif distillation_method == 'rkd':
                rkd_config = distillation_config.get('rkd', {})
                method_kwargs['distance_weight'] = rkd_config.get('distance_weight', 25.0)
                method_kwargs['angle_weight'] = rkd_config.get('angle_weight', 50.0)
                print(f"RKD enabled: alpha={alpha}, distance_weight={method_kwargs['distance_weight']}, angle_weight={method_kwargs['angle_weight']}")

            elif distillation_method == 'crd':
                crd_config = distillation_config.get('crd', {})
                method_kwargs['embed_dim'] = crd_config.get('embed_dim', 128)
                method_kwargs['crd_temperature'] = crd_config.get('temperature', 0.07)
                method_kwargs['use_memory_bank'] = crd_config.get('use_memory_bank', False)
                method_kwargs['memory_size'] = crd_config.get('memory_size', 16384)
                print(f"CRD enabled: alpha={alpha}, embed_dim={method_kwargs['embed_dim']}, temperature={method_kwargs['crd_temperature']}")

            # Include vanilla KD if requested
            include_vanilla_kd = distillation_config.get('include_vanilla_kd', False)
            vanilla_kd_weight = distillation_config.get('vanilla_kd_weight', 0.5)

            # Store config for later initialization (needs model dimensions)
            distillation_loss = {
                'type': 'feature_based',
                'method': distillation_method,
                'task_loss_fn': base_loss_fn,
                'alpha': alpha,
                'temperature': temperature,
                'prediction_horizons': prediction_horizons,
                'horizon_loss_weights': loss_weights,
                'include_vanilla_kd': include_vanilla_kd,
                'vanilla_kd_weight': vanilla_kd_weight,
                'student_dim': student_dim,
                'teacher_dim': teacher_dim,
                **method_kwargs
            }

        else:
            raise ValueError(f"Unknown distillation method: {distillation_method}")
    
    # Set up learning rate scheduler
    scheduler = create_scheduler(config, optimizer)
    
    # Set up early stopping
    early_stopping = None
    if training_config.get('early_stopping', {}).get('enabled', False):
        early_stopping = {
            'patience': training_config['early_stopping'].get('patience', 10),
            'min_delta': training_config['early_stopping'].get('min_delta', 0.001),
            'counter': 0,
            'best_value': float('inf'),
            'is_better': lambda curr, best: curr < best - training_config['early_stopping'].get('min_delta', 0.001)
        }
    
    # Set up checkpointing
    checkpointing = None
    if training_config.get('checkpointing', {}).get('enabled', False):
        checkpoint_horizon_idx = training_config['checkpointing'].get('horizon_index', 0)
        
        checkpointing = {
            'enabled': True,
            'frequency': training_config['checkpointing'].get('frequency', 1),
            'save_best_only': training_config['checkpointing'].get('save_best_only', True),
            'metric': training_config['checkpointing'].get('metric', 'val_accuracy'),
            'mode': training_config['checkpointing'].get('mode', 'max'),
            'horizon_index': checkpoint_horizon_idx,
            'best_value': float('inf') if training_config['checkpointing'].get('mode', 'max') == 'min' else float('-inf'),
            'is_better': lambda curr, best: curr < best if training_config['checkpointing'].get('mode', 'max') == 'min' else curr > best
        }
    
    return {
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'distillation_loss': distillation_loss,
        'scheduler': scheduler,
        'early_stopping': early_stopping,
        'checkpointing': checkpointing
    }

def setup_metrics(config: Dict[str, Any]) -> Dict[str, Callable]:
    """
    Set up evaluation metrics based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of metric functions
    """
    metrics = {}
    metrics_config = config.get('metrics', [])
    
    # Import metrics module
    metrics_module = importlib.import_module('metrics')
    
    # Set up each metric
    for metric_config in metrics_config:
        metric_name = metric_config.get('name')
        metric_params = metric_config.get('params', {})
        
        # Try to get metric from metrics module
        if hasattr(metrics_module, metric_name):
            metric_class = getattr(metrics_module, metric_name)
            
            # Check if it's a class (needs instantiation) or a function
            if isinstance(metric_class, type):
                metric = metric_class(**metric_params)
                metrics[metric_name.lower()] = metric
            else:
                # It's a function, partial with params
                from functools import partial
                metrics[metric_name.lower()] = partial(metric_class, **metric_params)
        else:
            raise ValueError(f"Metric {metric_name} not found in metrics module")
    
    return metrics


def setup_logging(config: Dict[str, Any], distributed: bool = False, rank: int = 0) -> Dict[str, Any]:
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary
        distributed: Whether to set up for distributed training
        rank: Current process rank for distributed training
        
    Returns:
        Dictionary containing logging components
    """
    import datetime
    
    logging_config = config.get('logging', {})
    
    # Only the main process should log in distributed setting
    if distributed and rank != 0:
        return {'enabled': False}
    
    # Get configuration values
    base_log_dir = logging_config.get('log_dir', 'outputs/logs')
    tensorboard = logging_config.get('tensorboard', False)
    log_frequency = logging_config.get('log_frequency', 10)
    
    # Create a unique run name based on timestamp and config details
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract useful identifiers from config
    dataset_name = config.get('dataset', {}).get('name', 'unknown')
    modalities = '-'.join(list(config.get('modalities', {}).keys()))
    model_names = '-'.join([m.get('name', 'unknown') for m in config.get('models', {}).values()])
    run_name = f"{timestamp}_{dataset_name}_{modalities}_{model_names}"
    
    # Create run-specific log directory
    log_dir = os.path.join(base_log_dir, run_name)
    
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ['SLURM_JOB_ID']
        log_dir = os.path.join(base_log_dir, f"{run_name}_job{job_id}")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file logging
    log_file = os.path.join(log_dir, "history.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')  # Simplified for console output
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log the run information
    logging.info(f"Starting new training run: {run_name}")
    logging.info(f"Logs will be saved to: {log_dir}")

    # Log SLURM job ID and node list if available
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ['SLURM_JOB_ID']
        logging.info(f"SLURM Job ID: {job_id}")
        logging.info(f"SLURM Node List: {os.environ.get('SLURM_NODELIST', 'unknown')}")
        
    
    # Set up tensorboard
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Save config to log directory for reproducibility
    config_path = os.path.join(log_dir, "config.yaml")
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Configuration saved to: {config_path}")
    except Exception as e:
        logging.warning(f"Could not save config file: {e}")
    
    return {
        'enabled': True,
        'log_dir': log_dir,
        'writer': writer,
        'log_frequency': log_frequency,
        'logger': root_logger,
        'run_name': run_name
    }


def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
               device: torch.device, epoch: int,
               logging_config: Dict[str, Any] = None,
               scheduler=None,
               teacher: nn.Module = None,
               distillation_loss: nn.Module = None,
               modality_order: List[str] = None) -> Dict[str, float]:
    """
    Train model for one epoch.
    Supports multi-horizon models and optional knowledge distillation.

    Args:
        model: Model to train (student in distillation mode)
        loader: Data loader
        optimizer: Optimizer
        loss_fn: Loss function (could be MultiHorizonLoss) - used when NOT doing distillation
        device: Device to use for training
        epoch: Current epoch
        logging_config: Logging configuration
        scheduler: Batch-level learning rate scheduler
        teacher: Optional teacher model for knowledge distillation (frozen)
        distillation_loss: Optional distillation loss function (handles multi-horizon)
        modality_order: List of modality names in order they appear in dataloader
            (from config['modalities'].keys()). Used for routing in distillation.

    Returns:
        Dictionary of training metrics
    """
    from tqdm import tqdm

    model.train()

    # Knowledge distillation mode
    use_distillation = teacher is not None and distillation_loss is not None
    use_feature_distillation = use_distillation and hasattr(distillation_loss, 'feature_kd_loss')
    if use_distillation:
        teacher.eval()  # Teacher always in eval mode

    # Get number of horizons
    if use_distillation:
        num_horizons = distillation_loss.num_horizons
    elif isinstance(loss_fn, MultiHorizonLoss):
        num_horizons = loss_fn.num_heads
    else:
        num_horizons = 1

    is_multi_horizon = num_horizons > 1

    # Initialize metrics tracking
    total_loss = 0.0
    total_task_loss = 0.0 if use_distillation else None
    total_kd_loss = 0.0 if use_distillation else None
    horizon_losses = [0.0] * num_horizons if is_multi_horizon and not use_distillation else None
    correct = [0] * num_horizons
    teacher_correct = [0] * num_horizons if use_distillation else None  # Track teacher accuracy
    total = 0

    start_time = time.time()
    
    # Get logging configuration
    log_enabled = logging_config and logging_config.get('enabled', False)
    log_freq = logging_config.get('log_frequency', 10) if log_enabled else 0
    writer = logging_config.get('writer', None) if log_enabled else None
    
    # Create progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)
    
    for i, batch in enumerate(pbar):
        # Get inputs, labels, transition flags (subjects ignored during training)
        # Batch format: (modality_data..., labels, transition_flags, subjects)
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 4:
                # New format with subjects
                inputs = [x.to(device) for x in batch[:-3]]
                labels = batch[-3]  # This is now a list of tensors for multi-horizon
                transition_flags = batch[-2]
                # subjects = batch[-1]  # Not needed for training
            elif len(batch) >= 3:
                # Old format without subjects (backward compatibility)
                inputs = [x.to(device) for x in batch[:-2]]
                labels = batch[-2]
                transition_flags = batch[-1]
            else:
                inputs = [x.to(device) for x in batch[:-1]]
                labels = batch[-1]

            # Handle single modality case
            if len(inputs) == 1:
                inputs = inputs[0]
        else:
            raise ValueError("Loader should return a list or tuple of tensors")
        
        # Handle labels for multi-horizon
        if is_multi_horizon:
            # For multi-horizon: dataset returns list of labels
            if isinstance(labels, list) and len(labels) > 0:
                if len(labels) != num_horizons:
                    raise ValueError(f"Number of label horizons ({len(labels)}) doesn't match expected horizons ({num_horizons})")
                
                # Move all horizon labels to device
                horizon_labels = []
                for h in range(num_horizons):
                    if isinstance(labels[h], torch.Tensor):
                        horizon_labels.append(labels[h].to(device))
                    else:
                        h_labels = torch.tensor(labels[h], dtype=torch.long).to(device)
                        horizon_labels.append(h_labels)
                labels = horizon_labels
            else:
                raise ValueError(f"Multi-horizon model expects list of label tensors, got: {type(labels)}")
        else:
            # For single-horizon: dataset returns single label value
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            elif isinstance(labels, list):
                # Handle case where collate function might still wrap in list
                if len(labels) == 1:
                    labels = torch.tensor(labels[0], dtype=torch.long).to(device) if not isinstance(labels[0], torch.Tensor) else labels[0].to(device)
                else:
                    labels = torch.tensor(labels, dtype=torch.long).to(device)
            else:
                labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        # Forward pass - handle distillation routing
        if use_distillation:
            # Build input dict if we have modality_order and multiple inputs
            if modality_order and isinstance(inputs, list) and len(inputs) > 1:
                input_dict = {mod: inputs[i] for i, mod in enumerate(modality_order)}
            else:
                input_dict = None

            # Route inputs to teacher based on teacher's expected modalities
            if hasattr(teacher, 'modalities') and input_dict:
                # Teacher is multimodal - provide dict of its modalities
                teacher_input = {mod: input_dict[mod] for mod in teacher.modalities if mod in input_dict}
            elif input_dict and modality_order:
                # Teacher is single-modal, find which modality it needs
                # Convention: video models expect 'video', IMU models expect 'raw_imu'
                if 'video' in input_dict and not hasattr(model, 'modalities'):
                    # Assume teacher takes video if student is IMU-only
                    teacher_input = input_dict.get('video', inputs[-1] if isinstance(inputs, list) else inputs)
                else:
                    teacher_input = inputs[0] if isinstance(inputs, list) else inputs
            else:
                teacher_input = inputs[0] if isinstance(inputs, list) else inputs

            # Route inputs to student based on student's expected modalities
            if hasattr(model, 'modalities') and input_dict:
                # Student is multimodal
                student_input = {mod: input_dict[mod] for mod in model.modalities if mod in input_dict}
            elif input_dict:
                # Student is single-modal (e.g., DeepConvLSTM for IMU)
                student_input = input_dict.get('raw_imu', inputs[0] if isinstance(inputs, list) else inputs)
            else:
                student_input = inputs[0] if isinstance(inputs, list) else inputs

            # Teacher forward pass (no gradients)
            teacher_features = None
            with torch.no_grad():
                if isinstance(teacher_input, dict):
                    teacher_outputs = teacher(**teacher_input)
                    # Extract features for feature-based distillation
                    if use_feature_distillation and hasattr(teacher, 'encode_features'):
                        teacher_features = teacher.encode_features(**teacher_input)
                else:
                    teacher_outputs = teacher(teacher_input)
                    # Extract features for feature-based distillation
                    if use_feature_distillation and hasattr(teacher, 'encode_features'):
                        teacher_features = teacher.encode_features(teacher_input)

                if not isinstance(teacher_outputs, list):
                    teacher_outputs = [teacher_outputs]

            # Student forward pass
            student_features = None
            if isinstance(student_input, dict):
                outputs = model(**student_input)
                # Extract features for feature-based distillation
                if use_feature_distillation and hasattr(model, 'encode_features'):
                    student_features = model.encode_features(**student_input)
            else:
                outputs = model(student_input)
                # Extract features for feature-based distillation
                if use_feature_distillation and hasattr(model, 'encode_features'):
                    student_features = model.encode_features(student_input)

            if not isinstance(outputs, list):
                outputs = [outputs]

        else:
            # Standard forward pass (no distillation)
            if hasattr(model, 'modalities') and isinstance(inputs, list):
                # For multimodal model
                modality_inputs = {model.modalities[i]: inputs[i] for i in range(len(inputs))}
                outputs = model(**modality_inputs)
            else:
                # For single modality models
                outputs = model(inputs)

        # Compute loss
        if use_distillation:
            # Ensure labels is a list for distillation loss
            if not isinstance(labels, list):
                labels = [labels]

            if use_feature_distillation and student_features is not None and teacher_features is not None:
                # Feature-based distillation (FitNets, RKD, CRD)
                loss_result = distillation_loss(
                    outputs, teacher_outputs,
                    student_features, teacher_features,
                    labels
                )
                loss = loss_result['total_loss']
                task_loss = loss_result['task_loss']
                kd_loss = loss_result['feature_kd_loss']
            else:
                # Vanilla KD (logit-based)
                loss_result = distillation_loss(outputs, teacher_outputs, labels)
                loss = loss_result['total_loss']
                task_loss = loss_result['task_loss']
                kd_loss = loss_result['kd_loss']
        elif is_multi_horizon:
            loss_result = loss_fn(outputs, labels)
            loss = loss_result['total_loss']
            individual_losses = loss_result['individual_losses']
        else:
            loss = loss_fn(outputs[0], labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step batch-level scheduler if provided
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Compute metrics
        total_loss += loss.item()

        if use_distillation:
            # Track distillation-specific losses
            total_task_loss += task_loss.item()
            total_kd_loss += kd_loss.item()

            # Ensure labels is list for accuracy computation
            if not isinstance(labels, list):
                labels = [labels]

            # Calculate accuracy for each horizon (student and teacher)
            for h in range(min(len(outputs), len(labels))):
                # Student accuracy
                _, predicted = torch.max(outputs[h], 1)
                correct[h] += (predicted == labels[h]).sum().item()

                # Teacher accuracy
                if h < len(teacher_outputs):
                    _, teacher_predicted = torch.max(teacher_outputs[h], 1)
                    teacher_correct[h] += (teacher_predicted == labels[h]).sum().item()
            total += labels[0].size(0)

            # Update progress bar with both student and teacher accuracy
            student_acc = correct[0] / total if total > 0 else 0.0
            teacher_acc = teacher_correct[0] / total if total > 0 else 0.0
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'task': f"{task_loss.item():.4f}",
                'kd': f"{kd_loss.item():.4f}",
                'stu_acc': f"{student_acc:.4f}",
                'tea_acc': f"{teacher_acc:.4f}"
            })

        elif is_multi_horizon:
            # Update horizon-specific losses
            for h, h_loss in enumerate(individual_losses):
                horizon_losses[h] += h_loss.item()
            
            # Calculate accuracy for each horizon
            for h, (output, label) in enumerate(zip(outputs, labels)):
                _, predicted = torch.max(output, 1)
                correct[h] += (predicted == label).sum().item()
            
            total += labels[0].size(0)  # All horizons have same batch size
            
            # Update progress bar with first horizon metrics
            current_accuracy = correct[0] / total if total > 0 else 0.0
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc_h0': f"{current_accuracy:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        else:
            # Single horizon: use first output from list
            if len(outputs[0].shape) > 1 and outputs[0].shape[1] > 1:
                _, predicted = torch.max(outputs[0], 1)
                correct[0] += (predicted == labels).sum().item()
                total += labels.size(0)

                current_accuracy = correct[0] / total if total > 0 else 0.0
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_accuracy:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # Log metrics
        if log_enabled and writer and i % log_freq == 0:
            global_step = epoch * len(loader) + i
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if use_distillation:
                writer.add_scalar('train/task_loss', task_loss.item(), global_step)
                writer.add_scalar('train/kd_loss', kd_loss.item(), global_step)
                for h in range(num_horizons):
                    if total > 0:
                        h_acc = correct[h] / total
                        writer.add_scalar(f'train/student_accuracy_horizon_{h}', h_acc, global_step)
                        # Log teacher accuracy
                        if teacher_correct is not None:
                            t_acc = teacher_correct[h] / total
                            writer.add_scalar(f'train/teacher_accuracy_horizon_{h}', t_acc, global_step)
            elif is_multi_horizon:
                # Log individual horizon losses and accuracies
                for h in range(num_horizons):
                    writer.add_scalar(f'train/loss_horizon_{h}', individual_losses[h].item(), global_step)
                    if total > 0:
                        h_acc = correct[h] / total
                        writer.add_scalar(f'train/accuracy_horizon_{h}', h_acc, global_step)
            else:
                if total > 0:
                    batch_accuracy = (predicted == labels).float().mean().item()
                    writer.add_scalar('train/accuracy', batch_accuracy, global_step)

    # Compute average metrics
    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time

    # Prepare return metrics
    metrics = {
        'loss': avg_loss,
        'epoch_time': epoch_time
    }

    if use_distillation:
        # Distillation-specific metrics
        metrics['task_loss'] = total_task_loss / len(loader)
        metrics['kd_loss'] = total_kd_loss / len(loader)

        # Per-horizon student accuracy
        for h in range(num_horizons):
            h_acc = correct[h] / total if total > 0 else 0.0
            metrics[f'student_accuracy_horizon_{h}'] = h_acc

        # Per-horizon teacher accuracy
        if teacher_correct is not None:
            for h in range(num_horizons):
                t_acc = teacher_correct[h] / total if total > 0 else 0.0
                metrics[f'teacher_accuracy_horizon_{h}'] = t_acc

        # Overall accuracy (student)
        if total > 0:
            overall_acc = sum(correct) / (total * num_horizons)
            metrics['accuracy'] = overall_acc
            # Overall teacher accuracy
            if teacher_correct is not None:
                overall_teacher_acc = sum(teacher_correct) / (total * num_horizons)
                metrics['teacher_accuracy'] = overall_teacher_acc

    elif is_multi_horizon:
        # Add horizon-specific metrics
        for h in range(num_horizons):
            avg_h_loss = horizon_losses[h] / len(loader)
            h_acc = correct[h] / total if total > 0 else 0.0

            metrics[f'loss_horizon_{h}'] = avg_h_loss
            metrics[f'accuracy_horizon_{h}'] = h_acc

        # Also add overall accuracy (average of all horizons)
        if total > 0:
            overall_acc = sum(correct) / (total * num_horizons)
            metrics['accuracy'] = overall_acc
    else:
        # Single horizon
        if total > 0:
            metrics['accuracy'] = correct[0] / total

    # Log epoch metrics
    if log_enabled and writer:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_time', epoch_time, epoch)

        if use_distillation:
            writer.add_scalar('train/epoch_task_loss', metrics['task_loss'], epoch)
            writer.add_scalar('train/epoch_kd_loss', metrics['kd_loss'], epoch)
            for h in range(num_horizons):
                writer.add_scalar(f'train/epoch_student_accuracy_horizon_{h}', metrics[f'student_accuracy_horizon_{h}'], epoch)
                if f'teacher_accuracy_horizon_{h}' in metrics:
                    writer.add_scalar(f'train/epoch_teacher_accuracy_horizon_{h}', metrics[f'teacher_accuracy_horizon_{h}'], epoch)
            writer.add_scalar('train/epoch_student_accuracy_overall', metrics.get('accuracy', 0), epoch)
            if 'teacher_accuracy' in metrics:
                writer.add_scalar('train/epoch_teacher_accuracy_overall', metrics['teacher_accuracy'], epoch)
        elif is_multi_horizon:
            for h in range(num_horizons):
                writer.add_scalar(f'train/epoch_loss_horizon_{h}', metrics[f'loss_horizon_{h}'], epoch)
                writer.add_scalar(f'train/epoch_accuracy_horizon_{h}', metrics[f'accuracy_horizon_{h}'], epoch)
            writer.add_scalar('train/epoch_accuracy_overall', metrics.get('accuracy', 0), epoch)
        else:
            if 'accuracy' in metrics:
                writer.add_scalar('train/epoch_accuracy', metrics['accuracy'], epoch)

    return metrics


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader,
            loss_fn: nn.Module, device: torch.device,
            metrics: Dict[str, Callable] = None,
            compute_per_subject: bool = False) -> Dict[str, Any]:
    """
    Evaluate model on a dataset with separate transition and steady-state metrics.
    Now supports per-horizon transition flags for accurate steady/transition classification
    and per-subject evaluation.

    Args:
        model: Model to evaluate
        loader: Data loader
        loss_fn: Loss function (could be MultiHorizonLoss)
        device: Device to use for evaluation
        metrics: Dictionary of metric functions
        compute_per_subject: If True, compute and return per-subject metrics

    Returns:
        Dictionary of evaluation metrics (overall, steady-state, and transition-state for each horizon).
        If compute_per_subject=True, also includes 'per_subject_metrics' key with per-subject results.
    """
    from tqdm import tqdm

    model.eval()

    # Check if we're dealing with multi-horizon
    is_multi_horizon = isinstance(loss_fn, MultiHorizonLoss)
    num_horizons = loss_fn.num_heads if is_multi_horizon else 1

    print(f"Evaluation: is_multi_horizon={is_multi_horizon}, num_horizons={num_horizons}")

    total_loss = 0.0
    all_outputs = [[] for _ in range(num_horizons)] if is_multi_horizon else []
    all_labels = [[] for _ in range(num_horizons)] if is_multi_horizon else []
    all_transition_flags = [[] for _ in range(num_horizons)]  # Per-horizon transition flags
    all_subjects = []  # Track subject IDs for per-subject evaluation

    # Create progress bar
    pbar = tqdm(loader, desc="Evaluating", leave=False)

    batch_count = 0
    with torch.no_grad():
        for batch in pbar:
            batch_count += 1

            # Get inputs, labels, transition flags, and subjects
            # New batch format: (modality_data..., labels, transition_flags, subjects)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 4:
                    # New format with subjects
                    inputs = [x.to(device) for x in batch[:-3]]
                    labels = batch[-3]
                    transition_flags = batch[-2]  # List of tensors, one per horizon
                    subjects = batch[-1]  # List of subject ID strings
                elif len(batch) >= 3:
                    # Old format without subjects (backward compatibility)
                    inputs = [x.to(device) for x in batch[:-2]]
                    labels = batch[-2]
                    transition_flags = batch[-1]
                    batch_size = labels[0].size(0) if isinstance(labels, list) else labels.size(0)
                    subjects = ["unknown"] * batch_size
                else:
                    inputs = [x.to(device) for x in batch[:-1]]
                    labels = batch[-1]
                    # Create dummy transition flags if not provided
                    batch_size = labels[0].size(0) if isinstance(labels, list) else labels.size(0)
                    transition_flags = [torch.zeros(batch_size, dtype=torch.bool) for _ in range(num_horizons)]
                    subjects = ["unknown"] * batch_size

                # Handle single modality case
                if len(inputs) == 1:
                    inputs = inputs[0]
            else:
                raise ValueError("Loader should return a list or tuple of tensors")
            
            # Handle labels for multi-horizon
            if is_multi_horizon:
                if isinstance(labels, list) and len(labels) > 0:
                    if len(labels) != num_horizons:
                        print(f"Warning: Expected {num_horizons} horizons, got {len(labels)} in batch {batch_count}")
                        # Take only the number we expect
                        labels = labels[:num_horizons]
                    
                    # Move all horizon labels to device
                    horizon_labels = []
                    for h in range(min(len(labels), num_horizons)):
                        if isinstance(labels[h], torch.Tensor):
                            horizon_labels.append(labels[h].to(device))
                        else:
                            # Convert to tensor if not already
                            h_labels = torch.tensor(labels[h], dtype=torch.long).to(device)
                            horizon_labels.append(h_labels)
                    labels = horizon_labels
                else:
                    print(f"Warning: Multi-horizon model expects list of label tensors, got: {type(labels)} in batch {batch_count}")
                    continue  # Skip this batch
            else:
                # Single horizon - convert to tensor
                if isinstance(labels, list):
                    if len(labels) == 1 and isinstance(labels[0], torch.Tensor):
                        # Single horizon in list format
                        labels = labels[0].to(device)
                    elif len(labels) > 1:
                        # Multiple horizons but single horizon model - take first
                        labels = labels[0].to(device) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels[0], dtype=torch.long).to(device)
                    else:
                        # List of label values
                        labels = torch.tensor(labels, dtype=torch.long).to(device)
                else:
                    labels = labels.to(device)
            
            # Handle per-horizon transition flags
            if isinstance(transition_flags, list) and len(transition_flags) == num_horizons:
                # Per-horizon flags as expected
                pass
            elif isinstance(transition_flags, torch.Tensor):
                # Single flag tensor - replicate for all horizons (backward compatibility)
                transition_flags = [transition_flags for _ in range(num_horizons)]
            else:
                # Unexpected format - create dummy flags
                batch_size = labels[0].size(0) if is_multi_horizon else labels.size(0)
                transition_flags = [torch.zeros(batch_size, dtype=torch.bool) for _ in range(num_horizons)]
            
            # Forward pass
            if hasattr(model, 'modalities') and isinstance(inputs, (list, tuple)) and len(inputs) > 1:
                # For multimodal model
                modality_inputs = {model.modalities[i]: inputs[i] for i in range(len(inputs))}
                outputs = model(**modality_inputs)
            else:
                # For single modality models
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]  # Take the first (and only) modality
                outputs = model(inputs)
            
            # Debug output structure
            if batch_count == 1:
                print(f"Model outputs type: {type(outputs)}")
                if isinstance(outputs, list):
                    print(f"Model outputs length: {len(outputs)}")
                    for i, out in enumerate(outputs):
                        if isinstance(out, torch.Tensor):
                            print(f"  Output {i} shape: {out.shape}")
                        else:
                            print(f"  Output {i} type: {type(out)}")
            
            # Compute loss
            if is_multi_horizon:
                try:
                    loss_result = loss_fn(outputs, labels)
                    loss = loss_result['total_loss']
                except Exception as e:
                    print(f"Error computing multi-horizon loss in batch {batch_count}: {e}")
                    continue
            else:
                try:
                    # Single horizon: model returns list but we need single tensor
                    single_output = outputs[0] if isinstance(outputs, list) else outputs
                    loss = loss_fn(single_output, labels)
                except Exception as e:
                    print(f"Error computing single-horizon loss in batch {batch_count}: {e}")
                    continue
            
            # Save outputs, labels, per-horizon transition flags, and subjects for metric computation
            if is_multi_horizon:
                for h in range(len(outputs)):  # Use actual length of outputs
                    if h < num_horizons:  # Only save if within expected range
                        all_outputs[h].append(outputs[h])
                        if h < len(labels):
                            all_labels[h].append(labels[h])
                        else:
                            print(f"Warning: Missing label for horizon {h} in batch {batch_count}")

                        # Store per-horizon transition flags
                        if h < len(transition_flags):
                            all_transition_flags[h].append(transition_flags[h])
                        else:
                            # Create dummy flags if missing
                            all_transition_flags[h].append(torch.zeros(labels[0].size(0), dtype=torch.bool))
            else:
                # Single horizon: use first output
                single_output = outputs[0] if isinstance(outputs, list) else outputs
                all_outputs.append(single_output)
                all_labels.append(labels)
                # Store transition flags for single horizon
                all_transition_flags[0].append(transition_flags[0])

            # Track subjects for per-subject evaluation
            all_subjects.extend(subjects)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    print(f"Processed {batch_count} batches")
    
    # Check if we have any data
    if batch_count == 0:
        print("Warning: No batches processed in evaluation")
        return {'loss': 0.0}
    
    # Concatenate all outputs, labels, and per-horizon transition flags
    try:
        if is_multi_horizon:
            for h in range(num_horizons):
                if len(all_outputs[h]) == 0:
                    print(f"Warning: No outputs collected for horizon {h}")
                    return {'loss': total_loss / max(batch_count, 1)}
                
                print(f"Concatenating {len(all_outputs[h])} tensors for horizon {h}")
                all_outputs[h] = torch.cat(all_outputs[h], dim=0)
                
                if len(all_labels[h]) == 0:
                    print(f"Warning: No labels collected for horizon {h}")
                    return {'loss': total_loss / max(batch_count, 1)}
                
                all_labels[h] = torch.cat(all_labels[h], dim=0)
                
                # Concatenate per-horizon transition flags
                if len(all_transition_flags[h]) > 0:
                    all_transition_flags[h] = torch.cat(all_transition_flags[h], dim=0)
                else:
                    # Create dummy flags if missing
                    all_transition_flags[h] = torch.zeros(len(all_labels[h]), dtype=torch.bool)
        else:
            if len(all_outputs) == 0:
                print("Warning: No outputs collected")
                return {'loss': total_loss / max(batch_count, 1)}
                
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Concatenate transition flags for single horizon
            if len(all_transition_flags[0]) > 0:
                all_transition_flags[0] = torch.cat(all_transition_flags[0], dim=0)
            else:
                all_transition_flags[0] = torch.zeros(len(all_labels), dtype=torch.bool)
        
    except Exception as e:
        print(f"Error concatenating tensors: {e}")
        print(f"all_outputs structure: {[len(x) if isinstance(x, list) else 'not list' for x in all_outputs] if is_multi_horizon else len(all_outputs)}")
        return {'loss': total_loss / max(batch_count, 1)}
    
    # Compute overall metrics
    results = {'loss': total_loss / len(loader)}
    
    # Get class names from dataset if available
    class_names = None
    if hasattr(loader.dataset, 'label_mapping'):
        try:
            idx_to_label = loader.dataset.label_mapping.get("idx_to_label", {})
            if idx_to_label:
                sorted_items = sorted([(int(idx), label) for idx, label in idx_to_label.items()])
                class_names = [label for _, label in sorted_items]
                logging.debug(f"Found {len(class_names)} class names from dataset: {class_names}")
        except Exception as e:
            logging.warning(f"Error getting class names from dataset: {e}")
    
    # Process each horizon with its specific transition flags
    for h in range(num_horizons):
        # Get horizon suffix for metric names
        h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
        
        # Get outputs and labels for this horizon
        if is_multi_horizon:
            h_outputs = all_outputs[h]
            h_labels = all_labels[h]
        else:
            h_outputs = all_outputs
            h_labels = all_labels
        
        # Get horizon-specific transition flags
        h_transition_flags = all_transition_flags[h]
        
        # Calculate accuracy for classification
        if h_outputs.shape[1] > 1:
            _, predicted = torch.max(h_outputs, 1)
            correct = (predicted == h_labels).sum().item()
            total = h_labels.size(0)
            results[f'accuracy{h_suffix}'] = correct / total
        
        # Separate data for transition and steady-state analysis using horizon-specific flags
        steady_mask = ~h_transition_flags
        transition_mask = h_transition_flags
        
        steady_count = steady_mask.sum().item()
        transition_count = transition_mask.sum().item()
        
        if h == 0:  # Log this info only once for first horizon
            logging.info(f"Evaluation samples - Total: {len(h_labels)}, Steady-state: {steady_count}, Transition: {transition_count}")
        else:
            logging.info(f"Horizon {h} - Steady-state: {steady_count}, Transition: {transition_count}")
        
        # Compute additional metrics for all three categories: overall, steady-state, transition
        if metrics:
            # Convert to CPU for metric computation
            outputs_np = h_outputs.cpu().numpy()
            labels_np = h_labels.cpu().numpy()
            transition_flags_np = h_transition_flags.cpu().numpy()
            
            # Separate data for steady-state and transition analysis
            steady_outputs = outputs_np[~transition_flags_np] if steady_count > 0 else np.array([]).reshape(0, outputs_np.shape[1])
            steady_labels = labels_np[~transition_flags_np] if steady_count > 0 else np.array([])
            
            transition_outputs = outputs_np[transition_flags_np] if transition_count > 0 else np.array([]).reshape(0, outputs_np.shape[1])
            transition_labels = labels_np[transition_flags_np] if transition_count > 0 else np.array([])
            
            for name, metric_fn in metrics.items():
                try:
                    # Compute metrics for all three categories
                    metric_results = {}
                    
                    # 1. Overall metrics
                    if hasattr(metric_fn, '__call__'):
                        if name.lower() in ['classificationreport', 'confusionmatrix'] and class_names is not None:
                            overall_result = metric_fn(outputs_np, labels_np, class_names=class_names)
                        else:
                            overall_result = metric_fn(outputs_np, labels_np)
                        metric_results['overall'] = overall_result
                    
                    # 2. Steady-state metrics
                    if steady_count > 0:
                        if name.lower() in ['classificationreport', 'confusionmatrix'] and class_names is not None:
                            steady_result = metric_fn(steady_outputs, steady_labels, class_names=class_names)
                        else:
                            steady_result = metric_fn(steady_outputs, steady_labels)
                        metric_results['steady'] = steady_result
                    else:
                        metric_results['steady'] = None
                    
                    # 3. Transition-state metrics
                    if transition_count > 0:
                        if name.lower() in ['classificationreport', 'confusionmatrix'] and class_names is not None:
                            transition_result = metric_fn(transition_outputs, transition_labels, class_names=class_names)
                        else:
                            transition_result = metric_fn(transition_outputs, transition_labels)
                        metric_results['transition'] = transition_result
                    else:
                        metric_results['transition'] = None
                    
                    # Process results for each category
                    for category, result in metric_results.items():
                        if result is None:
                            continue
                            
                        category_prefix = f"{category}_" if category != 'overall' else ""
                        
                        # Handle different return types
                        if isinstance(result, dict):
                            if name.lower() == 'classificationreport':
                                for key, value in result.items():
                                    results[f"{category_prefix}{name}_{key}{h_suffix}"] = value

                                if 'report' in result and isinstance(result['report'], dict):
                                    report_dict = result['report']
                                    if 'macro avg' in report_dict:
                                        for metric_key, metric_val in report_dict['macro avg'].items():
                                            if isinstance(metric_val, (int, float)):
                                                results[f"{category_prefix}{name}_macro_{metric_key}{h_suffix}"] = float(metric_val)
                                    if 'weighted avg' in report_dict:
                                        for metric_key, metric_val in report_dict['weighted avg'].items():
                                            if isinstance(metric_val, (int, float)):
                                                results[f"{category_prefix}{name}_weighted_{metric_key}{h_suffix}"] = float(metric_val)
                            elif name.lower() == 'confusionmatrix':
                                # Handle confusion matrix with dataframe
                                if 'dataframe' in result:
                                    results[f"{category_prefix}{name}_dataframe{h_suffix}"] = result['dataframe']
                                if 'metrics' in result:
                                    for key, value in result['metrics'].items():
                                        if isinstance(value, (int, float)):
                                            results[f"{category_prefix}{name}_{key}{h_suffix}"] = float(value)
                            else:
                                for key, value in result.items():
                                    if isinstance(value, (int, float)):
                                        results[f"{category_prefix}{name}_{key}{h_suffix}"] = float(value)
                        else:
                            results[f"{category_prefix}{name}{h_suffix}"] = float(result)
                
                except Exception as e:
                    print(f"Error computing metric {name} for horizon {h}: {e}")
                    import traceback
                    traceback.print_exc()

    # Compute per-subject metrics if requested
    if compute_per_subject and metrics:
        per_subject_metrics = _compute_per_subject_metrics(
            all_outputs=all_outputs,
            all_labels=all_labels,
            all_transition_flags=all_transition_flags,
            all_subjects=all_subjects,
            metrics=metrics,
            class_names=class_names,
            num_horizons=num_horizons,
            is_multi_horizon=is_multi_horizon
        )
        results['per_subject_metrics'] = per_subject_metrics

    return results


def _compute_per_subject_metrics(
    all_outputs,
    all_labels,
    all_transition_flags,
    all_subjects: List[str],
    metrics: Dict[str, Callable],
    class_names: List[str],
    num_horizons: int,
    is_multi_horizon: bool
) -> Dict[str, Dict]:
    """
    Compute metrics for each subject separately.

    Args:
        all_outputs: Concatenated outputs (list of tensors for multi-horizon, single tensor otherwise)
        all_labels: Concatenated labels (list of tensors for multi-horizon, single tensor otherwise)
        all_transition_flags: Per-horizon transition flags
        all_subjects: List of subject IDs for each sample
        metrics: Dictionary of metric functions
        class_names: List of class names
        num_horizons: Number of prediction horizons
        is_multi_horizon: Whether model is multi-horizon

    Returns:
        Dictionary mapping subject IDs to their metrics per horizon and condition
    """
    import numpy as np

    # Convert subjects to numpy array for indexing
    subjects_array = np.array(all_subjects)
    unique_subjects = sorted(set(all_subjects))

    print(f"Computing per-subject metrics for {len(unique_subjects)} subjects: {unique_subjects}")

    per_subject_results = {}

    for subject in unique_subjects:
        subject_mask = subjects_array == subject
        subject_results = {}

        for h in range(num_horizons):
            h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""

            # Get data for this horizon
            if is_multi_horizon:
                h_outputs = all_outputs[h].cpu().numpy()
                h_labels = all_labels[h].cpu().numpy()
            else:
                h_outputs = all_outputs.cpu().numpy()
                h_labels = all_labels.cpu().numpy()

            h_transition_flags = all_transition_flags[h].cpu().numpy()

            # Filter for this subject
            subj_outputs = h_outputs[subject_mask]
            subj_labels = h_labels[subject_mask]
            subj_transition_flags = h_transition_flags[subject_mask]

            if len(subj_labels) == 0:
                print(f"Warning: No samples for subject {subject} at horizon {h}")
                continue

            # Compute accuracy
            predicted = np.argmax(subj_outputs, axis=1)
            correct = (predicted == subj_labels).sum()
            total = len(subj_labels)
            subject_results[f'accuracy{h_suffix}'] = correct / total

            # Separate steady and transition
            steady_mask = ~subj_transition_flags
            transition_mask = subj_transition_flags

            steady_count = steady_mask.sum()
            transition_count = transition_mask.sum()

            # Compute metrics for each category
            for name, metric_fn in metrics.items():
                try:
                    # Overall metrics for this subject
                    if name.lower() in ['classificationreport', 'confusionmatrix'] and class_names is not None:
                        overall_result = metric_fn(subj_outputs, subj_labels, class_names=class_names)
                    else:
                        overall_result = metric_fn(subj_outputs, subj_labels)

                    # Store overall metrics
                    _store_metric_result(subject_results, name, overall_result, "", h_suffix)

                    # Steady-state metrics
                    if steady_count > 0:
                        steady_outputs = subj_outputs[steady_mask]
                        steady_labels = subj_labels[steady_mask]
                        if name.lower() in ['classificationreport', 'confusionmatrix'] and class_names is not None:
                            steady_result = metric_fn(steady_outputs, steady_labels, class_names=class_names)
                        else:
                            steady_result = metric_fn(steady_outputs, steady_labels)
                        _store_metric_result(subject_results, name, steady_result, "steady_", h_suffix)

                    # Transition metrics
                    if transition_count > 0:
                        transition_outputs = subj_outputs[transition_mask]
                        transition_labels = subj_labels[transition_mask]
                        if name.lower() in ['classificationreport', 'confusionmatrix'] and class_names is not None:
                            transition_result = metric_fn(transition_outputs, transition_labels, class_names=class_names)
                        else:
                            transition_result = metric_fn(transition_outputs, transition_labels)
                        _store_metric_result(subject_results, name, transition_result, "transition_", h_suffix)

                except Exception as e:
                    print(f"Error computing metric {name} for subject {subject}, horizon {h}: {e}")

        per_subject_results[subject] = subject_results

    return per_subject_results


def _store_metric_result(results_dict: Dict, metric_name: str, result: Any,
                         category_prefix: str, h_suffix: str):
    """Helper to store metric results in the appropriate format."""
    if isinstance(result, dict):
        if metric_name.lower() == 'classificationreport':
            for key, value in result.items():
                results_dict[f"{category_prefix}{metric_name}_{key}{h_suffix}"] = value

            if 'report' in result and isinstance(result['report'], dict):
                report_dict = result['report']
                if 'macro avg' in report_dict:
                    for metric_key, metric_val in report_dict['macro avg'].items():
                        if isinstance(metric_val, (int, float)):
                            results_dict[f"{category_prefix}{metric_name}_macro_{metric_key}{h_suffix}"] = float(metric_val)
                if 'weighted avg' in report_dict:
                    for metric_key, metric_val in report_dict['weighted avg'].items():
                        if isinstance(metric_val, (int, float)):
                            results_dict[f"{category_prefix}{metric_name}_weighted_{metric_key}{h_suffix}"] = float(metric_val)
        elif metric_name.lower() == 'confusionmatrix':
            if 'dataframe' in result:
                results_dict[f"{category_prefix}{metric_name}_dataframe{h_suffix}"] = result['dataframe']
            if 'metrics' in result:
                for key, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        results_dict[f"{category_prefix}{metric_name}_{key}{h_suffix}"] = float(value)
        else:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    results_dict[f"{category_prefix}{metric_name}_{key}{h_suffix}"] = float(value)
    elif isinstance(result, (int, float)):
        results_dict[f"{category_prefix}{metric_name}{h_suffix}"] = float(result)



def save_checkpoint(checkpoint_dir: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: Any, epoch: int, metrics: Dict[str, float], 
                   prediction_horizons: List[float] = None,
                   is_best: bool = False, keep_last_n: int = 1) -> str:
    """
    Save model checkpoint with separate folders for each horizon and time-based naming.
    Now supports multi-horizon models with cleaner organization.
    
    Args:
        checkpoint_dir: Directory to save checkpoint in
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch
        metrics: Dictionary of metrics
        prediction_horizons: List of prediction horizon values in seconds
        is_best: Whether this is the best model so far
        keep_last_n: Number of last checkpoints to keep
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'prediction_horizons': prediction_horizons
    }
    
    # Add scheduler state if exists
    if scheduler is not None:
        if hasattr(scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        else:
            # Scheduler might be a config dict or None
            checkpoint['scheduler_state_dict'] = None
    
    # Save main checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Determine how many horizons we have and get horizon values
    num_horizons = 1
    horizon_values = [0.0]  # Default for single horizon
    
    # Get prediction horizons from model or parameter
    if prediction_horizons is not None:
        horizon_values = prediction_horizons
        num_horizons = len(horizon_values)
    elif hasattr(model, 'get_prediction_horizons'):
        horizon_values = model.get_prediction_horizons()
        num_horizons = len(horizon_values)
    else:
        # Try to infer from metrics
        for key in metrics.keys():
            if "_horizon_" in key:
                try:
                    h_num = int(key.split("_horizon_")[1].split("_")[0]) + 1
                    num_horizons = max(num_horizons, h_num)
                except:
                    pass
        if num_horizons > 1:
            horizon_values = list(range(num_horizons))  # Fallback to indices
    
    # Save detailed metrics CSV files for each category and horizon
    categories = ['overall', 'steady', 'transition']
    saved_csv_files = []

    for h in range(num_horizons):
        h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
        horizon_time = horizon_values[h]
        horizon_str = f"{horizon_time:.1f}s"

        # Create horizon-specific directory
        horizon_dir = os.path.join(checkpoint_dir, f"horizon_{horizon_str}")
        os.makedirs(horizon_dir, exist_ok=True)

        for category in categories:
            category_suffix = f'_{category}' if category != 'overall' else ''
            category_prefix = f'{category}_' if category != 'overall' else ''

            # Save classification report CSV
            classreport_key = f'{category_prefix}classificationreport_dataframe{h_suffix}'
            if classreport_key in metrics:
                try:
                    # Save epoch-specific file
                    csv_filename = f'metrics_report{category_suffix}_epoch_{epoch}.csv'
                    csv_path = os.path.join(horizon_dir, csv_filename)
                    metrics[classreport_key].to_csv(csv_path, index=False)
                    logging.info(f"Saved {category} horizon {horizon_str} metrics report: {csv_path}")
                    saved_csv_files.append(csv_path)

                    # Save a copy if this is the best model
                    if is_best:
                        best_csv_filename = f'best_model_metrics{category_suffix}.csv'
                        best_csv_path = os.path.join(horizon_dir, best_csv_filename)
                        metrics[classreport_key].to_csv(best_csv_path, index=False)
                        logging.info(f"Saved best {category} horizon {horizon_str} model metrics: {best_csv_path}")

                except Exception as e:
                    logging.warning(f"Error saving {category} horizon {horizon_str} classification report: {e}")

            # Save confusion matrix CSV only for best model
            confmat_key = f'{category_prefix}confusionmatrix_dataframe{h_suffix}'
            if confmat_key in metrics and is_best:
                try:
                    best_cm_filename = f'best_model_confusion_matrix{category_suffix}.csv'
                    best_cm_path = os.path.join(horizon_dir, best_cm_filename)
                    metrics[confmat_key].to_csv(best_cm_path, index=False)
                    logging.info(f"Saved best {category} horizon {horizon_str} confusion matrix: {best_cm_path}")
                    saved_csv_files.append(best_cm_path)

                except Exception as e:
                    logging.warning(f"Error saving {category} horizon {horizon_str} confusion matrix: {e}")
    
    # Clean up old checkpoints and associated files
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                    if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            old_ckpt_path = os.path.join(checkpoint_dir, old_ckpt)
            os.remove(old_ckpt_path)
            logging.info(f"Removed old checkpoint: {old_ckpt}")
            
            # Also remove associated metrics reports from horizon directories
            epoch_num = old_ckpt.split('_')[2].split('.')[0]
            for h in range(num_horizons):
                horizon_time = horizon_values[h]
                horizon_str = f"{horizon_time:.1f}s"
                horizon_dir = os.path.join(checkpoint_dir, f"horizon_{horizon_str}")

                if os.path.exists(horizon_dir):
                    for category in categories:
                        category_suffix = f'_{category}' if category != 'overall' else ''

                        # Remove old classification report
                        old_csv = f'metrics_report{category_suffix}_epoch_{epoch_num}.csv'
                        old_csv_path = os.path.join(horizon_dir, old_csv)
                        if os.path.exists(old_csv_path):
                            os.remove(old_csv_path)
                            logging.info(f"Removed old {category} horizon {horizon_str} metrics report: {old_csv}")

    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best model checkpoint: {best_path}")

    return checkpoint_path


def save_per_subject_results(
    output_dir: str,
    per_subject_metrics: Dict[str, Dict],
    prediction_horizons: List[float] = None,
    prefix: str = "test"
) -> Dict[str, str]:
    """
    Save per-subject evaluation results to CSV files.

    Creates a structure similar to LOSO results:
    - output_dir/
        - per_subject_results/
            - subject_{subject_id}/
                - horizon_{time}s/
                    - metrics.csv
                    - metrics_steady.csv
                    - metrics_transition.csv
                    - confusion_matrix.csv
                    - confusion_matrix_steady.csv
                    - confusion_matrix_transition.csv
            - {prefix}_subject_results.csv (summary across all subjects)
            - {prefix}_subject_results_horizon_{time}s.csv (per-horizon summary)

    Args:
        output_dir: Directory to save results
        per_subject_metrics: Dictionary mapping subject IDs to their metrics
        prediction_horizons: List of prediction horizon values in seconds
        prefix: Prefix for output files (e.g., "test", "val")

    Returns:
        Dictionary with paths to saved files
    """
    import pandas as pd
    import shutil

    if not per_subject_metrics:
        logging.warning("No per-subject metrics to save")
        return {}

    # Create per-subject results directory (clean up old results first)
    per_subject_dir = os.path.join(output_dir, "per_subject_results")
    if os.path.exists(per_subject_dir):
        # Remove old per-subject results to avoid stale files
        shutil.rmtree(per_subject_dir)
        logging.info(f"Cleaned up old per-subject results: {per_subject_dir}")
    os.makedirs(per_subject_dir, exist_ok=True)

    saved_files = {}

    # Determine horizons
    if prediction_horizons is None:
        prediction_horizons = [0.0]
    num_horizons = len(prediction_horizons)

    # Save individual subject results
    for subject_id, subject_metrics in per_subject_metrics.items():
        subject_dir = os.path.join(per_subject_dir, f"subject_{subject_id}")
        os.makedirs(subject_dir, exist_ok=True)

        for h, horizon_time in enumerate(prediction_horizons):
            h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
            horizon_str = f"{horizon_time:.1f}s"
            horizon_dir = os.path.join(subject_dir, f"horizon_{horizon_str}")
            os.makedirs(horizon_dir, exist_ok=True)

            # Save metrics for each category (overall, steady, transition)
            for category in ['', 'steady_', 'transition_']:
                category_suffix = f"_{category.rstrip('_')}" if category else ""

                # Save classification report DataFrame if available
                # Note: metric names are stored in lowercase (e.g., 'classificationreport')
                classreport_key = f'{category}classificationreport_dataframe{h_suffix}'
                if classreport_key in subject_metrics:
                    try:
                        df = subject_metrics[classreport_key]
                        if isinstance(df, pd.DataFrame):
                            csv_filename = f'best_model_metrics{category_suffix}.csv'
                            csv_path = os.path.join(horizon_dir, csv_filename)
                            df.to_csv(csv_path, index=False)
                            saved_files[f"{subject_id}_{horizon_str}{category_suffix}_metrics"] = csv_path
                    except Exception as e:
                        logging.warning(f"Error saving {subject_id} {category}metrics: {e}")

                # Save confusion matrix DataFrame if available
                confmat_key = f'{category}confusionmatrix_dataframe{h_suffix}'
                if confmat_key in subject_metrics:
                    try:
                        df = subject_metrics[confmat_key]
                        if isinstance(df, pd.DataFrame):
                            csv_filename = f'best_model_confusion_matrix{category_suffix}.csv'
                            csv_path = os.path.join(horizon_dir, csv_filename)
                            df.to_csv(csv_path, index=False)
                            saved_files[f"{subject_id}_{horizon_str}{category_suffix}_confmat"] = csv_path
                    except Exception as e:
                        logging.warning(f"Error saving {subject_id} {category}confusion matrix: {e}")

    # Create summary DataFrame across all subjects (like loso_subject_results.csv)
    summary_rows = []
    for subject_id, subject_metrics in per_subject_metrics.items():
        row = {'subject': subject_id}

        # Extract key metrics for each horizon
        for h, horizon_time in enumerate(prediction_horizons):
            h_suffix = f"_horizon_{h}" if num_horizons > 1 else ""
            horizon_str = f"{horizon_time:.1f}s"

            # Get accuracy
            acc_key = f'accuracy{h_suffix}'
            if acc_key in subject_metrics:
                if num_horizons > 1:
                    row[f'accuracy_{horizon_str}'] = subject_metrics[acc_key]
                else:
                    row['accuracy'] = subject_metrics[acc_key]

            # Get macro F1 from classification report (metric names are lowercase)
            macro_f1_key = f'classificationreport_macro_f1-score{h_suffix}'
            if macro_f1_key in subject_metrics:
                if num_horizons > 1:
                    row[f'macro_f1_{horizon_str}'] = subject_metrics[macro_f1_key]
                else:
                    row['macro_f1'] = subject_metrics[macro_f1_key]

            # Get weighted F1
            weighted_f1_key = f'classificationreport_weighted_f1-score{h_suffix}'
            if weighted_f1_key in subject_metrics:
                if num_horizons > 1:
                    row[f'weighted_f1_{horizon_str}'] = subject_metrics[weighted_f1_key]
                else:
                    row['weighted_f1'] = subject_metrics[weighted_f1_key]

            # Steady state metrics
            steady_macro_f1_key = f'steady_classificationreport_macro_f1-score{h_suffix}'
            if steady_macro_f1_key in subject_metrics:
                if num_horizons > 1:
                    row[f'steady_macro_f1_{horizon_str}'] = subject_metrics[steady_macro_f1_key]
                else:
                    row['steady_macro_f1'] = subject_metrics[steady_macro_f1_key]

            # Transition metrics
            transition_macro_f1_key = f'transition_classificationreport_macro_f1-score{h_suffix}'
            if transition_macro_f1_key in subject_metrics:
                if num_horizons > 1:
                    row[f'transition_macro_f1_{horizon_str}'] = subject_metrics[transition_macro_f1_key]
                else:
                    row['transition_macro_f1'] = subject_metrics[transition_macro_f1_key]

        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        # Skip per-subject analysis if only one subject (e.g., LOSO experiments)
        # In LOSO, per-subject analysis is done at the fold aggregation level
        if len(summary_rows) == 1:
            logging.info("Only one subject in evaluation set - skipping per-subject summary "
                        "(per-subject analysis is done at LOSO fold aggregation level)")
            return saved_files

        # Add mean and std rows (only meaningful with multiple subjects)
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        mean_row = {'subject': 'MEAN'}
        std_row = {'subject': 'STD'}
        for col in numeric_cols:
            mean_row[col] = summary_df[col].mean()
            # Use ddof=0 to get 0 for single subject instead of NaN
            std_row[col] = summary_df[col].std(ddof=0) if len(summary_df) > 1 else 0.0
        summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        # Save overall summary
        summary_path = os.path.join(per_subject_dir, f'{prefix}_subject_results.csv')
        summary_df.to_csv(summary_path, index=False)
        saved_files['summary'] = summary_path
        logging.info(f"Saved per-subject summary: {summary_path}")

        # Save per-horizon summaries if multi-horizon
        if num_horizons > 1:
            for h, horizon_time in enumerate(prediction_horizons):
                horizon_str = f"{horizon_time:.1f}s"
                horizon_cols = ['subject'] + [col for col in summary_df.columns
                                              if col == 'subject' or horizon_str in col]
                if len(horizon_cols) > 1:
                    horizon_df = summary_df[horizon_cols].copy()
                    # Rename columns to remove horizon suffix
                    horizon_df.columns = [col.replace(f'_{horizon_str}', '') for col in horizon_df.columns]
                    horizon_path = os.path.join(per_subject_dir, f'{prefix}_subject_results_horizon_{horizon_str}.csv')
                    horizon_df.to_csv(horizon_path, index=False)
                    saved_files[f'summary_horizon_{horizon_str}'] = horizon_path
                    logging.info(f"Saved horizon {horizon_str} summary: {horizon_path}")

    return saved_files


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing loaded checkpoint information
    """
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint