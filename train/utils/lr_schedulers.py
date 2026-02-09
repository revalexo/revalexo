# utils/lr_schedulers.py
import math
import torch
from torch.optim.lr_scheduler import (
    LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, 
    CosineAnnealingWarmRestarts, OneCycleLR, ExponentialLR, PolynomialLR
)
from typing import Dict, Any, Optional, Union, Callable

class WarmupWrapper:
    """Wrapper for learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, scheduler, warmup_steps, warmup_start_lr=None):
        """
        Initialize warmup wrapper for scheduler.
        
        Args:
            optimizer: Optimizer to wrap
            scheduler: Learning rate scheduler 
            warmup_steps: Number of warmup steps
            warmup_start_lr: Starting learning rate for warmup, if None uses optimizer lr * 0.01
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Get base learning rates from optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Set warmup start learning rate
        if warmup_start_lr is None:
            self.warmup_start_lrs = [lr * 0.01 for lr in self.base_lrs]
        else:
            self.warmup_start_lrs = [warmup_start_lr for _ in self.base_lrs]
            
        # Set initial learning rate to warmup start learning rate
        for param_group, lr in zip(optimizer.param_groups, self.warmup_start_lrs):
            param_group['lr'] = lr
            
    def step(self, metrics=None):
        """Step the learning rate scheduler with warmup."""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            progress = float(self.step_count) / float(max(1, self.warmup_steps))
            for param_group, start_lr, base_lr in zip(
                self.optimizer.param_groups, self.warmup_start_lrs, self.base_lrs
            ):
                param_group['lr'] = start_lr + progress * (base_lr - start_lr)
        else:
            # After warmup, use the scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if metrics is None:
                    raise ValueError("metrics must be provided for ReduceLROnPlateau")
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()
                
    def get_last_lr(self):
        """Get last learning rates."""
        if self.step_count <= self.warmup_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        # ReduceLROnPlateau doesn't have get_last_lr(), fall back to reading from optimizer
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]


def create_scheduler(config: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Any:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        config: Configuration dictionary
        optimizer: Optimizer to use with scheduler
        
    Returns:
        Learning rate scheduler
    """
    training_config = config.get('training', {})
    scheduler_config = training_config.get('lr_scheduler', {})
    warmup_config = training_config.get('warmup', {})
    
    if not scheduler_config:
        return None
        
    scheduler_name = scheduler_config.get('name')
    scheduler_params = scheduler_config.get('params', {})
    
    # Total epochs
    epochs = training_config.get('epochs', 100)
    
    # Create basic scheduler based on name
    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, **scheduler_params)
        
    elif scheduler_name == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, **scheduler_params)
        
    elif scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, **scheduler_params)
        
    elif scheduler_name == 'CosineAnnealingLR':
        # Default T_max to number of epochs if not provided
        if 'T_max' not in scheduler_params:
            scheduler_params['T_max'] = epochs
        scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
        
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
        
    elif scheduler_name == 'OneCycleLR':
        # If max_lr not provided, use 10x current learning rate
        if 'max_lr' not in scheduler_params:
            base_lr = optimizer.param_groups[0]['lr']
            scheduler_params['max_lr'] = base_lr * 10.0
        
        # If total_steps not provided, calculate from epochs
        if 'total_steps' not in scheduler_params:
            # This will be updated later when we know steps_per_epoch
            scheduler_params['total_steps'] = epochs
            
        scheduler = OneCycleLR(optimizer, **scheduler_params)
        
    elif scheduler_name == 'PolynomialLR':
        # If total_iters not provided, use number of epochs
        if 'total_iters' not in scheduler_params:
            scheduler_params['total_iters'] = epochs
        scheduler = PolynomialLR(optimizer, **scheduler_params)
        
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
        
    elif scheduler_name == 'CustomCosineWithWarmup':
        # A flag for a special cosine schedule with warmup built-in
        # This will be handled differently later
        scheduler = {
            'name': 'CustomCosineWithWarmup',
            'epochs': epochs,
            'warmup_epochs': warmup_config.get('epochs', 0),
            'warmup_ratio': warmup_config.get('ratio', 0.01),
            'min_lr': scheduler_params.get('min_lr', 0.0)
        }
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    # Apply warmup if configured
    if warmup_config.get('enabled', False) and not isinstance(scheduler, dict):
        warmup_epochs = warmup_config.get('epochs', 0)
        warmup_ratio = warmup_config.get('ratio', 0.01)
        base_lr = optimizer.param_groups[0]['lr']
        warmup_start_lr = base_lr * warmup_ratio
        
        # Store scheduler details for later initialization once we know steps_per_epoch
        scheduler = {
            'scheduler': scheduler,
            'warmup_epochs': warmup_epochs,
            'warmup_start_lr': warmup_start_lr,
            'is_batch_level': scheduler_name in ('OneCycleLR',),
            'is_validation_based': scheduler_name in ('ReduceLROnPlateau',)
        }
        
    return scheduler


def finalize_scheduler(scheduler_config, optimizer, steps_per_epoch):
    """
    Finalize scheduler configuration once steps_per_epoch is known.
    
    Args:
        scheduler_config: Scheduler configuration dictionary or scheduler instance
        optimizer: Optimizer to use
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        Finalized scheduler
    """
    if not scheduler_config:
        return None
        
    if not isinstance(scheduler_config, dict):
        # Already a scheduler instance
        return scheduler_config
    
    if scheduler_config.get('name') == 'CustomCosineWithWarmup':
        # Create custom cosine scheduler with warmup
        epochs = scheduler_config['epochs']
        warmup_epochs = scheduler_config['warmup_epochs']
        warmup_ratio = scheduler_config['warmup_ratio']
        min_lr = scheduler_config['min_lr']
        
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = epochs * steps_per_epoch

        # Capture initial LR before LambdaLR modifies it
        initial_lr = optimizer.param_groups[0]['lr']

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return warmup_ratio + (1.0 - warmup_ratio) * (float(current_step) / float(max(1, warmup_steps)))

            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            # Scale between 1.0 and min_lr/initial_lr
            min_lr_ratio = min_lr / initial_lr if initial_lr > 0 else 0.0
            factor = cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

            return factor
            
        return LambdaLR(optimizer, lr_lambda)
    
    # For standard scheduler with warmup wrapper
    scheduler = scheduler_config['scheduler']
    warmup_epochs = scheduler_config['warmup_epochs']
    warmup_start_lr = scheduler_config['warmup_start_lr']
    
    # For OneCycleLR, update total_steps if needed
    if isinstance(scheduler, OneCycleLR):
        # TODO: Fix get epochs, currently not using OneCycleLR 
        scheduler.total_steps = steps_per_epoch * epochs
    
    # For batch-level schedulers like OneCycleLR
    if scheduler_config.get('is_batch_level', False):
        # These are stepped every batch, no warmup wrapper needed
        return scheduler
    
    # For validation-based schedulers like ReduceLROnPlateau or epoch-level schedulers
    warmup_steps = warmup_epochs * steps_per_epoch
    if warmup_steps > 0:
        return WarmupWrapper(optimizer, scheduler, warmup_steps, warmup_start_lr)
    
    return scheduler