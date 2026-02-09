# utils/model_utils.py

import importlib
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union
from utils.config_utils import load_config
import json
import os
import yaml
import logging


def load_pretrained_encoder(model: nn.Module, checkpoint_path: str, model_name: str) -> None:
    """
    Load pretrained encoder weights into a model, skipping incompatible layers.
    
    Args:
        model: Target model to load weights into
        checkpoint_path: Path to pretrained checkpoint
        model_name: Name of the model for logging
    """
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Pretrained checkpoint not found at {checkpoint_path}")
        return
    
    logging.info(f"Loading pretrained encoder from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
    else:
        pretrained_state_dict = checkpoint
    
    # Get current model state dict
    model_state_dict = model.state_dict()
    
    # Filter out incompatible keys
    filtered_state_dict = {}
    loaded_keys = []
    skipped_keys = []
    
    for key, value in pretrained_state_dict.items():
        if key in model_state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
        else:
            # Skip classifier layers
            if 'classifier' not in key and 'fc' not in key:
                skipped_keys.append(f"{key} (not in target model)")
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Log what was loaded
    logging.info(f"Loaded {len(loaded_keys)} layers for {model_name}")
    if loaded_keys:
        logging.info(f"Loaded layers: {', '.join(loaded_keys[:5])}{'...' if len(loaded_keys) > 5 else ''}")
    
    if skipped_keys:
        logging.info(f"Skipped {len(skipped_keys)} incompatible layers")
        for key in skipped_keys[:5]:  # Show first 5 skipped
            logging.info(f"  Skipped: {key}")
        if len(skipped_keys) > 5:
            logging.info(f"  ... and {len(skipped_keys) - 5} more")
    
    # Verify encoder layers were loaded
    encoder_layers_loaded = any('conv' in k.lower() or 'lstm' in k.lower() or 
                                'backbone' in k.lower() or 'layer' in k.lower() or
                                'features' in k.lower() for k in loaded_keys)
    if encoder_layers_loaded:
        logging.info(f"Successfully loaded pretrained encoder for {model_name}")
    else:
        logging.warning(f"No encoder layers were loaded for {model_name}")


def load_pretrained_fusion_model(fusion_model: nn.Module, checkpoint_path: str) -> None:
    """
    Load pretrained weights for a fusion model, handling the nested encoder structure.
    
    Args:
        fusion_model: Fusion model to load weights into
        checkpoint_path: Path to pretrained checkpoint
    """
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Fusion model checkpoint not found at {checkpoint_path}")
        return
    
    logging.info(f"Loading pretrained fusion model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
    else:
        pretrained_state_dict = checkpoint
    
    # Get current model state dict
    model_state_dict = fusion_model.state_dict()
    
    # Track what we load
    loaded_encoders = {modality: [] for modality in fusion_model.modalities if hasattr(fusion_model, 'modalities')}
    loaded_fusion = []
    skipped_keys = []
    
    # Filter and load compatible weights
    filtered_state_dict = {}
    
    for key, value in pretrained_state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
                
                # Track which component this belongs to
                if 'modality_encoders' in key:
                    # Extract modality name from key
                    for modality in loaded_encoders.keys():
                        if f'modality_encoders.{modality}' in key:
                            loaded_encoders[modality].append(key)
                            break
                else:
                    loaded_fusion.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
        else:
            skipped_keys.append(f"{key} (not in target model)")
    
    # Load the filtered state dict
    fusion_model.load_state_dict(filtered_state_dict, strict=False)
    
    # Log what was loaded
    logging.info(f"Loaded {len(filtered_state_dict)} total layers for fusion model")
    
    for modality, keys in loaded_encoders.items():
        if keys:
            logging.info(f"  {modality} encoder: {len(keys)} layers loaded")
    
    if loaded_fusion:
        logging.info(f"  Fusion layers: {len(loaded_fusion)} layers loaded")
    
    if skipped_keys:
        logging.info(f"Skipped {len(skipped_keys)} incompatible layers")
        for key in skipped_keys[:5]:
            logging.info(f"  Skipped: {key}")
        if len(skipped_keys) > 5:
            logging.info(f"  ... and {len(skipped_keys) - 5} more")


def resolve_checkpoint_path(checkpoint_path: str, config_dir: str = None) -> str:
    """
    Resolve checkpoint path, trying multiple locations.
    
    Args:
        checkpoint_path: Path from config (may be relative)
        config_dir: Directory of config file for relative path resolution
        
    Returns:
        Resolved absolute path if found, otherwise original path
    """
    # If already absolute and exists, return it
    if os.path.isabs(checkpoint_path) and os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Try relative to current working directory
    cwd_relative = os.path.join(os.getcwd(), checkpoint_path)
    if os.path.exists(cwd_relative):
        return cwd_relative

    # Try relative to config directory
    if config_dir:
        config_relative = os.path.join(config_dir, checkpoint_path)
        if os.path.exists(config_relative):
            return config_relative
    
    # Else, return original path
    return checkpoint_path


def build_models(config: Dict[str, Any], device: torch.device, prediction_horizons: List[float] = None) -> Dict[str, nn.Module]:
    """
    Build models based on configuration.
    Now supports loading pretrained checkpoints and fusion-only models.
    
    Args:
        config: Configuration dictionary
        device: Device to place models on
        prediction_horizons: List of prediction horizons for multi-horizon models (if None, gets from config)
        
    Returns:
        Dictionary of initialized models
    """
    models = {}
    
    # Get config directory for relative path resolution
    config_file_path = config.get('__config_file__', None)
    config_dir = os.path.dirname(config_file_path) if config_file_path else None
    
    # Get prediction horizons from parameter or config
    if prediction_horizons is None:
        prediction_horizons = config.get('dataset', {}).get('prediction_horizons', [0])
    
    # Get dataset config to determine number of classes
    dataset_config_path = config['dataset']['config_path']
    dataset_config = load_config(dataset_config_path)
    
    # Load label mapping to get number of classes
    label_mapping_path = os.path.join(
        dataset_config['root_path'],
        dataset_config['label_mapping_file']
    )
    
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    # Determine number of classes
    original_num_classes = len(label_mapping.get("idx_to_label", {}))
    background_label_str = config['dataset'].get("background_class_value", "background")
    if config['dataset'].get('exclude_background_class', False):
        num_classes = 0
        for label in label_mapping.get("label_to_idx", {}).keys():
             if label != background_label_str:
                 num_classes += 1
        print(f"Background class exclusion is ON. Effective number of classes for model: {num_classes}")
    else:
        num_classes = original_num_classes
    
    if num_classes == 0:
        print(f"Warning: Could not determine number of classes from label mapping. Using default: {num_classes}")
        raise ValueError("Number of classes could not be determined. Please check label mapping file.")
    else:
        print(f"Number of classes determined from label mapping: {num_classes}")
    
    # Print prediction horizons info
    print(f"Building models with prediction horizons: {prediction_horizons}")
    
    # Import models module
    models_module = importlib.import_module('models')
    
    # Check if we have a fusion-only model like EVI-MAE
    fusion_only = False
    if 'fusion_model' in config.get('models', {}):
        fusion_config = config['models']['fusion_model']
        fusion_class_name = fusion_config.get('name')
        
        # EVI-MAE and similar models handle all modalities internally
        if fusion_class_name in ['EVI_MAE_Fusion', 'SFTIK_Fusion']:
            fusion_only = True
            print(f"{fusion_class_name} is a fusion-only model, skipping individual modality models")
    
    # Single modality models first (skip if fusion-only)
    if not fusion_only:
        for model_name, model_config in config.get('models', {}).items():
            if model_name == 'fusion_model':
                continue
                
            # Model class
            model_class_name = model_config.get('name')
            model_params = model_config.get('params', {}).copy()  # Create a copy to avoid modifying original
            
            # Add num_classes if not explicitly set in config
            if 'num_classes' not in model_params:
                model_params['num_classes'] = num_classes
            
            # Add prediction horizons to model parameters
            if 'prediction_horizons' not in model_params:
                model_params['prediction_horizons'] = prediction_horizons
            
            # Add shared classifier layers option
            if 'shared_classifier_layers' not in model_params:
                shared_classifier_layers = model_config.get('shared_classifier_layers', True)
                model_params['shared_classifier_layers'] = shared_classifier_layers
            
            # Check if model class exists
            if hasattr(models_module, model_class_name):
                model_class = getattr(models_module, model_class_name)
                
                # Initialize model
                model = model_class(**model_params)
                model = model.to(device)

                # Load pretrained weights if specified
                if 'pretrained_checkpoint' in model_config:
                    checkpoint_path = model_config['pretrained_checkpoint']
                    resolved_path = resolve_checkpoint_path(checkpoint_path, config_dir)
                    
                    if os.path.exists(resolved_path):
                        load_pretrained_encoder(model, resolved_path, model_class_name)
                    else:
                        logging.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
                
                # Store model
                models[model_name] = model
                
                print(f"Built {model_class_name} with {len(prediction_horizons)} prediction heads")
            else:
                raise ValueError(f"Model class {model_class_name} not found in models module")
    
    # Build fusion model if defined
    if 'fusion_model' in config.get('models', {}):
        fusion_config = config['models']['fusion_model']
        fusion_class_name = fusion_config.get('name')
        fusion_params = fusion_config.get('params', {}).copy()  # Create a copy
        
        if 'num_classes' not in fusion_params:
            fusion_params['num_classes'] = num_classes
        
        # Add prediction horizons to fusion model
        if 'prediction_horizons' not in fusion_params:
            fusion_params['prediction_horizons'] = prediction_horizons
            
        # Add shared classifier layers option
        if 'shared_classifier_layers' not in fusion_params:
            shared_classifier_layers = fusion_config.get('shared_classifier_layers', True)
            fusion_params['shared_classifier_layers'] = shared_classifier_layers
            
        # Check if the fusion model class exists
        if hasattr(models_module, fusion_class_name):
            fusion_class = getattr(models_module, fusion_class_name)
            
            # Handle different types of fusion models
            if fusion_class_name in ['EVI_MAE_Fusion', 'SFTIK_Fusion']:
                # These models handle modalities internally, don't need separate encoders
                # Resolve pretrained checkpoint path before passing to constructor
                if 'pretrained_checkpoint' in fusion_params:
                    fusion_params['pretrained_checkpoint'] = resolve_checkpoint_path(
                        fusion_params['pretrained_checkpoint'], config_dir
                    )
                # Just initialize with the provided parameters
                fusion_model = fusion_class(**fusion_params)
                fusion_model = fusion_model.to(device)
                
                # Load pretrained weights if specified
                if 'pretrained_checkpoint' in fusion_config:
                    checkpoint_path = fusion_config['pretrained_checkpoint']
                    resolved_path = resolve_checkpoint_path(checkpoint_path, config_dir)
                    
                    if os.path.exists(resolved_path):
                        # For EVI-MAE, the checkpoint loading is handled internally
                        print(f"Loading pretrained checkpoint from {resolved_path}")
                        # The model's __init__ handles this, but we can also load here if needed
                    else:
                        logging.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            
            else:
                # Regular fusion models that combine separate encoders
                # Prepare modality encoders dictionary
                modality_encoders = {}
                
                # Map modalities to their corresponding encoder models
                for modality in config.get('modalities', {}):
                    # The model name is expected to be "{modality}_model"
                    model_name = f"{modality}_model"
                    if model_name in models:
                        modality_encoders[modality] = models[model_name]
                    else:
                        raise ValueError(f"Model {model_name} for modality {modality} not found")
                
                # For both FusionModel and EnhancedFusionModel, pass modality_encoders
                if fusion_class_name in ['FusionModel', 'EnhancedFusionModel']:
                    fusion_params['modality_encoders'] = modality_encoders
                
                # For legacy MultimodalModel, handle the old way
                elif fusion_class_name == 'MultimodalModel':
                    # Collect modality models and feature dimensions
                    modality_models = {}
                    feature_dimensions = {}
                    
                    if 'modality_models' in fusion_config:
                        # Get feature dimensions from config if available
                        feature_dims_config = fusion_config.get('feature_dimensions', {})
                        
                        for modality, model_name in fusion_config['modality_models'].items():
                            if model_name in models:
                                modality_models[modality] = models[model_name]
                                
                                # Get feature dimension from config if specified
                                if modality in feature_dims_config:
                                    feature_dimensions[modality] = feature_dims_config[modality]
                            else:
                                raise ValueError(f"Model {model_name} referenced in fusion model not found")
                    
                    # Update fusion parameters
                    fusion_params['modality_models'] = modality_models
                    if feature_dimensions:
                        fusion_params['feature_dimensions'] = feature_dimensions
                
                # Initialize fusion model
                fusion_model = fusion_class(**fusion_params)
                fusion_model = fusion_model.to(device)
                
                # Load pretrained fusion model weights if specified
                if 'pretrained_checkpoint' in fusion_config:
                    checkpoint_path = fusion_config['pretrained_checkpoint']
                    resolved_path = resolve_checkpoint_path(checkpoint_path, config_dir)
                    
                    if os.path.exists(resolved_path):
                        load_pretrained_fusion_model(fusion_model, resolved_path)
                    else:
                        logging.warning(f"Fusion model checkpoint not found: {checkpoint_path}")
            
            # Store fusion model
            models['fusion_model'] = fusion_model
            
            print(f"Built {fusion_class_name} with {len(prediction_horizons)} prediction heads")
        else:
            raise ValueError(f"Fusion model class {fusion_class_name} not found in models module")
    
    return models


def get_main_model(config: Dict[str, Any], models: Dict[str, nn.Module]) -> nn.Module:
    """
    Get the main model for training based on configuration.
    
    Args:
        config: Configuration dictionary
        models: Dictionary of initialized models
        
    Returns:
        Main model for training
    """
    # Check if there's a fusion model
    if 'fusion_model' in models:
        return models['fusion_model']
    
    # If no fusion model, check for modalities
    modalities = config.get('modalities', {})
    
    if 'raw_imu' in modalities and 'raw_imu_model' in models:
        return models['raw_imu_model']
    
    if 'video' in modalities and 'video_model' in models:
        return models['video_model']
    
    if 'image' in modalities and 'image_model' in models:
        return models['image_model']
    
    # If no models match, raise an error
    raise ValueError("No main model found in configuration")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                  scheduler: Optional[Any] = None, device: torch.device = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = next(model.parameters()).device
    
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


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
                   epoch: int, metrics: Dict[str, float], checkpoint_path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_path: Path to save checkpoint to
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def get_model_prediction_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get prediction information from a model.
    
    Args:
        model: Model to inspect
        
    Returns:
        Dictionary with prediction information
    """
    info = {
        'is_multi_horizon': False,
        'num_heads': 1,
        'prediction_horizons': [0]
    }
    
    if hasattr(model, 'get_num_prediction_heads'):
        info['num_heads'] = model.get_num_prediction_heads()
        info['is_multi_horizon'] = info['num_heads'] > 1
        
    if hasattr(model, 'get_prediction_horizons'):
        info['prediction_horizons'] = model.get_prediction_horizons()
        
    return info