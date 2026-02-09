# src/utils/load_pretrained.py

"""
Utility functions for loading pretrained IMU2CLIP encoders.
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any


def load_pretrained_imu_encoder(
    checkpoint_path: str,
    model_type: str = 'fullbody',
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load a pretrained IMU encoder from IMU2CLIP training.
    
    Args:
        checkpoint_path: Path to the encoder checkpoint (e.g., 'imu_encoder_final.pt')
        model_type: 'fullbody' (102 channels) or 'lower' (42 channels)
        device: Device to load model on
        
    Returns:
        Loaded IMU encoder model
    """
    from src.models.deepconvlstm import DeepConvLSTM
    
    # Set parameters based on model type
    if model_type == 'fullbody':
        channels = 102  # 17 body parts × 3 axes × 2 sensors
    elif model_type == 'lower':
        channels = 42   # 7 lower body parts × 3 axes × 2 sensors
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model
    model = DeepConvLSTM(
        channels=channels,
        num_classes=11,  # Will be replaced by your downstream task
        window_size=120,
        conv_kernels=64,
        conv_kernel_size=21,
        lstm_units=1024,
        lstm_layers=1,
        dropout=0.5,
        feature_dim=512,
        prediction_horizons=[0]  # Will be replaced by your task horizons
    )
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # strict=False to ignore classifier heads
    
    model.to(device)
    print(f"Loaded pretrained IMU encoder from: {checkpoint_path}")
    
    return model


def load_pretrained_visual_encoder(
    checkpoint_path: str,
    encoder_type: str = 'resnet18',
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load a pretrained visual encoder from IMU2CLIP training.
    
    Args:
        checkpoint_path: Path to the encoder checkpoint
        encoder_type: 'resnet18' or 'x3d'
        device: Device to load model on
        
    Returns:
        Loaded visual encoder model
    """
    if encoder_type == 'resnet18':
        from src.models.resnet_models import ResNet18_Image
        model = ResNet18_Image(
            num_classes=11,  # Will be replaced
            pretrained=False,  # We're loading our own weights
            feature_dim=512,
            prediction_horizons=[0]
        )
    elif encoder_type == 'x3d':
        from src.models.pytorchvideo_models import X3D_Video
        model = X3D_Video(
            num_classes=11,
            pretrained=False,
            model_size='xs',
            feature_dim=512,
            prediction_horizons=[0]
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    print(f"Loaded pretrained visual encoder from: {checkpoint_path}")
    
    return model


def freeze_encoder_backbone(model: torch.nn.Module, freeze_all: bool = False):
    """
    Freeze encoder backbone for fine-tuning.
    
    Args:
        model: Encoder model
        freeze_all: If True, freeze all layers. If False, only freeze early layers.
    """
    if freeze_all:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        print("Froze all encoder parameters")
    else:
        # Freeze only the feature extraction layers, not the final layers
        # This is model-specific and may need adjustment
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"Froze {frozen_count} encoder parameters")


def load_for_downstream_task(
    imu_checkpoint: str,
    visual_checkpoint: Optional[str] = None,
    model_type: str = 'fullbody',
    num_classes: int = 11,
    prediction_horizons: list = [0, 0.1, 0.2, 0.3, 0.5, 1.0],
    freeze_backbone: bool = False,
    device: str = 'cuda'
) -> Dict[str, torch.nn.Module]:
    """
    Load pretrained encoders and prepare them for downstream tasks.
    
    Args:
        imu_checkpoint: Path to pretrained IMU encoder
        visual_checkpoint: Optional path to pretrained visual encoder
        model_type: 'fullbody' or 'lower'
        num_classes: Number of classes for downstream task
        prediction_horizons: Prediction horizons for downstream task
        freeze_backbone: Whether to freeze encoder backbone
        device: Device to load models on
        
    Returns:
        Dictionary containing loaded models
    """
    from src.models.deepconvlstm import DeepConvLSTM
    
    models = {}
    
    # Load IMU encoder with downstream task configuration
    if model_type == 'fullbody':
        channels = 102
    else:
        channels = 42
    
    imu_model = DeepConvLSTM(
        channels=channels,
        num_classes=num_classes,
        window_size=120,
        conv_kernels=64,
        conv_kernel_size=21,
        lstm_units=1024,
        lstm_layers=1,
        dropout=0.5,
        feature_dim=512,
        prediction_horizons=prediction_horizons  # Use task-specific horizons
    )
    
    # Load pretrained weights (only for encoder part)
    pretrained_state = torch.load(imu_checkpoint, map_location=device)
    
    # Filter out classifier weights that don't match
    encoder_state = {k: v for k, v in pretrained_state.items() 
                     if not k.startswith('classifier')}
    
    imu_model.load_state_dict(encoder_state, strict=False)
    
    if freeze_backbone:
        freeze_encoder_backbone(imu_model, freeze_all=False)
    
    imu_model.to(device)
    models['imu'] = imu_model
    print(f"Loaded IMU encoder for downstream task with {num_classes} classes")
    
    # Load visual encoder if provided
    if visual_checkpoint:
        # Determine encoder type from checkpoint path
        if 'resnet' in visual_checkpoint.lower():
            from src.models.resnet_models import ResNet18_Image
            visual_model = ResNet18_Image(
                num_classes=num_classes,
                pretrained=False,
                feature_dim=512,
                prediction_horizons=prediction_horizons
            )
        elif 'x3d' in visual_checkpoint.lower():
            from src.models.pytorchvideo_models import X3D_Video
            visual_model = X3D_Video(
                num_classes=num_classes,
                pretrained=False,
                model_size='xs',
                feature_dim=512,
                prediction_horizons=prediction_horizons
            )
        
        pretrained_visual_state = torch.load(visual_checkpoint, map_location=device)
        encoder_visual_state = {k: v for k, v in pretrained_visual_state.items() 
                                if not k.startswith('classifier')}
        
        visual_model.load_state_dict(encoder_visual_state, strict=False)
        
        if freeze_backbone:
            freeze_encoder_backbone(visual_model, freeze_all=False)
        
        visual_model.to(device)
        models['visual'] = visual_model
        print(f"Loaded visual encoder for downstream task")
    
    return models


# Example usage
if __name__ == '__main__':
    # Example: Load pretrained encoders for downstream task
    models = load_for_downstream_task(
        imu_checkpoint='checkpoints/imu2clip_deepconv_fullbody_resnet18/imu_encoder_final.pt',
        visual_checkpoint='checkpoints/imu2clip_deepconv_fullbody_resnet18/visual_encoder_final.pt',
        model_type='fullbody',
        num_classes=13,  # Your number of activity classes
        prediction_horizons=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
        freeze_backbone=False,  # Set to True for linear probing
        device='cuda'
    )
    
    print("Models loaded successfully!")
    print(f"IMU model: {models['imu']}")
    print(f"Visual model: {models.get('visual', 'Not loaded')}")