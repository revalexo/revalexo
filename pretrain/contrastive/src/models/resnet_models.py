# models/resnet_models.py

import torch
import torch.nn as nn
import torchvision.models as models
from .base_models import BaseEncoder, MultiHorizonClassifier
from typing import List

class ResNet_Image(BaseEncoder):
    """
    Image model using pretrained ResNet backbone from torchvision.
    Supports ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152.
    
    Adapted to inherit from BaseEncoder for consistent interface in multimodal settings.
    Now supports multiple prediction horizons.
    """
    def __init__(
        self, 
        num_classes=11, 
        pretrained=True, 
        model_size="50",  # Options: "18", "34", "50", "101", "152"
        feature_dim=None, 
        freeze_backbone=False,
        freeze_early_layers=False,  # Freeze conv1 through layer3
        prediction_horizons=[0],
        shared_classifier_layers=True
    ):
        # Set feature_dim to a default if not provided
        super().__init__(
            feature_dim=feature_dim or 512,
            prediction_horizons=prediction_horizons
        )
        
        # Model size to architecture mapping
        model_dict = {
            "18": models.resnet18,
            "34": models.resnet34,
            "50": models.resnet50,
            "101": models.resnet101,
            "152": models.resnet152
        }
        
        if model_size not in model_dict:
            raise ValueError(f"Invalid ResNet size. Choose from: {list(model_dict.keys())}")
        
        # Load the pretrained model
        weights = 'DEFAULT' if pretrained else None
        self.backbone = model_dict[model_size](weights=weights)
        
        # Get the feature dimension from the last layer
        self.backbone_dim = self.backbone.fc.in_features
        print(f"ResNet-{model_size} backbone dimension: {self.backbone_dim}")
        
        # Remove the classification head
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_early_layers:
            # Freeze only early layers (conv1, bn1, layer1, layer2, layer3)
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']):
                    param.requires_grad = False
        
        # Feature projection pathway
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.feature_dim),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=0.5,
            shared_layers=shared_classifier_layers
        )
        
        print(f"ResNet-{model_size} model initialized with {num_classes} classes")
        if freeze_backbone:
            print("All backbone layers frozen")
        elif freeze_early_layers:
            print("Early layers (conv1-layer3) frozen, layer4 trainable")
        else:
            print("All layers trainable")
    
    def encode_features(self, x):
        """
        Extract features without classification head.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            Expected to be [B, 3, H, W] for RGB images
            
        Returns:
            Feature representation of shape [batch_size, feature_dim]
        """
        # Check input shape
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape: {x.shape}")
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Project features to desired dimension
        features = self.feature_projector(backbone_features)
        
        return features
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            List[torch.Tensor]: List of classification outputs for each prediction horizon
        """
        # Extract features
        features = self.encode_features(x)
        
        # Apply multi-horizon classification
        outputs = self.classifier(features)
        
        return outputs

# Convenience classes for specific ResNet variants
class ResNet18_Image(ResNet_Image):
    """ResNet-18 for image classification"""
    def __init__(self, **kwargs):
        super().__init__(model_size="18", **kwargs)

class ResNet34_Image(ResNet_Image):
    """ResNet-34 for image classification"""
    def __init__(self, **kwargs):
        super().__init__(model_size="34", **kwargs)

class ResNet50_Image(ResNet_Image):
    """ResNet-50 for image classification"""
    def __init__(self, **kwargs):
        super().__init__(model_size="50", **kwargs)