# models/mobilenet_models.py

import torch
import torch.nn as nn
import torchvision.models as models
from .base_models import BaseEncoder, MultiHorizonClassifier
from typing import List

class MobileNet_Image(BaseEncoder):
    """
    Image model using pretrained MobileNet backbone from torchvision.
    Supports MobileNetV2 and MobileNetV3 (small and large).
    
    Adapted to inherit from BaseEncoder for consistent interface in multimodal settings.
    Supports multiple prediction horizons.
    """
    def __init__(
        self, 
        num_classes=11, 
        pretrained=True, 
        model_variant="v3_large",  # Options: "v2", "v3_small", "v3_large"
        feature_dim=None, 
        freeze_backbone=False,
        freeze_early_layers=False,  # Freeze early layers
        prediction_horizons=[0],
        shared_classifier_layers=True
    ):
        # Set feature_dim to a default if not provided
        super().__init__(
            feature_dim=feature_dim or 512,
            prediction_horizons=prediction_horizons
        )
        
        # Model variant to architecture mapping
        model_dict = {
            "v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT if pretrained else None),
            "v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None),
            "v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None),
        }
        
        if model_variant not in model_dict:
            raise ValueError(f"Invalid MobileNet variant. Choose from: {list(model_dict.keys())}")
        
        # Load the model
        model_fn, weights = model_dict[model_variant]
        self.backbone = model_fn(weights=weights)
        self.model_variant = model_variant
        
        # Get the feature dimension from the classifier layer
        if model_variant == "v2":
            # MobileNetV2 has a different structure
            self.backbone_dim = self.backbone.classifier[1].in_features
        else:
            # MobileNetV3 variants
            self.backbone_dim = self.backbone.classifier[0].in_features
        
        print(f"MobileNet{model_variant.upper()} backbone dimension: {self.backbone_dim}")
        
        # Remove the classification head
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_early_layers:
            # Freeze early layers based on model variant
            for name, param in self.backbone.named_parameters():
                if model_variant == "v2":
                    # For MobileNetV2, freeze first 10 inverted residual blocks
                    if any(f'features.{i}' in name for i in range(10)):
                        param.requires_grad = False
                else:
                    # For MobileNetV3, freeze first 8 blocks
                    if any(f'features.{i}' in name for i in range(8)):
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
        
        print(f"MobileNet{model_variant.upper()} model initialized with {num_classes} classes")
        if freeze_backbone:
            print("All backbone layers frozen")
        elif freeze_early_layers:
            if model_variant == "v2":
                print("Early layers (features[0:10]) frozen")
            else:
                print("Early layers (features[0:8]) frozen")
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


# Convenience classes for specific MobileNet variants
class MobileNetV2_Image(MobileNet_Image):
    """MobileNetV2 for image classification"""
    def __init__(self, **kwargs):
        super().__init__(model_variant="v2", **kwargs)


class MobileNetV3Small_Image(MobileNet_Image):
    """MobileNetV3-Small for image classification"""
    def __init__(self, **kwargs):
        super().__init__(model_variant="v3_small", **kwargs)


class MobileNetV3Large_Image(MobileNet_Image):
    """MobileNetV3-Large for image classification"""
    def __init__(self, **kwargs):
        super().__init__(model_variant="v3_large", **kwargs)