# models/mobileone_image.py
"""
MobileOne-S0 encoder for image data, matching KIFNET's implementation
Adapted to work with AidWear codebase structure
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .base_models import BaseEncoder, MultiHorizonClassifier
import sys
import os


class MobileOneS0_Image(BaseEncoder):
    """
    MobileOne-S0 encoder for image data, as used in KIFNET.
    
    Note: This requires the ml-mobileone package to be installed.
    You'll need to:
    1. Clone https://github.com/apple/ml-mobileone
    2. Download MobileOne-S0 weights
    3. Update the path in the config
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_early_layers: bool = True,
        mobileone_path: str = None,
        mobileone_weights: str = None,
        num_classes: int = 13,
        prediction_horizons: List[float] = [0],
        shared_classifier_layers: bool = True
    ):
        """
        Args:
            feature_dim: Output feature dimension (default 128 as in KIFNET)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze most of the backbone
            freeze_early_layers: Whether to freeze early layers (keeps last stage trainable)
            mobileone_path: Path to ml-mobileone directory
            mobileone_weights: Path to MobileOne-S0 weights file
            num_classes: Number of output classes
            prediction_horizons: List of prediction horizons
            shared_classifier_layers: Whether to share classifier layers
        """
        super().__init__(
            feature_dim=feature_dim,
            prediction_horizons=prediction_horizons
        )
        
        self.num_classes = num_classes
        self.shared_classifier_layers = shared_classifier_layers
        
        # Try to import MobileOne
        if mobileone_path:
            sys.path.insert(0, mobileone_path)
        
        try:
            from mobileone import mobileone
        except ImportError:
            raise ImportError(
                "Could not import mobileone. Please ensure ml-mobileone is installed "
                "and the path is correctly set. You can get it from: "
                "https://github.com/apple/ml-mobileone"
            )
        
        # Initialize MobileOne-S0
        self.backbone = mobileone(variant='s0')
        
        # Load pretrained weights if provided
        if pretrained and mobileone_weights:
            # Handle .pth.tar extension
            if os.path.exists(mobileone_weights):
                checkpoint = torch.load(mobileone_weights, map_location='cpu')
                self.backbone.load_state_dict(checkpoint)
                print(f"Loaded MobileOne weights from {mobileone_weights}")
            else:
                print(f"Warning: MobileOne weights file not found at {mobileone_weights}")
        
        # Get the number of features from the last layer
        self.num_ftrs = self.backbone.linear.in_features
        
        # Replace the final linear layer with our feature projection
        self.backbone.linear = nn.Linear(self.num_ftrs, feature_dim)
        
        # Freeze parameters as in KIFNET
        if freeze_backbone:
            # Freeze all parameters first
            for p in self.backbone.parameters():
                p.requires_grad = False
            
            if freeze_early_layers:
                # Only train the last stage (approximately 27.5% of parameters)
                # and the new linear layer
                # Note: The exact layers may vary, but typically includes:
                # - Global average pooling (gap)
                # - Final linear layer
                self.backbone.gap.requires_grad = True
                for p in self.backbone.gap.parameters():
                    p.requires_grad = True
                
                # Always train the new linear layer
                for p in self.backbone.linear.parameters():
                    p.requires_grad = True
                    
                # Also unfreeze the last stage (stage4 in MobileOne)
                if hasattr(self.backbone, 'stage4'):
                    for p in self.backbone.stage4.parameters():
                        p.requires_grad = True
            else:
                # Unfreeze entire backbone
                for p in self.backbone.parameters():
                    p.requires_grad = True
        
        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=0.0,  # KIFNET doesn't use dropout
            shared_layers=shared_classifier_layers
        )
        
        print(f"Built MobileOneS0_Image with {self.num_prediction_heads} prediction heads")
    
    def encode_features(self, x):
        """
        Extract features from image data.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               Expected to be normalized images (typically 3x224x224 or 3x256x256)
        
        Returns:
            Features of shape [batch_size, feature_dim]
        """
        # MobileOne forward pass already includes pooling and projection
        features = self.backbone(x)
        return features
    
    def forward(self, x):
        """
        Forward pass with classification heads.
        
        Args:
            x: Input tensor
            
        Returns:
            List of classification outputs for each prediction horizon
        """
        features = self.encode_features(x)
        outputs = self.classifier(features)
        return outputs


class MobileOneS0_Fallback(BaseEncoder):
    """
    Fallback implementation using torchvision's MobileNetV3 as a substitute
    when MobileOne is not available. This won't match KIFNET exactly but
    provides a similar lightweight architecture.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_early_layers: bool = True,
        num_classes: int = 13,
        prediction_horizons: List[float] = [0],
        shared_classifier_layers: bool = True,
        **kwargs  # Ignore extra parameters like mobileone_path, mobileone_weights
    ):
        super().__init__(
            feature_dim=feature_dim,
            prediction_horizons=prediction_horizons
        )
        
        self.num_classes = num_classes
        self.shared_classifier_layers = shared_classifier_layers
        
        # Use MobileNetV3-Small as a lightweight alternative
        from torchvision import models
        
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Get the number of features from the last layer
        self.num_ftrs = self.backbone.classifier[-1].in_features
        
        # Replace the classifier with our feature projection
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs, feature_dim)
        )
        
        # Freeze parameters similar to KIFNET
        if freeze_backbone:
            # Freeze all feature layers
            for p in self.backbone.features.parameters():
                p.requires_grad = False
            
            if freeze_early_layers:
                # Unfreeze last few blocks (approximately last 25-30% of network)
                # In MobileNetV3-Small, this would be roughly the last 3-4 blocks
                for i in range(9, len(self.backbone.features)):  # Last ~4 blocks
                    for p in self.backbone.features[i].parameters():
                        p.requires_grad = True
            
            # Always train the new classifier
            for p in self.backbone.classifier.parameters():
                p.requires_grad = True
        
        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=0.0,  # KIFNET doesn't use dropout
            shared_layers=shared_classifier_layers
        )
        
        print("Note: Using MobileNetV3-Small as fallback. For exact KIFNET replication, "
              "please install ml-mobileone and use MobileOneS0_Image.")
        print(f"Built MobileOneS0_Fallback with {self.num_prediction_heads} prediction heads")
    
    def encode_features(self, x):
        """Extract features from image data."""
        features = self.backbone(x)
        return features
    
    def forward(self, x):
        """Forward pass with classification heads."""
        features = self.encode_features(x)
        outputs = self.classifier(features)
        return outputs