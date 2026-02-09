# src/models/clip_encoder.py
"""
CLIP visual encoder wrapper for IMU2CLIP training.
This implements the original paper's approach using frozen CLIP encoders.
"""

import torch
import torch.nn as nn
import clip
from typing import Optional, List
import sys

from src.models.base_models import BaseEncoder


class CLIPVisualEncoder(BaseEncoder):
    """
    CLIP visual encoder wrapper that matches the original IMU2CLIP paper.
    Uses frozen pretrained CLIP visual encoder.
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        feature_dim: int = 512,
        freeze: bool = True,
        prediction_horizons: List[float] = [0],
        device: str = 'cuda',
        **kwargs
    ):
        """
        Initialize CLIP visual encoder.
        
        Args:
            clip_model_name: CLIP model variant ("ViT-B/32", "ViT-B/16", "RN50", etc.)
            feature_dim: Output feature dimension (CLIP outputs 512 for ViT-B/32)
            freeze: Whether to freeze CLIP weights (True in original paper)
            prediction_horizons: Prediction horizons (not used for CLIP)
            device: Device to load model on
        """
        super().__init__(feature_dim=feature_dim, prediction_horizons=prediction_horizons)
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.clip_model_name = clip_model_name
        
        # CLIP's actual output dimension
        self.clip_feature_dim = self.clip_model.visual.output_dim
        
        # Projection layer if needed
        if self.clip_feature_dim != feature_dim:
            self.feature_projector = nn.Linear(self.clip_feature_dim, feature_dim)
        else:
            self.feature_projector = nn.Identity()
        
        # Freeze CLIP if specified (as in the paper)
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        # Dummy classifier for compatibility (won't be used)
        from .base_models import MultiHorizonClassifier
        self.classifier = MultiHorizonClassifier(
            input_dim=feature_dim,
            num_classes=11,  # Dummy
            prediction_horizons=prediction_horizons,
            dropout=0.0
        )
    
    def encode_features(self, x):
        """
        Extract CLIP visual features.
        
        Args:
            x: Input tensor [B, C, H, W] for images
               Note: CLIP expects images in range [0, 1] after normalization
            
        Returns:
            Feature representation [B, feature_dim]
        """
        # Ensure CLIP model is in eval mode if frozen
        if not self.training or all(not p.requires_grad for p in self.clip_model.parameters()):
            self.clip_model.eval()
        
        # Extract visual features using CLIP
        with torch.amp.autocast('cuda'):
            # CLIP's encode_image handles its own normalization internally
            clip_features = self.clip_model.encode_image(x)
        
        # Project if needed
        features = self.feature_projector(clip_features.float())
        
        return features
    
    def forward(self, x):
        """
        Forward pass (for compatibility).
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dummy classification outputs (not used in pretraining)
        """
        features = self.encode_features(x)
        outputs = self.classifier(features)
        return outputs