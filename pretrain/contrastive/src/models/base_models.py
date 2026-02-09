# models/base_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np 

class BaseEncoder(nn.Module, ABC):
    """
    Base class for all encoder models.
    
    This class defines the interface that all encoder models should implement
    to ensure compatibility with multimodal fusion architectures.
    """
    def __init__(self, feature_dim: int, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self._feature_extraction_mode = False
        
        # Added support for multiple prediction horizons
        self.prediction_horizons = kwargs.get('prediction_horizons', [0])
        self.num_prediction_heads = len(self.prediction_horizons)
    
    @abstractmethod
    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input without classification head.
        
        Args:
            x: Input tensor of appropriate shape for the modality
            
        Returns:
            Feature representation of shape [batch_size, feature_dim]
        """
        pass
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public method to extract features.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature representation
        """
        return self.encode_features(x)
    
    def get_feature_dim(self) -> int:
        """
        Get the output feature dimension of this encoder.
        
        Returns:
            Feature dimension
        """
        return self.feature_dim
        
    def get_prediction_horizons(self) -> List[float]:
        """Get the prediction horizons for this model."""
        return self.prediction_horizons.copy()
        
    def get_num_prediction_heads(self) -> int:
        """Get the number of prediction heads."""
        return self.num_prediction_heads


class MultiHorizonClassifier(nn.Module):
    """
    Multi-horizon classifier that creates separate heads for each prediction horizon.
    """
    def __init__(self, input_dim: int, num_classes: int, prediction_horizons: List[float], 
                 dropout: float = 0.5, shared_layers: bool = True):
        """
        Initialize multi-horizon classifier.
        
        Args:
            input_dim (int): Input feature dimension
            num_classes (int): Number of output classes
            prediction_horizons (List[float]): List of prediction horizons
            dropout (float): Dropout rate
            shared_layers (bool): Whether to use shared layers before individual heads
        """
        super().__init__()
        
        self.prediction_horizons = prediction_horizons
        self.num_heads = len(prediction_horizons)
        self.num_classes = num_classes
        self.shared_layers = shared_layers
        
        if shared_layers and self.num_heads > 1:
            # Shared layers before individual heads
            self.shared_net = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Individual heads for each horizon
            self.heads = nn.ModuleList([
                nn.Linear(input_dim, num_classes) for _ in range(self.num_heads)
            ])
        else:
            # Individual heads without shared layers
            self.shared_net = nn.Identity()
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, num_classes)
                ) for _ in range(self.num_heads)
            ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all prediction heads.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            List[torch.Tensor]: List of outputs for each prediction horizon
        """
        # Apply shared layers
        shared_features = self.shared_net(x)
        
        # Get predictions from each head
        outputs = []
        for head in self.heads:
            outputs.append(head(shared_features))
            
        return outputs