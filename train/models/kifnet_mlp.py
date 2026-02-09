# models/kifnet_mlp.py
"""
KIFNET-style MLP encoder for IMU/kinematic data
Adapted to work with RevalExo codebase structure
# https://github.com/Anvilondre/kifnet
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .base_models import BaseEncoder, MultiHorizonClassifier


class LinearBlock(nn.Module):
    """Linear -> LeakyReLU block as used in KIFNET"""
    
    def __init__(self, inp_dim: int, out_dim: int, slope: float = -0.01):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
        self.relu = nn.LeakyReLU(slope)
        
    def forward(self, x):
        return self.relu(self.fc(x))


class KIFNetMLP(BaseEncoder):
    """
    KIFNET-style MLP encoder for kinematic/IMU data.
    Flattens the input and passes through MLP layers.
    
    This is equivalent to KIFNET's Single_Single model but adapted
    to inherit from BaseEncoder for compatibility with the fusion model.
    """
    
    def __init__(
        self,
        channels: int,
        window_size: int,
        hidden_dims: List[int] = [256, 256, 128],
        feature_dim: int = 128,
        dropout: float = 0.0,
        leaky_relu_slope: float = -0.01,
        num_classes: int = 13,
        prediction_horizons: List[float] = [0],
        shared_classifier_layers: bool = True
    ):
        """
        Args:
            channels: Number of input channels (e.g., 102 for full-body IMU)
            window_size: Number of time steps (e.g., 120 for 2 seconds at 60Hz)
            hidden_dims: List of hidden layer dimensions
            feature_dim: Output feature dimension
            dropout: Dropout rate (not used in original KIFNET but kept for compatibility)
            leaky_relu_slope: Negative slope for LeakyReLU (default -0.01 as in KIFNET)
            num_classes: Number of output classes
            prediction_horizons: List of prediction horizons
            shared_classifier_layers: Whether to share classifier layers across horizons
        """
        # Call parent init with feature_dim as required
        super().__init__(
            feature_dim=feature_dim,
            prediction_horizons=prediction_horizons
        )
        
        self.channels = channels
        self.window_size = window_size
        self.num_classes = num_classes
        self.leaky_relu_slope = leaky_relu_slope
        self.shared_classifier_layers = shared_classifier_layers
        
        # Input dimension is flattened: channels * window_size
        inp_dim = channels * window_size
        
        # Build the encoder layers
        layers = []
        
        # First layer
        if len(hidden_dims) > 0:
            layers.append(LinearBlock(inp_dim, hidden_dims[0], leaky_relu_slope))
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.append(LinearBlock(hidden_dims[i], hidden_dims[i+1], leaky_relu_slope))
            
            # Final projection to feature_dim (without activation)
            layers.append(nn.Linear(hidden_dims[-1], feature_dim))
        else:
            # Direct projection if no hidden layers
            layers.append(nn.Linear(inp_dim, feature_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Optional dropout (not in original KIFNET but kept for compatibility)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=dropout,
            shared_layers=shared_classifier_layers
        )
        
        print(f"Built KIFNetMLP with {self.num_prediction_heads} prediction heads")
    
    def encode_features(self, x):
        """
        Extract features from IMU data (required by BaseEncoder).
        
        Args:
            x: Input tensor of shape [batch_size, window_size, channels]
               or [batch_size, channels, window_size]
        
        Returns:
            Features of shape [batch_size, feature_dim]
        """
        # Handle different input formats
        if x.dim() == 3:
            # Assume shape is [batch_size, window_size, channels] or [batch_size, channels, window_size]
            if x.shape[1] == self.window_size:
                # Shape is [batch_size, window_size, channels]
                batch_size = x.shape[0]
            elif x.shape[2] == self.window_size:
                # Shape is [batch_size, channels, window_size]
                # Transpose to [batch_size, window_size, channels]
                x = x.transpose(1, 2)
                batch_size = x.shape[0]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            
            # Flatten to [batch_size, window_size * channels]
            x = x.reshape(batch_size, -1)
        elif x.dim() == 2:
            # Already flattened
            batch_size = x.shape[0]
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape: {x.shape}")
        
        # Pass through encoder
        features = self.encoder(x)
        features = self.dropout(features)
        
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


class KIFNetMLPSeparateDecoders(BaseEncoder):
    """
    KIFNET-style MLP encoder with separate decoders for each output.
    This matches the original KIFNET architecture more closely where
    each joint angle has its own decoder.
    
    For classification, each "decoder" becomes a classification head.
    """
    
    def __init__(
        self,
        channels: int,
        window_size: int,
        encoder_dims: List[int] = [256, 256, 128],
        decoder_dims: List[int] = [64, 32],
        feature_dim: int = 128,
        dropout: float = 0.0,
        leaky_relu_slope: float = -0.01,
        num_classes: int = 13,
        prediction_horizons: List[float] = [0],
        shared_encoder: bool = True
    ):
        """
        Args:
            channels: Number of input channels
            window_size: Number of time steps
            encoder_dims: List of encoder hidden layer dimensions
            decoder_dims: List of decoder hidden layer dimensions
            feature_dim: Intermediate feature dimension (encoder output)
            dropout: Dropout rate
            leaky_relu_slope: Negative slope for LeakyReLU
            num_classes: Number of output classes
            prediction_horizons: List of prediction horizons
            shared_encoder: Whether to share encoder across all decoders
        """
        # The last encoder dim is the actual feature dim
        if len(encoder_dims) > 0:
            actual_feature_dim = encoder_dims[-1]
        else:
            actual_feature_dim = feature_dim
            
        # Call parent init with actual feature dim
        super().__init__(
            feature_dim=actual_feature_dim,
            prediction_horizons=prediction_horizons
        )
        
        self.channels = channels
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_prediction_heads = len(prediction_horizons)
        self.leaky_relu_slope = leaky_relu_slope
        
        # Input dimension
        inp_dim = channels * window_size
        
        # Build shared encoder
        encoder_layers = []
        if len(encoder_dims) > 0:
            encoder_layers.append(LinearBlock(inp_dim, encoder_dims[0], leaky_relu_slope))
            for i in range(len(encoder_dims) - 1):
                encoder_layers.append(LinearBlock(encoder_dims[i], encoder_dims[i+1], leaky_relu_slope))
            # Note: No final linear layer here, encoder_dims[-1] is the feature_dim
            self.actual_feature_dim = encoder_dims[-1]
        else:
            encoder_layers.append(nn.Linear(inp_dim, feature_dim))
            self.actual_feature_dim = feature_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build separate decoders for each prediction horizon
        self.decoders = nn.ModuleList()
        for _ in range(self.num_prediction_heads):
            decoder_layers = []
            
            # First decoder layer
            if len(decoder_dims) > 0:
                decoder_layers.append(LinearBlock(self.actual_feature_dim, decoder_dims[0], leaky_relu_slope))
                
                # Hidden decoder layers
                for i in range(len(decoder_dims) - 1):
                    decoder_layers.append(LinearBlock(decoder_dims[i], decoder_dims[i+1], leaky_relu_slope))
                
                # Final classification layer
                decoder_layers.append(nn.Linear(decoder_dims[-1], num_classes))
            else:
                # Direct classification if no decoder layers
                decoder_layers.append(nn.Linear(self.actual_feature_dim, num_classes))
            
            self.decoders.append(nn.Sequential(*decoder_layers))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        print(f"Built KIFNetMLPSeparateDecoders with {self.num_prediction_heads} separate decoders")
    
    def encode_features(self, x):
        """Extract features using the shared encoder."""
        # Handle input reshaping
        if x.dim() == 3:
            if x.shape[1] == self.window_size:
                batch_size = x.shape[0]
            elif x.shape[2] == self.window_size:
                x = x.transpose(1, 2)
                batch_size = x.shape[0]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            x = x.reshape(batch_size, -1)
        elif x.dim() == 2:
            batch_size = x.shape[0]
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape: {x.shape}")
        
        # Pass through encoder
        features = self.encoder(x)
        features = self.dropout(features)
        
        return features
    
    def forward(self, x):
        """
        Forward pass with separate decoders for each horizon.
        
        Returns:
            List of classification outputs for each prediction horizon
        """
        features = self.encode_features(x)
        
        # Pass through separate decoders
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(features))
        
        return outputs