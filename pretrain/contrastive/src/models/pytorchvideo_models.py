# models/pytorchvideo_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import BaseEncoder, MultiHorizonClassifier

class X3D_Video(BaseEncoder):
    """
    Video model using pretrained X3D backbone from PyTorchVideo
    
    Adapted to inherit from BaseEncoder for consistent interface in multimodal settings.
    Now supports multiple prediction horizons.
    """
    def __init__(
        self, 
        num_classes=19, 
        pretrained=True, 
        model_size="m", 
        feature_dim=None, 
        freeze_backbone=False,
        prediction_horizons=[0],
        shared_classifier_layers=True
    ):
        # Set feature_dim to a default if not provided
        self.backbone_dim = None  # Will be set after loading backbone
        super().__init__(
            feature_dim=feature_dim or 512,
            prediction_horizons=prediction_horizons
        )
        
        # Create X3D model with the specified size (xs, s, m, or l)
        model_name = f"x3d_{model_size}"
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("blocks.5"):
                    param.requires_grad = False

        last_block = self.backbone.blocks[-1]
        
        if hasattr(last_block, 'proj'):
            # Get the input dimension to the projection layer
            self.backbone_dim = last_block.proj.in_features
            print(f"Backbone feature dimension (from proj.in_features): {self.backbone_dim}")
            
            # Save the old proj output dimension for reference
            old_out_features = last_block.proj.out_features
            print(f"Original projection output dimension: {old_out_features}")
            
            # Replace the final projection layer with identity
            last_block.proj = nn.Identity()
        else:
            print("Warning: Could not find 'proj' attribute in the last block.")
            print("Available attributes:", dir(last_block))
            
            # Use a test forward pass to determine the output dimension
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 16, 224, 224)
                features = self.backbone(dummy_input)
                self.backbone_dim = features.shape[1]
                print(f"Backbone feature dimension (from test forward pass): {self.backbone_dim}")
        
        # Now that we know the backbone_dim, update the feature_dim if it wasn't provided
        if feature_dim is None:
            self.feature_dim = 512
            
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

    def _print_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def encode_features(self, x):
        """
        Extract features without classification head.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W]
            
        Returns:
            torch.Tensor: Feature representation of shape [B, feature_dim]
        """
        # Check input shape
        if len(x.shape) != 5:
            raise ValueError(f"Expected 5D input tensor [B, C, T, H, W], got shape: {x.shape}")
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Project features to desired dimension
        features = self.feature_projector(backbone_features)
        
        return features
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W]
        
        Returns:
            List[torch.Tensor]: List of classification outputs for each prediction horizon
        """
        # Extract features
        features = self.encode_features(x)
        
        # Apply multi-horizon classification
        outputs = self.classifier(features)

        return outputs


class MViT_Video(BaseEncoder):
    """
    Video model using pretrained MViT-B backbone from PyTorchVideo

    Adapted to inherit from BaseEncoder for consistent interface in multimodal settings.
    Now supports multiple prediction horizons.
    """
    def __init__(
        self,
        num_classes=19,
        pretrained=True,
        variant="16x4",
        feature_dim=None,
        freeze_backbone=False,
        prediction_horizons=[0],
        shared_classifier_layers=True
    ):
        # Set default feature dimension
        super().__init__(
            feature_dim=feature_dim or 512,
            prediction_horizons=prediction_horizons
        )

        # Check for valid variant
        valid_variants = ["16x4", "32x3"]
        if variant not in valid_variants:
            raise ValueError(f"Invalid MViT variant. Choose from: {valid_variants}")

        # Create MViT model with the specified variant
        if variant == "16x4":
            model_name = "mvit_base_16x4"
        else:
            model_name = "mvit_base_32x3"

        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)

        # Get the backbone output dimension
        # MViT returns a tensor [B, N, D] where N is the number of tokens and D is the dimension
        self.backbone_dim = 768  # This is the dimension of the embeddings in MViT-B

        # Replace the classifier head to remove the classification layer
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()

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

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def encode_features(self, x):
        """
        Extract features from input without classification head.

        Args:
            x: Input tensor of shape [B, C, T, H, W]

        Returns:
            Feature representation of shape [B, feature_dim]
        """
        # Check input shape
        if len(x.shape) != 5:
            raise ValueError(f"Expected 5D input tensor [B, C, T, H, W], got shape: {x.shape}")

        # Extract backbone features
        features = self.backbone(x)  # Shape: [B, N, D]

        # For MViT, we need to extract just the class token (first token)
        class_token = features[:, 0]  # Shape: [B, D]

        # Project features to desired dimension
        features = self.feature_projector(class_token)

        return features

    def forward(self, x):
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape [B, C, T, H, W]

        Returns:
            List[torch.Tensor]: List of classification outputs for each prediction horizon
        """
        # Extract features
        features = self.encode_features(x)

        # Apply multi-horizon classification
        outputs = self.classifier(features)

        return outputs
