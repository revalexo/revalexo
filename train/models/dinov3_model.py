# models/dinov3_model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from .base_models import BaseEncoder, MultiHorizonClassifier


class DINOv3_Image(BaseEncoder):
    """
    Image model using pretrained DINOv3 ViT backbone from HuggingFace.
    Uses the pooler output (CLS token) as the feature representation.

    Supports frozen backbone for linear probing.
    """
    def __init__(
        self,
        num_classes=11,
        pretrained=True,
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        feature_dim=None,
        freeze_backbone=False,
        prediction_horizons=[0],
        shared_classifier_layers=True
    ):
        super().__init__(
            feature_dim=feature_dim or 512,
            prediction_horizons=prediction_horizons
        )

        # Load DINOv3 backbone
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.backbone_dim = self.backbone.config.hidden_size
        print(f"DINOv3 backbone dimension: {self.backbone_dim}")

        # Freeze backbone parameters if requested
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

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

        print(f"DINOv3 model initialized with {num_classes} classes")
        if freeze_backbone:
            print("All backbone layers frozen")
        else:
            print("All layers trainable")

    def train(self, mode=True):
        """Override train to keep frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze_backbone and mode:
            self.backbone.eval()
        return self

    def encode_features(self, x):
        """
        Extract features without classification head.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Feature representation of shape [batch_size, feature_dim]
        """
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape: {x.shape}")

        # Extract backbone features
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x)
            backbone_features = outputs.pooler_output.detach()
        else:
            outputs = self.backbone(pixel_values=x)
            backbone_features = outputs.pooler_output

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
        features = self.encode_features(x)
        outputs = self.classifier(features)
        return outputs
