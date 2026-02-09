from .base_models import BaseEncoder, MultiHorizonClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union, List

class FusionModel(nn.Module):
    """
    Generic fusion model that combines features from multiple modality encoders.
    Now supports multiple prediction horizons and additional fusion methods including AdaIN.
    
    This model supports different fusion methods and can work with any encoders
    that inherit from BaseEncoder.
    """
    def __init__(
        self,
        modality_encoders: Dict[str, BaseEncoder],
        fusion_method: str = "concatenate",
        hidden_dim: int = 512,
        num_classes: int = 13,
        dropout: float = 0.5,
        transformer_params: Optional[Dict[str, Any]] = None,
        prediction_horizons: List[float] = [0],
        shared_classifier_layers: bool = True
    ):
        super().__init__()
        
        self.modality_encoders = nn.ModuleDict()
        self.modalities = []
        self.feature_dims = {}
        self.fusion_method = fusion_method
        self.prediction_horizons = prediction_horizons
        self.num_prediction_heads = len(prediction_horizons)

        self.normalize_features = True
        
        # Process the provided encoders
        for modality, encoder in modality_encoders.items():
            self.modality_encoders[modality] = encoder
            self.modalities.append(modality)
            self.feature_dims[modality] = encoder.get_feature_dim()

        # Ensure we have exactly 2 modalities for certain fusion methods
        if fusion_method in ["adain_content_style", "adain_style_content", "dot_attention"]:
            if len(self.modalities) != 2:
                raise ValueError(f"Fusion method {fusion_method} requires exactly 2 modalities, got {len(self.modalities)}")

        self.modality_norms = nn.ModuleDict()
        for modality, encoder in modality_encoders.items():
            feature_dim = encoder.get_feature_dim()
            self.modality_norms[modality] = nn.BatchNorm1d(feature_dim)
        
        print(f"Feature dimensions for fusion: {self.feature_dims}")
        print(f"Prediction horizons: {self.prediction_horizons}")
        
        # Default transformer parameters
        default_transformer_params = {
            'nhead': 8,
            'num_layers': 2,
            'dropout': 0.1,
            'position_embedding': True
        }
        
        # Use provided transformer params or defaults
        if transformer_params is None:
            transformer_params = default_transformer_params
        else:
            # Fill in any missing parameters with defaults
            for k, v in default_transformer_params.items():
                if k not in transformer_params:
                    transformer_params[k] = v
        
        # Set up fusion based on method
        if fusion_method == "average":
            # Simple average fusion
            # Ensure all modalities have same dimension
            self.common_dim = max(self.feature_dims.values())
            self.projections = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                if dim != self.common_dim:
                    self.projections[modality] = nn.Linear(dim, self.common_dim)
            
            self.fusion = nn.Sequential(
                nn.Linear(self.common_dim, hidden_dim),
                nn.ReLU()
            )
            
        elif fusion_method == "maximum":
            # Maximum fusion
            self.common_dim = max(self.feature_dims.values())
            self.projections = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                if dim != self.common_dim:
                    self.projections[modality] = nn.Linear(dim, self.common_dim)
            
            self.fusion = nn.Sequential(
                nn.Linear(self.common_dim, hidden_dim),
                nn.ReLU()
            )
        
        elif fusion_method == "concatenate":
            # Concatenation-based fusion
            total_dim = sum(self.feature_dims.values())
            print(f"Total concatenated dimension: {total_dim}")
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU()
            )

        elif fusion_method == "concat_batchnorm":
            total_dim = sum(self.feature_dims.values())
            self.concat_bn = nn.BatchNorm1d(total_dim)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU()
            )

        elif fusion_method == "concat_layernorm":
            total_dim = sum(self.feature_dims.values())
            self.concat_ln = nn.LayerNorm(total_dim)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU()
            )
        
        elif fusion_method == "dot_attention":
            # Dot product attention (only for 2 modalities)
            if len(self.modalities) != 2:
                raise ValueError("Dot attention requires exactly 2 modalities")
            
            # Ensure same dimensions
            self.common_dim = max(self.feature_dims.values())
            self.projections = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                if dim != self.common_dim:
                    self.projections[modality] = nn.Linear(dim, self.common_dim)
            
            self.fusion = nn.Sequential(
                nn.Linear(self.common_dim, hidden_dim),
                nn.ReLU()
            )
        
        elif fusion_method == "mlp_attention":
            # MLP-based attention
            self.attention_layers = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                self.attention_layers[modality] = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
            
            # Common feature dimension for attention mechanism
            self.common_dim = max(self.feature_dims.values())
            
            # Linear projection for each modality to ensure common dimension
            self.projections = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                if dim != self.common_dim:
                    self.projections[modality] = nn.Linear(dim, self.common_dim)
            
            self.fusion = nn.Sequential(
                nn.Linear(self.common_dim, hidden_dim),
                nn.ReLU()
            )
        
        elif fusion_method in ["adain_content_style", "adain_style_content"]:
            # AdaIN fusion - adaptive instance normalization
            # First modality is content, second is style (or reversed)
            if len(self.modalities) != 2:
                raise ValueError("AdaIN requires exactly 2 modalities")
            
            # Project to same dimension
            self.common_dim = max(self.feature_dims.values())
            self.projections = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                if dim != self.common_dim:
                    self.projections[modality] = nn.Linear(dim, self.common_dim)
            
            self.fusion = nn.Sequential(
                nn.Linear(self.common_dim, hidden_dim),
                nn.ReLU()
            )
        
        elif fusion_method == "weighted":
            # Weighted average fusion with learnable weights
            self.modality_weights = nn.ParameterDict()
            for modality in self.feature_dims.keys():
                self.modality_weights[modality] = nn.Parameter(torch.tensor(1.0 / len(self.feature_dims)))
            
            # Common feature dimension for weighted fusion
            self.common_dim = max(self.feature_dims.values())
            
            # Linear projection for each modality to ensure common dimension
            self.projections = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                if dim != self.common_dim:
                    self.projections[modality] = nn.Linear(dim, self.common_dim)
                    
            self.fusion = nn.Sequential(
                nn.Linear(self.common_dim, hidden_dim),
                nn.ReLU()
            )
            
        elif fusion_method == "transformer":
            # Transformer-based fusion
            self.embed_dim = hidden_dim
            
            # Projections to common embedding space
            self.projections = nn.ModuleDict()
            for modality, dim in self.feature_dims.items():
                self.projections[modality] = nn.Linear(dim, self.embed_dim)
            
            # Positional embedding for transformer
            self.position_embedding = transformer_params['position_embedding']
            if self.position_embedding:
                # Simple learnable position embeddings
                self.pos_embedding = nn.Parameter(torch.zeros(1, len(self.modalities), self.embed_dim))
                nn.init.normal_(self.pos_embedding, std=0.02)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=transformer_params['nhead'],
                dim_feedforward=self.embed_dim * 4,
                dropout=transformer_params['dropout'],
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_params['num_layers']
            )
            
            # No additional fusion needed after transformer
            self.fusion = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=dropout,
            shared_layers=shared_classifier_layers
        )
    
    def adaptive_instance_norm(self, content, style):
        """
        Apply Adaptive Instance Normalization.
        
        Args:
            content: Content features [batch_size, feature_dim]
            style: Style features [batch_size, feature_dim]
            
        Returns:
            Normalized features [batch_size, feature_dim]
        """
        # Calculate statistics
        content_mean = content.mean(dim=1, keepdim=True)
        content_std = content.std(dim=1, keepdim=True) + 1e-6  # Add epsilon for stability
        
        style_mean = style.mean(dim=1, keepdim=True)
        style_std = style.std(dim=1, keepdim=True) + 1e-6
        
        # Apply AdaIN
        normalized = (content - content_mean) / content_std
        stylized = normalized * style_std + style_mean
        
        return stylized
    
    def forward(self, **inputs):
        """
        Forward pass with arbitrary modality inputs.
        
        Args:
            **inputs: Modality inputs as keyword arguments (e.g., raw_imu=tensor, video=tensor)
            
        Returns:
            List[torch.Tensor]: List of classification outputs for each prediction horizon
        """
        # Process each available modality
        features = {}
        
        for modality in self.modality_encoders:
            if modality in inputs and inputs[modality] is not None:
                feat = self.modality_encoders[modality].extract_features(inputs[modality])
                
                # Normalize features
                if self.normalize_features:
                    feat = self.modality_norms[modality](feat)
                
                features[modality] = feat
        
        # Check if we have any features
        if not features:
            raise ValueError("No valid inputs were provided for any modality")
        
        # Apply fusion based on method
        if self.fusion_method == "average":
            # Project features to common dimension if needed
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat
            
            # Simple average
            x = torch.stack(list(projected_features.values())).mean(dim=0)
            x = self.fusion(x)
            
        elif self.fusion_method == "maximum":
            # Project features to common dimension if needed
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat
            
            # Element-wise maximum
            x = torch.stack(list(projected_features.values())).max(dim=0)[0]
            x = self.fusion(x)
        
        elif self.fusion_method == "concatenate":
            combined = torch.cat(list(features.values()), dim=1)
            x = self.fusion(combined)

        elif self.fusion_method == "concat_batchnorm":
            combined = torch.cat(list(features.values()), dim=1)
            combined = self.concat_bn(combined)
            x = self.fusion(combined)

        elif self.fusion_method == "concat_layernorm":
            combined = torch.cat(list(features.values()), dim=1)
            combined = self.concat_ln(combined)
            x = self.fusion(combined)
        
        elif self.fusion_method == "dot_attention":
            # Dot product attention between two modalities
            mod_list = list(features.keys())
            feat1 = features[mod_list[0]]
            feat2 = features[mod_list[1]]
            
            # Project to common dimension if needed
            if mod_list[0] in self.projections:
                feat1 = self.projections[mod_list[0]](feat1)
            if mod_list[1] in self.projections:
                feat2 = self.projections[mod_list[1]](feat2)
            
            # Calculate attention weight using dot product
            alpha = (feat1 * feat2).sum(dim=1, keepdim=True)
            alpha = torch.sigmoid(alpha)
            
            # Weighted combination
            x = feat1 * alpha + feat2 * (1 - alpha)
            x = self.fusion(x)
        
        elif self.fusion_method == "mlp_attention":
            # Project features to common dimension if needed
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat
            
            # Apply attention to each modality
            attention_scores = {}
            for modality, feat in features.items():
                attention_scores[modality] = self.attention_layers[modality](feat)
            
            # Normalize attention scores with softmax
            attention_values = torch.cat(list(attention_scores.values()), dim=1)
            attention_weights = nn.functional.softmax(attention_values, dim=1)
            
            # Apply attention weights to features
            attended_features = torch.zeros(
                features[list(features.keys())[0]].size(0),
                self.common_dim,
                device=features[list(features.keys())[0]].device
            )
            
            for i, (modality, feat) in enumerate(projected_features.items()):
                attended_features += feat * attention_weights[:, i].unsqueeze(1)
            
            x = self.fusion(attended_features)
        
        elif self.fusion_method == "adain_content_style":
            # First modality as content, second as style
            mod_list = list(features.keys())
            content = features[mod_list[0]]
            style = features[mod_list[1]]
            
            # Project to common dimension if needed
            if mod_list[0] in self.projections:
                content = self.projections[mod_list[0]](content)
            if mod_list[1] in self.projections:
                style = self.projections[mod_list[1]](style)
            
            # Apply AdaIN
            x = self.adaptive_instance_norm(content, style)
            x = self.fusion(x)
        
        elif self.fusion_method == "adain_style_content":
            # First modality as style, second as content
            mod_list = list(features.keys())
            style = features[mod_list[0]]
            content = features[mod_list[1]]
            
            # Project to common dimension if needed
            if mod_list[0] in self.projections:
                style = self.projections[mod_list[0]](style)
            if mod_list[1] in self.projections:
                content = self.projections[mod_list[1]](content)
            
            # Apply AdaIN
            x = self.adaptive_instance_norm(content, style)
            x = self.fusion(x)
        
        elif self.fusion_method == "weighted":
            # Project features to common dimension if needed
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat
            
            # Get weights for present modalities
            present_modalities = list(features.keys())
            weight_values = torch.stack([self.modality_weights[m] for m in present_modalities])
            normalized_weights = nn.functional.softmax(weight_values, dim=0)
            
            # Apply weighted sum
            weighted_sum = torch.zeros(
                features[list(features.keys())[0]].size(0),
                self.common_dim,
                device=features[list(features.keys())[0]].device
            )
            
            for i, modality in enumerate(present_modalities):
                modality_weight = normalized_weights[i]
                weighted_sum += projected_features[modality] * modality_weight
            
            x = self.fusion(weighted_sum)
            
        elif self.fusion_method == "transformer":
            # Project each modality's features to common dimension
            # and collect as a sequence of tokens
            batch_size = list(features.values())[0].size(0)
            modality_tokens = []
            modality_indices = []  # Keep track of which modalities are present
            
            for i, modality in enumerate(self.modalities):
                if modality in features:
                    # Project to common dimension
                    token = self.projections[modality](features[modality])
                    modality_tokens.append(token)
                    modality_indices.append(i)
            
            # Stack tokens along sequence dimension
            tokens = torch.stack(modality_tokens, dim=1)  # [batch_size, num_modalities, embed_dim]
            
            # Add position embeddings if enabled
            if self.position_embedding:
                # Only use position embeddings for modalities that are present
                pos_embed = self.pos_embedding[:, modality_indices, :]
                tokens = tokens + pos_embed
            
            # Apply transformer encoder
            encoded = self.transformer_encoder(tokens)
            
            # Global pooling across modalities (mean pooling)
            x = encoded.mean(dim=1)  # [batch_size, embed_dim]
        
        # Multi-horizon classification
        outputs = self.classifier(x)
        
        return outputs
        
    def get_prediction_horizons(self) -> List[float]:
        """Get the prediction horizons for this model."""
        return self.prediction_horizons.copy()

    def get_num_prediction_heads(self) -> int:
        """Get the number of prediction heads."""
        return self.num_prediction_heads

    def encode_features(self, **inputs) -> torch.Tensor:
        """
        Extract fused features before classification head.

        This method is used for feature-based knowledge distillation.
        Returns the fused feature representation after the fusion layer
        but before the classification heads.

        Args:
            **inputs: Modality inputs as keyword arguments

        Returns:
            torch.Tensor: Fused features [batch_size, hidden_dim]
        """
        # Process each available modality
        features = {}

        for modality in self.modality_encoders:
            if modality in inputs and inputs[modality] is not None:
                feat = self.modality_encoders[modality].extract_features(inputs[modality])

                # Normalize features
                if self.normalize_features:
                    feat = self.modality_norms[modality](feat)

                features[modality] = feat

        # Check if we have any features
        if not features:
            raise ValueError("No valid inputs were provided for any modality")

        # Apply fusion based on method (same logic as forward, but return before classifier)
        if self.fusion_method == "average":
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat
            x = torch.stack(list(projected_features.values())).mean(dim=0)
            x = self.fusion(x)

        elif self.fusion_method == "maximum":
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat
            x = torch.stack(list(projected_features.values())).max(dim=0)[0]
            x = self.fusion(x)

        elif self.fusion_method == "concatenate":
            combined = torch.cat(list(features.values()), dim=1)
            x = self.fusion(combined)

        elif self.fusion_method == "concat_batchnorm":
            combined = torch.cat(list(features.values()), dim=1)
            combined = self.concat_bn(combined)
            x = self.fusion(combined)

        elif self.fusion_method == "concat_layernorm":
            combined = torch.cat(list(features.values()), dim=1)
            combined = self.concat_ln(combined)
            x = self.fusion(combined)

        elif self.fusion_method == "dot_attention":
            mod_list = list(features.keys())
            feat1 = features[mod_list[0]]
            feat2 = features[mod_list[1]]

            if mod_list[0] in self.projections:
                feat1 = self.projections[mod_list[0]](feat1)
            if mod_list[1] in self.projections:
                feat2 = self.projections[mod_list[1]](feat2)

            alpha = (feat1 * feat2).sum(dim=1, keepdim=True)
            alpha = torch.sigmoid(alpha)
            x = feat1 * alpha + feat2 * (1 - alpha)
            x = self.fusion(x)

        elif self.fusion_method == "mlp_attention":
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat

            attention_scores = {}
            for modality, feat in features.items():
                attention_scores[modality] = self.attention_layers[modality](feat)

            attention_values = torch.cat(list(attention_scores.values()), dim=1)
            attention_weights = nn.functional.softmax(attention_values, dim=1)

            attended_features = torch.zeros(
                features[list(features.keys())[0]].size(0),
                self.common_dim,
                device=features[list(features.keys())[0]].device
            )

            for i, (modality, feat) in enumerate(projected_features.items()):
                attended_features += feat * attention_weights[:, i].unsqueeze(1)

            x = self.fusion(attended_features)

        elif self.fusion_method == "adain_content_style":
            mod_list = list(features.keys())
            content = features[mod_list[0]]
            style = features[mod_list[1]]

            if mod_list[0] in self.projections:
                content = self.projections[mod_list[0]](content)
            if mod_list[1] in self.projections:
                style = self.projections[mod_list[1]](style)

            x = self.adaptive_instance_norm(content, style)
            x = self.fusion(x)

        elif self.fusion_method == "adain_style_content":
            mod_list = list(features.keys())
            style = features[mod_list[0]]
            content = features[mod_list[1]]

            if mod_list[0] in self.projections:
                style = self.projections[mod_list[0]](style)
            if mod_list[1] in self.projections:
                content = self.projections[mod_list[1]](content)

            x = self.adaptive_instance_norm(content, style)
            x = self.fusion(x)

        elif self.fusion_method == "weighted":
            projected_features = {}
            for modality, feat in features.items():
                if modality in self.projections:
                    projected_features[modality] = self.projections[modality](feat)
                else:
                    projected_features[modality] = feat

            present_modalities = list(features.keys())
            weight_values = torch.stack([self.modality_weights[m] for m in present_modalities])
            normalized_weights = nn.functional.softmax(weight_values, dim=0)

            weighted_sum = torch.zeros(
                features[list(features.keys())[0]].size(0),
                self.common_dim,
                device=features[list(features.keys())[0]].device
            )

            for i, modality in enumerate(present_modalities):
                modality_weight = normalized_weights[i]
                weighted_sum += projected_features[modality] * modality_weight

            x = self.fusion(weighted_sum)

        elif self.fusion_method == "transformer":
            batch_size = list(features.values())[0].size(0)
            modality_tokens = []
            modality_indices = []

            for i, modality in enumerate(self.modalities):
                if modality in features:
                    token = self.projections[modality](features[modality])
                    modality_tokens.append(token)
                    modality_indices.append(i)

            tokens = torch.stack(modality_tokens, dim=1)

            if self.position_embedding:
                pos_embed = self.pos_embedding[:, modality_indices, :]
                tokens = tokens + pos_embed

            encoded = self.transformer_encoder(tokens)
            x = encoded.mean(dim=1)

        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        return x

    def get_feature_dim(self) -> int:
        """
        Get the dimension of fused features (output of encode_features).

        Returns:
            int: Feature dimension (hidden_dim from fusion layer)
        """
        # The fusion layer outputs hidden_dim features
        if self.fusion_method == "transformer":
            return self.embed_dim
        else:
            # For other methods, fusion outputs hidden_dim
            # Get it from the first layer of fusion Sequential
            if hasattr(self.fusion, '__getitem__'):
                first_layer = self.fusion[0]
                if hasattr(first_layer, 'out_features'):
                    return first_layer.out_features
            # Default fallback - check classifier input
            return self.classifier.shared_fc[0].in_features if hasattr(self.classifier, 'shared_fc') else 512