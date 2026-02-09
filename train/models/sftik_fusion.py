# models/sftik_fusion.py
# https://github.com/RuoqiZhao116/SFTIK

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .sftik_model import SFTIK

class SFTIK_Fusion(nn.Module):
    """
    Wrapper to integrate SFTIK with your multimodal framework.
    """
    
    def __init__(self,
                 modality_encoders: Dict = None,
                 num_classes: int = 13,
                 prediction_horizons: List[float] = [0, 0.1, 0.2, 0.3, 0.5, 1],
                 window_size: int = 120,
                 imu_channels: int = 102,
                 patch_len: int = 12,
                 stride: int = 12,
                 embed_dim: int = 768,
                 pre_depth: int = 6,
                 late_depth: int = 6,
                 feature_dim: int = 256,
                 dropout: float = 0.2,
                 n_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 shared_classifier_layers: bool = True,
                 **kwargs):
        
        super().__init__()
        
        # Store dummy encoders if provided
        self.modality_encoders = nn.ModuleDict(modality_encoders) if modality_encoders else None
        
        # Remove duplicate parameters from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['n_heads', 'mlp_ratio', 'shared_classifier_layers']}
        
        # Create the actual SFTIK model
        self.model = SFTIK(
            c_in=imu_channels,
            context_window=window_size,
            target_window=window_size,
            patch_len=patch_len,
            stride=stride,
            embed_dim=embed_dim,
            pre_depth=pre_depth,
            late_depth=late_depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            num_classes=num_classes,
            feature_dim=feature_dim,
            prediction_horizons=prediction_horizons,
            dropout=dropout,
            **filtered_kwargs  # Pass only non-duplicate kwargs
        )
        
        self.window_size = window_size
        self.imu_channels = imu_channels
        self.modalities = ['raw_imu', 'video']
        self.prediction_horizons = prediction_horizons
        self.num_prediction_heads = len(prediction_horizons)
    
    def forward(self, **inputs):
        """Convert your data format to SFTIK format."""
        imu_data = inputs.get('raw_imu')
        video_data = inputs.get('video')
        
        if imu_data is None or video_data is None:
            raise ValueError("SFTIK requires both IMU and video data")
        
        # Ensure batch dimension
        if len(imu_data.shape) == 2:
            imu_data = imu_data.unsqueeze(0)
        
        # Convert to [batch, channels, time]
        if imu_data.shape[-1] != self.window_size:
            if imu_data.shape[1] == self.window_size:
                imu_data = imu_data.transpose(1, 2)
        
        batch_size = imu_data.shape[0]
        
        # Process video to get two frames (first and last)
        if len(video_data.shape) == 5:  # [batch, C, T, H, W]
            first_frame = video_data[:, :, 0, :, :]  # First frame
            last_frame = video_data[:, :, -1, :, :]  # Last frame
            
            # Convert to grayscale if RGB (to match depth?) If using their SFTIK RGB implementation, no need to do this
            # if first_frame.shape[1] == 3:
            #     weights = torch.tensor([0.299, 0.587, 0.114], device=first_frame.device)
            #     first_frame = torch.sum(first_frame * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
            #     last_frame = torch.sum(last_frame * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
            
            # Ensure correct size
            if first_frame.shape[-1] != 224 or first_frame.shape[-2] != 224:
                first_frame = F.interpolate(first_frame, size=(224, 224), mode='bilinear', align_corners=False)
                last_frame = F.interpolate(last_frame, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unexpected video shape: {video_data.shape}")
        
        # Forward through model
        return self.model(imu_data, first_frame, last_frame)
    
    def get_num_prediction_heads(self) -> int:
        return self.num_prediction_heads
    
    def get_prediction_horizons(self) -> List[float]:
        return self.prediction_horizons.copy()