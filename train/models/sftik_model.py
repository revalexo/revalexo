# models/sftik_model.py
# https://github.com/RuoqiZhao116/SFTIK

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base_models import BaseEncoder, MultiHorizonClassifier

import collections.abc
from itertools import repeat
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union
from torch import _assert
from timm.models.vision_transformer import PatchEmbed, Block

class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x

class CustomPatchEmbed(nn.Module):
    """ Custom 2D Image to Patch Embedding for vertical strips """
    def __init__(
            self,
            img_size: int = 224,
            patch_width: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)  # Ensure img_size is a tuple
        self.patch_size = (self.img_size[0], patch_width)  # Patch size (224,16)
        self.grid_size = (self.img_size[1] // patch_width, 1)  # Grid size in width, fixed height
        self.num_patches = self.grid_size[0]

        self.flatten = flatten
        self.output_fmt = Format(output_fmt) if output_fmt is not None else Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        # Convolution to create patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=(self.img_size[0],patch_width), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Checking image size
        if self.strict_img_size:
            _assert(H == self.img_size[0] and W == self.img_size[1], f"Input size doesn't match model ({self.img_size}).")
        elif not self.dynamic_img_pad:
            _assert(H == self.img_size[0], f"Input height ({H}) should match patch height ({self.img_size[0]}).")
            _assert(W % self.patch_size[0] == 0, f"Input width ({W}) should be divisible by patch width ({self.patch_size[0]}).")

        # Apply padding if necessary
        if self.dynamic_img_pad:
            pad_w = (self.patch_size[0] - W % self.patch_size[0]) % self.patch_size[0]
            x = F.pad(x, (0, pad_w, 0, 0))

        # Projecting to patches
        x = self.proj(x)
        
        # Flatten and transpose if necessary
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        
        # Apply normalization
        x = self.norm(x)
        return x

class SFTIK(BaseEncoder):
    """
    SFTIK model adapted for multi-horizon classification.
    Key differences from original:
    1. No angle channel (we only use IMU data)
    2. Classification instead of regression
    3. Multi-horizon support
    """
    
    def __init__(self,
                 c_in: int,  # Number of IMU channels (no angle)
                 context_window: int,
                 target_window: int,  # Not used but kept for compatibility
                 patch_len: int,
                 stride: int,
                 embed_dim: int = 768,
                 pre_depth: int = 6,
                 late_depth: int = 6,
                 n_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 norm_layer=nn.LayerNorm,
                 num_classes: int = 13,
                 feature_dim: int = 256,
                 prediction_horizons: List[float] = [0, 0.1, 0.2, 0.3, 0.5, 1],
                 dropout: float = 0.2,
                 **kwargs):
        
        super().__init__(feature_dim=feature_dim, prediction_horizons=prediction_horizons)
        
        # Backbone - matching original SFTIK
        self.patch_len = patch_len
        self.stride = stride
        self.context_window = context_window
        
        self.ts_patch_num = int((context_window - patch_len) / stride + 1)
        self.W_P = nn.Linear(c_in * patch_len, embed_dim)
        self.ts_pos_embed = nn.Parameter(torch.zeros(1, self.ts_patch_num, embed_dim))
        
        self.img_patch = CustomPatchEmbed(patch_width=16, in_chans=3, embed_dim=embed_dim)
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.img_patch.num_patches, embed_dim))
        
        # Transformer blocks - exactly as in SFTIK
        self.former_blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(pre_depth)
        ])
        
        self.single_blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(pre_depth)
        ])
        
        self.later_blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(late_depth)
        ])
        
        # Output Head - modified for classification
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, feature_dim)
        )
        
        # Multi-horizon classifier
        self.classifier = MultiHorizonClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=0.5
        )
    
    def encode_features(self, pre_imu, image1, image2):
        """
        Extract features matching SFTIK's forward method.
        
        Args:
            pre_imu: IMU data [bs, channels, context_window]
            image1: First image [bs, 1, 224, 224] 
            image2: Second image [bs, 1, 224, 224]
        """
        # IMU Time Series Embedding with position encoding
        z = pre_imu
        
        # Create patches
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 2, 1, 3)  # [bs, patch_num, nvars, patch_len]
        bs, patch_num, _, _ = z.size()
        z_flattened = z.reshape(bs, patch_num, -1)
        z = self.W_P(z_flattened)
        ts_patches = z + self.ts_pos_embed
        
        # Image Embedding with position encoding
        img1_patches = self.img_patch(image1) + self.img_pos_embed
        img2_patches = self.img_patch(image2) + self.img_pos_embed
        
        # Early Fusion (sandwich mechanism)
        mix_patches = torch.cat([ts_patches, img1_patches], dim=1)
        
        for blk in self.former_blocks:
            mix_patches = blk(mix_patches)
        
        for blk in self.single_blocks:
            img2_patches = blk(img2_patches)
        
        fuse_patches = torch.cat([mix_patches, img2_patches], dim=1)
        
        for blk in self.later_blocks:
            fuse_patches = blk(fuse_patches)
        
        pooled = torch.mean(fuse_patches, dim=1)
        
        output = self.ffn(pooled)
        return output
    
    def forward(self, pre_imu, image1, image2):
        """Forward pass for multi-horizon classification."""
        features = self.encode_features(pre_imu, image1, image2)
        outputs = self.classifier(features)
        return outputs


# DUMMY ENCODERS FOR FRAMEWORK COMPATIBILITY

class SFTIK_IMUEncoder(BaseEncoder):
    """Dummy IMU encoder for framework compatibility."""
    
    def __init__(self, channels=102, window_size=120, feature_dim=256, 
                 prediction_horizons=[0], **kwargs):
        super().__init__(feature_dim=feature_dim, prediction_horizons=prediction_horizons)
        self.channels = channels
        self.window_size = window_size
        # Dummy projection
        self.proj = nn.Linear(channels * window_size, feature_dim)
    
    def encode_features(self, x):
        # This won't actually be used, but needs to exist
        B = x.shape[0] if len(x.shape) > 0 else 1
        return torch.zeros(B, self.feature_dim, device=x.device if hasattr(x, 'device') else 'cpu')
    
    def forward(self, x):
        features = self.encode_features(x)
        # Return dummy multi-horizon outputs if needed
        return [torch.zeros(x.shape[0], 13, device=x.device) for _ in range(6)]


class SFTIK_VideoEncoder(BaseEncoder):
    """Dummy video encoder for framework compatibility."""
    
    def __init__(self, feature_dim=256, prediction_horizons=[0], **kwargs):
        super().__init__(feature_dim=feature_dim, prediction_horizons=prediction_horizons)
        # Dummy projection
        self.proj = nn.Conv2d(3, feature_dim, kernel_size=1)
    
    def encode_features(self, x):
        # This won't actually be used, but needs to exist
        B = x.shape[0] if len(x.shape) > 0 else 1
        return torch.zeros(B, self.feature_dim, device=x.device if hasattr(x, 'device') else 'cpu')
    
    def forward(self, x):
        features = self.encode_features(x)
        # Return dummy multi-horizon outputs if needed
        return [torch.zeros(x.shape[0], 13, device=x.device) for _ in range(6)]