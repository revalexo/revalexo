# transforms/__init__.py
from .imu_transforms import *

import numpy as np
import random
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from typing import List, Tuple, Union, Optional, Sequence

class Compose:
    """Composes several transforms together."""
    
    def __init__(self, transforms):
        """
        Initialize composed transforms.
        
        Args:
            transforms (list): List of transforms to compose
        """
        self.transforms = transforms
    
    def __call__(self, data):
        """
        Apply transforms sequentially.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        for transform in self.transforms:
            data = transform(data)
        return data

# ===== IMU Transforms =====

class NormalizeIMU:
    """Normalize IMU data with proper handling of zero variance channels."""
    
    def __init__(self, method="standard", eps=1e-8):
        """
        Initialize normalization transform.
        
        Args:
            method (str): Normalization method ('standard' or 'minmax')
            eps (float): Small value to prevent division by zero
        """
        self.method = method
        self.eps = eps
    
    def __call__(self, data):
        """
        Apply normalization.
        
        Args:
            data (np.ndarray): IMU data to normalize
            
        Returns:
            np.ndarray: Normalized IMU data
        """
        # Handle empty data
        if data.size == 0:
            return data
            
        if self.method == "standard":
            # Z-score normalization with protection against zero variance
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
                
                # Replace zero std with eps to avoid division by zero
                std = np.where(std < self.eps, self.eps, std)
                
                normalized = (data - mean) / std
                
                # Replace any NaN or inf values with 0
                normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
                
            return normalized
        
        elif self.method == "minmax":
            # Min-max normalization with protection
            with np.errstate(divide='ignore', invalid='ignore'):
                min_vals = np.min(data, axis=0, keepdims=True)
                max_vals = np.max(data, axis=0, keepdims=True)
                range_vals = max_vals - min_vals
                
                # Replace zero range with eps
                range_vals = np.where(range_vals < self.eps, self.eps, range_vals)
                
                normalized = (data - min_vals) / range_vals
                
                # Replace any NaN or inf values with 0
                normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
                
            return normalized
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

class AddGaussianNoise:
    """Add Gaussian noise to IMU data."""
    
    def __init__(self, mean=0.0, std=0.01, probability=0.5):
        """
        Initialize Gaussian noise transform.
        
        Args:
            mean (float): Mean of the Gaussian noise
            std (float): Standard deviation of the Gaussian noise
            probability (float): Probability of applying the transform
        """
        self.mean = mean
        self.std = std
        self.probability = probability
    
    def __call__(self, data):
        """
        Apply Gaussian noise.
        
        Args:
            data (np.ndarray): IMU data
            
        Returns:
            np.ndarray: IMU data with noise
        """
        if random.random() < self.probability:
            noise = np.random.normal(self.mean, self.std, data.shape).astype(data.dtype)
            return data + noise
        return data

class Scale:
    """Scale the magnitude of the time-series data."""
    def __init__(self, factor_range=(0.8, 1.2), probability=0.5):
        """
        Args:
            factor_range (tuple): Min and max scaling factor.
            probability (float): Probability of applying the transform.
        """
        self.factor_range = factor_range
        self.probability = probability

    def __call__(self, data):
        """
        Args:
            data (np.ndarray): Input data (time_steps, channels)
        Returns:
            np.ndarray: Scaled data or original data
        """
        if random.random() < self.probability:
            scale_factor = random.uniform(self.factor_range[0], self.factor_range[1])
            return (data * scale_factor).astype(data.dtype)
        return data

class SwapIMUChannels:
    """Correct IMU data dimensions."""
    
    def __call__(self, data):
        """
        Apply dimension correction.
        
        Args:
            data (np.ndarray): IMU data
        """
        return data.T


# =============================================================================
# Video Transforms (Using PyTorchVideo and TorchVision)
# =============================================================================

# Import transforms from existing libraries
try:
    from pytorchvideo.transforms import (
        UniformTemporalSubsample,
        ShortSideScale,
        ConvertUint8ToFloat,
        ConvertFloatToUint8,
    )
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False
    print("Warning: PyTorchVideo not available. Some video transforms will not work.")

try:
    from torchvision.transforms._transforms_video import (
        CenterCropVideo,
        NormalizeVideo,
        ToTensorVideo,
        RandomHorizontalFlipVideo,
        RandomResizedCropVideo,
    )
    TORCHVISION_VIDEO_AVAILABLE = True
except ImportError:
    TORCHVISION_VIDEO_AVAILABLE = False
    print("Warning: TorchVision video transforms not available. Some video transforms will not work.")


# =============================================================================
# Simple wrapper for division by 255
# =============================================================================

class DivideBy255:
    """Divide video tensor by 255 to normalize from [0, 255] to [0, 1]."""
    def __init__(self):
        pass

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video / 255.0

    def __repr__(self) -> str:
        return "DivideBy255()"


# The imports above already make the transforms available in the module namespace
# No need for additional globals() assignments

class Identity:
    """Identity transform that returns input unchanged."""
    
    def __init__(self):
        pass
    
    def __call__(self, data):
        """
        Return data unchanged.
        
        Args:
            data: Input data
            
        Returns:
            Input data unchanged
        """
        return data
    
    def __repr__(self):
        return "Identity()"


# Add these transforms to transforms/__init__.py

# =============================================================================
# Image Transforms (for single frame)
# =============================================================================

class ExtractCenterFrame:
    """Extract the center frame from a video tensor."""
    def __init__(self):
        pass
    
    def __call__(self, video):
        """
        Extract center frame from video tensor.
        
        Args:
            video: Tensor of shape [C, T, H, W] or [T, H, W, C]
            
        Returns:
            Tensor of shape [C, H, W] (single frame)
        """
        if not isinstance(video, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(video)}")
            
        if video.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {video.dim()}D")
        
        if video.shape[0] in [1, 3, 4] and video.shape[1] > 4:
            # [C, T, H, W]
            center_idx = video.shape[1] // 2
            return video[:, center_idx, :, :]
        elif video.shape[-1] in [1, 3, 4] and video.shape[0] > 4:
            # [T, H, W, C]
            center_idx = video.shape[0] // 2
            frame = video[center_idx]
            return frame.permute(2, 0, 1)  # Convert to [C, H, W]
        else:
            raise ValueError(f"Cannot determine video format for shape {video.shape}. "
                        f"Expected [C, T, H, W] or [T, H, W, C]")
    
    def __repr__(self):
        return "ExtractCenterFrame()"

class ResizeImage:
    """Resize image to given size."""
    def __init__(self, size):
        """
        Args:
            size: Desired output size. If int, resize smaller edge to size keeping aspect ratio.
                  If tuple/list, resize to exact size.
        """
        self.size = size
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor [C, H, W]
            
        Returns:
            Resized image
        """
        if isinstance(img, torch.Tensor):
            return F.resize(img, self.size, interpolation=F.InterpolationMode.BILINEAR)
        else:
            return F.resize(img, self.size, interpolation=Image.Resampling.BILINEAR)
    
    def __repr__(self):
        return f"ResizeImage(size={self.size})"

class CenterCropImage:
    """Center crop image to given size."""
    def __init__(self, size):
        """
        Args:
            size: Desired output size (int or tuple)
        """
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor [C, H, W]
            
        Returns:
            Center cropped image
        """
        return F.center_crop(img, self.size)
    
    def __repr__(self):
        return f"CenterCropImage(size={self.size})"

class RandomResizedCropImage:
    """Random resized crop for image augmentation."""
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        """
        Args:
            size: Expected output size
            scale: Range of size of the origin size cropped
            ratio: Range of aspect ratio of the origin aspect ratio cropped
        """
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor [C, H, W]
            
        Returns:
            Randomly cropped and resized image
        """
        return F.resized_crop(
            img, 
            *T.RandomResizedCrop.get_params(img, self.scale, self.ratio),
            self.size,
            interpolation=F.InterpolationMode.BILINEAR
        )
    
    def __repr__(self):
        return f"RandomResizedCropImage(size={self.size}, scale={self.scale}, ratio={self.ratio})"

class RandomHorizontalFlipImage:
    """Randomly flip image horizontally."""
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor [C, H, W]
            
        Returns:
            Possibly flipped image
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img
    
    def __repr__(self):
        return f"RandomHorizontalFlipImage(p={self.p})"

class NormalizeImage:
    """Normalize image tensor with mean and std."""
    def __init__(self, mean, std):
        """
        Args:
            mean: Sequence of means for each channel
            std: Sequence of standard deviations for each channel
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Args:
            img: Tensor [C, H, W]
            
        Returns:
            Normalized tensor
        """
        return F.normalize(img, mean=self.mean, std=self.std)
    
    def __repr__(self):
        return f"NormalizeImage(mean={self.mean}, std={self.std})"

class ToTensorImage:
    """Convert PIL Image to tensor."""
    def __init__(self):
        pass
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
            
        Returns:
            Tensor [C, H, W] with values in [0, 1]
        """
        return F.to_tensor(img)
    
    def __repr__(self):
        return "ToTensorImage()"

# Update __all__ to include new transforms
__all__ = [
    'Compose',
    # IMU transforms
    'NormalizeIMU',
    'AddGaussianNoise',
    'Scale',
    'SwapIMUChannels',
    # Video transforms
    'DivideBy255',
    'Identity',
    # Image transforms
    'ExtractCenterFrame',
    'ResizeImage',
    'CenterCropImage',
    'RandomResizedCropImage',
    'RandomHorizontalFlipImage',
    'NormalizeImage',
    'ToTensorImage', 
    'AdaptiveIMUChannelMapping',
    'EVI_MAE_IMUTransform',
    'IMUChannelSelector',
    'IMUToSpectrogram'
]

# Add exports to __all__ if available
if PYTORCHVIDEO_AVAILABLE:
    __all__.extend([
        'UniformTemporalSubsample',
        'ShortSideScale', 
        'ConvertUint8ToFloat',
        'ConvertFloatToUint8'
    ])

if TORCHVISION_VIDEO_AVAILABLE:
    __all__.extend([
        'CenterCropVideo',
        'RandomHorizontalFlipVideo',
        'ToTensorVideo',
        'NormalizeVideo',
        'RandomResizedCropVideo'
    ])