# src/transforms/__init__.py

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


class TimeWarpingIMU:
    """Apply time warping to IMU data."""
    def __init__(self, sigma=0.2, knot=4, p=0.5):
        self.sigma = sigma
        self.knot = knot
        self.p = p
        
    def __call__(self, data):
        """data: [time_steps, channels]"""
        if random.random() < self.p:
            time_steps, channels = data.shape
            
            # Create original time grid
            orig_steps = np.arange(time_steps)
            
            # Create sparse control points for warping
            warp_control_points = np.linspace(0, time_steps-1, num=self.knot+2)
            
            # Generate random warping at control points
            random_warp_values = np.random.normal(loc=1.0, scale=self.sigma, size=self.knot+2)
            random_warp_values[0] = 1.0  # Fix endpoints
            random_warp_values[-1] = 1.0
            
            # Cumulative product to create smooth warping
            warp_cumulative = np.cumprod(random_warp_values)
            
            # Normalize to maintain the same total duration
            warp_scale = warp_cumulative / warp_cumulative[-1]
            
            # Create the warped time indices
            warped_indices = np.zeros(time_steps)
            warped_indices[0] = 0
            for i in range(1, time_steps):
                # Interpolate the warping scale at this time point
                scale = np.interp(i, warp_control_points, warp_scale)
                warped_indices[i] = warped_indices[i-1] + scale
            
            # Normalize to fit in [0, time_steps-1]
            warped_indices = warped_indices / warped_indices[-1] * (time_steps - 1)
            
            # Apply warping by resampling
            warped = np.zeros_like(data)
            for c in range(channels):
                warped[:, c] = np.interp(orig_steps, warped_indices, data[:, c])
            
            return warped.astype(data.dtype)
        return data

class ChannelDropoutIMU:
    """Randomly drop IMU channels."""
    def __init__(self, p=0.2, channel_drop_prob=0.1):
        self.p = p
        self.channel_drop_prob = channel_drop_prob
        
    def __call__(self, data):
        """data: [time_steps, channels]"""
        if random.random() < self.p:
            time_steps, channels = data.shape
            # Randomly select channels to drop
            channel_mask = np.random.random(channels) > self.channel_drop_prob
            data_copy = data.copy()
            data_copy[:, ~channel_mask] = 0
            return data_copy
        return data

class MagnitudeWarpingIMU:
    """Apply magnitude warping to IMU data."""
    def __init__(self, sigma=0.2, knot=4, p=0.5):
        self.sigma = sigma
        self.knot = knot
        self.p = p
        
    def __call__(self, data):
        """data: [time_steps, channels]"""
        if random.random() < self.p:
            time_steps, channels = data.shape
            
            # Generate smooth random curves for magnitude scaling
            orig_steps = np.linspace(0, time_steps-1, num=self.knot+2)
            random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot+2, channels))
            
            # Interpolate to all time steps
            warp_steps = np.arange(time_steps)
            warps = np.zeros((time_steps, channels))
            
            for c in range(channels):
                warps[:, c] = np.interp(warp_steps, orig_steps, random_warps[:, c])
            
            return (data * warps).astype(data.dtype)
        return data



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


# ============= Image Augmentations =============

class ColorJitterImage:
    """Apply color jitter to image."""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # Random order of transforms
            transforms = []
            if self.brightness > 0:
                brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                transforms.append(lambda x: F.adjust_brightness(x, brightness_factor))
            if self.contrast > 0:
                contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                transforms.append(lambda x: F.adjust_contrast(x, contrast_factor))
            if self.saturation > 0:
                saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                transforms.append(lambda x: F.adjust_saturation(x, saturation_factor))
            if self.hue > 0:
                hue_factor = random.uniform(-self.hue, self.hue)
                transforms.append(lambda x: F.adjust_hue(x, hue_factor))
            
            random.shuffle(transforms)
            for t in transforms:
                img = t(img)
        return img

class RandomErasingImage:
    """Randomly erase rectangular regions in image."""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, torch.Tensor):
                return self._erase_tensor(img)
            else:
                # Convert to tensor, erase, convert back
                img_tensor = F.to_tensor(img)
                erased = self._erase_tensor(img_tensor)
                return F.to_pil_image(erased)
        return img
    
    def _erase_tensor(self, img):
        """Erase on tensor format [C, H, W]"""
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        
        c, h, w = img.shape
        area = h * w
        
        for _ in range(10):  # Try 10 times
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h_erase = int(round(np.sqrt(erase_area * aspect_ratio)))
            w_erase = int(round(np.sqrt(erase_area / aspect_ratio)))
            
            if h_erase < h and w_erase < w:
                i = random.randint(0, h - h_erase)
                j = random.randint(0, w - w_erase)
                
                if self.value == 'random':
                    img[:, i:i+h_erase, j:j+w_erase] = torch.rand_like(img[:, i:i+h_erase, j:j+w_erase])
                else:
                    img[:, i:i+h_erase, j:j+w_erase] = self.value
                return img
        return img

class RandomGrayscaleImage:
    """Randomly convert image to grayscale."""
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            return F.rgb_to_grayscale(img, num_output_channels=3)
        return img

class GaussianBlurImage:
    """Apply Gaussian blur to image."""
    def __init__(self, kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            kernel_size = random.randrange(self.kernel_size[0], self.kernel_size[1] + 1, 2)
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        return img

class ColorJitterVideo:
    """Apply color jitter to video consistently across frames."""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        
    def __call__(self, video):
        """video: [C, T, H, W]"""
        if random.random() < self.p:
            # Sample parameters once for the whole video
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            hue_factor = random.uniform(-self.hue, self.hue)
            
            # Apply to each frame
            c, t, h, w = video.shape
            for i in range(t):
                if self.brightness > 0:
                    video[:, i] = F.adjust_brightness(video[:, i], brightness_factor)
                if self.contrast > 0:
                    video[:, i] = F.adjust_contrast(video[:, i], contrast_factor)
                if self.saturation > 0:
                    video[:, i] = F.adjust_saturation(video[:, i], saturation_factor)
                if self.hue > 0:
                    video[:, i] = F.adjust_hue(video[:, i], hue_factor)
        return video

class RandomGrayscaleVideo:
    """Randomly convert video to grayscale."""
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, video):
        """video: [C, T, H, W]"""
        if random.random() < self.p:
            c, t, h, w = video.shape
            for i in range(t):
                video[:, i] = F.rgb_to_grayscale(video[:, i], num_output_channels=3)
        return video

class RandomErasingVideo:
    """Randomly erase rectangular regions consistently across video frames."""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, video):
        """video: [C, T, H, W]"""
        if random.random() < self.p:
            c, t, h, w = video.shape
            area = h * w
            
            # Sample erase parameters once
            for _ in range(10):
                erase_area = random.uniform(self.scale[0], self.scale[1]) * area
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
                
                h_erase = int(round(np.sqrt(erase_area * aspect_ratio)))
                w_erase = int(round(np.sqrt(erase_area / aspect_ratio)))
                
                if h_erase < h and w_erase < w:
                    i = random.randint(0, h - h_erase)
                    j = random.randint(0, w - w_erase)
                    
                    # Apply same erasure to all frames
                    erase_value = torch.rand(c, 1, h_erase, w_erase)
                    for frame_idx in range(t):
                        video[:, frame_idx, i:i+h_erase, j:j+w_erase] = erase_value.squeeze(1)
                    break
        return video

# Update __all__ to include new transforms
__all__ = [
    'Compose',
    # IMU transforms
    'NormalizeIMU',
    'AddGaussianNoise',
    'Scale',
    'SwapIMUChannels',
    'TimeWarpingIMU',
    'ChannelDropoutIMU',
    'MagnitudeWarpingIMU',
    # Video transforms
    'DivideBy255',
    'Identity',
    'TimeWarpingVideo',
    'ChannelDropoutVideo',
    'MagnitudeWarpingVideo',
    # Image transforms
    'ExtractCenterFrame',
    'ResizeImage',
    'CenterCropImage',
    'RandomResizedCropImage',
    'RandomHorizontalFlipImage',
    'NormalizeImage',
    'ToTensorImage',
    'RandomErasingImage'
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