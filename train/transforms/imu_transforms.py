# transforms/imu_transforms.py
"""
IMU data transformations for EVI-MAE model.
Converts raw IMU data to STFT spectrograms.
Fixed for AidWear dataset with proper channel mapping and STFT parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from typing import Optional, Tuple, List, Dict


class IMUToSpectrogram(nn.Module):
    """
    Transform raw IMU data to STFT spectrograms for EVI-MAE model.
    
    This transform:
    1. Selects appropriate IMU channels (4 IMUs: left arm, right arm, left leg, right leg)
    2. Extracts only accelerometer data (3 axes per IMU)
    3. Resamples to match EVI-MAE expected length
    4. Converts to STFT spectrograms
    5. Normalizes the spectrograms
    """
    def __init__(
        self,
        n_fft: int = 256,  # Original EVI-MAE parameter
        win_length: int = 24,  # Original EVI-MAE parameter
        hop_length: int = 1,
        target_length: int = 320,
        target_height: int = 128,
        norm_mean: float = -54.33,  # WEAR dataset mean
        norm_std: float = 26.04,     # WEAR dataset std
        sampling_rate: int = 60,     # IMU sampling rate
        target_sampling_rate: int = 125,  # EVI-MAE expects 125Hz for WEAR
        use_log_scale: bool = True,
        channel_mapping: Optional[Dict[str, List[int]]] = None,
        expected_signal_length: int = 250  # EVI-MAE expects 250 samples
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.target_length = target_length
        self.target_height = target_height
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sampling_rate = sampling_rate
        self.target_sampling_rate = target_sampling_rate
        self.use_log_scale = use_log_scale
        self.expected_signal_length = expected_signal_length
        
        # Default channel mapping for AidWear data
        # Based on the actual column names provided
        if channel_mapping is None:
            # For AidWear: use Hand for arms, Foot for legs
            # Indices based on the column order you provided
            self.channel_mapping = {
                'left_arm': [30, 31, 32],    # acc_Left_Hand_X/Y/Z (indices 30-32)
                'right_arm': [18, 19, 20],   # acc_Right_Hand_X/Y/Z (indices 18-20)
                'left_leg': [48, 49, 50],    # acc_Left_Foot_X/Y/Z (indices 48-50)
                'right_leg': [39, 40, 41]    # acc_Right_Foot_X/Y/Z (indices 39-41)
            }
        else:
            self.channel_mapping = channel_mapping
        
        # Create STFT transform
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            power=2  # Power spectrogram
        )
    
    def map_imu_channels(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Map IMU channels based on body part names.
        
        Args:
            imu_data: Raw IMU data [B, T, C] where C is all IMU channels
        
        Returns:
            Mapped IMU data [B, 4, 3, T] for 4 IMUs with 3 axes each
        """
        batch_size, time_steps, num_channels = imu_data.shape
        
        # Initialize output tensor
        mapped_data = torch.zeros(batch_size, 4, 3, time_steps, device=imu_data.device)
        
        # Map channels for each IMU
        imu_names = ['left_arm', 'right_arm', 'left_leg', 'right_leg']
        for i, imu_name in enumerate(imu_names):
            channels = self.channel_mapping[imu_name]
            if max(channels) < num_channels:
                mapped_data[:, i, :, :] = imu_data[:, :, channels].permute(0, 2, 1)
            else:
                print(f"Warning: Channel indices {channels} exceed available channels {num_channels}")
                # Use first available channels as fallback
                fallback_channels = [i*3, i*3+1, i*3+2]
                if max(fallback_channels) < num_channels:
                    mapped_data[:, i, :, :] = imu_data[:, :, fallback_channels].permute(0, 2, 1)
        
        return mapped_data
    
    def compute_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT spectrogram for a single signal.
        
        Args:
            signal: Input signal [B, T]
        
        Returns:
            Spectrogram [B, H, W]
        """
        batch_size, signal_length = signal.shape
        
        # Resample signal to expected length (250 samples for WEAR)
        if signal_length != self.expected_signal_length:
            # Use interpolation to resample
            signal = signal.unsqueeze(1)  # [B, 1, T]
            signal = F.interpolate(
                signal,
                size=self.expected_signal_length,
                mode='linear',
                align_corners=False
            )
            signal = signal.squeeze(1)  # [B, T]
        
        # Add channel dimension for STFT
        signal = signal.unsqueeze(1)  # [B, 1, T]
        
        # Compute STFT
        spec = self.stft_transform(signal)  # [B, 1, F, T]
        spec = spec.squeeze(1)  # [B, F, T]
        
        # Apply log scale if specified
        if self.use_log_scale:
            spec = torch.log(spec + 1e-6)
        
        # Resize to target dimensions
        if spec.shape[-2:] != (self.target_height, self.target_length):
            spec = F.interpolate(
                spec.unsqueeze(1),
                size=(self.target_height, self.target_length),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        return spec
    
    def normalize_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram using dataset statistics.
        
        Args:
            spec: Spectrogram [B, H, W]
        
        Returns:
            Normalized spectrogram
        """
        return (spec - self.norm_mean) / self.norm_std
    
    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Transform raw IMU data to spectrograms.
        
        Args:
            imu_data: Raw IMU data [B, T, C] or [B, C, T] or numpy array
                     where T is time steps and C is channels
        
        Returns:
            Spectrograms [B, 4, 3, H, W] for 4 IMUs with 3 axes each
        """
        # Convert numpy array to tensor if needed
        if isinstance(imu_data, np.ndarray):
            imu_data = torch.from_numpy(imu_data).float()
        
        # Handle different input formats
        if imu_data.dim() == 3:
            if imu_data.shape[1] > imu_data.shape[2]:
                # Assume [B, T, C] format
                pass
            else:
                # Assume [B, C, T] format, convert to [B, T, C]
                imu_data = imu_data.permute(0, 2, 1)
        elif imu_data.dim() == 2:
            # Single sample without batch dimension [T, C]
            imu_data = imu_data.unsqueeze(0)  # Add batch dimension
        
        batch_size = imu_data.shape[0]
        
        # Map IMU channels to 4 IMUs × 3 axes
        mapped_data = self.map_imu_channels(imu_data)  # [B, 4, 3, T]
        
        # Initialize output spectrograms
        spectrograms = torch.zeros(
            batch_size, 4, 3, self.target_height, self.target_length,
            device=imu_data.device
        )
        
        # Compute spectrogram for each IMU and axis
        for imu_idx in range(4):
            for axis_idx in range(3):
                signal = mapped_data[:, imu_idx, axis_idx, :]  # [B, T]
                spec = self.compute_spectrogram(signal)  # [B, H, W]
                spec = self.normalize_spectrogram(spec)
                spectrograms[:, imu_idx, axis_idx, :, :] = spec
        
        return spectrograms


class IMUChannelSelector(nn.Module):
    """
    Select specific IMU channels based on body part patterns.
    Useful for filtering accelerometer-only or specific body parts.
    """
    def __init__(
        self,
        channel_patterns: List[str],
        available_columns: List[str]
    ):
        super().__init__()
        
        self.channel_patterns = channel_patterns
        self.available_columns = available_columns
        
        # Find matching columns
        self.selected_indices = []
        self.selected_columns = []
        
        for pattern in channel_patterns:
            for i, col in enumerate(available_columns):
                if self._match_pattern(pattern, col):
                    self.selected_indices.append(i)
                    self.selected_columns.append(col)
        
        print(f"Selected {len(self.selected_indices)} IMU channels from {len(available_columns)} available")
    
    def _match_pattern(self, pattern: str, column: str) -> bool:
        """Check if column matches pattern (supports wildcards)."""
        import re
        pattern_regex = pattern.replace("*", ".*")
        return re.match(pattern_regex, column) is not None
    
    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Select specific channels from IMU data.
        
        Args:
            imu_data: Raw IMU data [B, T, C] or [B, C, T]
        
        Returns:
            Selected IMU data with same format as input
        """
        if imu_data.dim() == 3:
            if imu_data.shape[1] > imu_data.shape[2]:
                # [B, T, C] format
                return imu_data[:, :, self.selected_indices]
            else:
                # [B, C, T] format
                return imu_data[:, self.selected_indices, :]
        else:
            raise ValueError(f"Expected 3D tensor, got {imu_data.dim()}D")


class AdaptiveIMUChannelMapping:
    """
    Adaptive channel mapping based on available column names.
    Automatically maps IMU columns to left/right arm/leg based on naming conventions.
    """
    def __init__(self, column_names: List[str]):
        self.column_names = column_names
        self.channel_mapping = self._create_adaptive_mapping()
    
    def _create_adaptive_mapping(self) -> Dict[str, List[int]]:
        """Create adaptive mapping based on column names for AidWear dataset."""
        mapping = {
            'left_arm': [],
            'right_arm': [],
            'left_leg': [],
            'right_leg': []
        }
        
        # For AidWear dataset, based on the actual column names:
        # We'll use Hand sensors for arms and Foot sensors for legs
        for idx, col in enumerate(self.column_names):
            col_lower = col.lower()
            
            # Only process accelerometer columns
            if not col_lower.startswith('acc_'):
                continue
            
            # Map based on body part names
            if 'left_hand' in col_lower:
                if len(mapping['left_arm']) < 3:  # X, Y, Z
                    mapping['left_arm'].append(idx)
            elif 'right_hand' in col_lower:
                if len(mapping['right_arm']) < 3:
                    mapping['right_arm'].append(idx)
            elif 'left_foot' in col_lower:
                if len(mapping['left_leg']) < 3:
                    mapping['left_leg'].append(idx)
            elif 'right_foot' in col_lower:
                if len(mapping['right_leg']) < 3:
                    mapping['right_leg'].append(idx)
        
        # Fallback: If hands/feet not found, use forearm/lower_leg
        if not mapping['left_arm']:
            for idx, col in enumerate(self.column_names):
                if 'acc_left_forearm' in col.lower() and len(mapping['left_arm']) < 3:
                    mapping['left_arm'].append(idx)
        
        if not mapping['right_arm']:
            for idx, col in enumerate(self.column_names):
                if 'acc_right_forearm' in col.lower() and len(mapping['right_arm']) < 3:
                    mapping['right_arm'].append(idx)
        
        if not mapping['left_leg']:
            for idx, col in enumerate(self.column_names):
                if 'acc_left_lower_leg' in col.lower() and len(mapping['left_leg']) < 3:
                    mapping['left_leg'].append(idx)
        
        if not mapping['right_leg']:
            for idx, col in enumerate(self.column_names):
                if 'acc_right_lower_leg' in col.lower() and len(mapping['right_leg']) < 3:
                    mapping['right_leg'].append(idx)
        
        # Verify each body part has exactly 3 channels
        for body_part in mapping:
            if len(mapping[body_part]) != 3:
                print(f"Warning: {body_part} has {len(mapping[body_part])} channels instead of 3")
        
        return mapping
    
    def get_mapping(self) -> Dict[str, List[int]]:
        """Return the channel mapping."""
        return self.channel_mapping


# Composite transform for easy use
class EVI_MAE_IMUTransform(nn.Module):
    """
    Complete IMU transformation pipeline for EVI-MAE model.
    Resamples data to match EVI-MAE's expected input format.

    Supports multiple dataset configurations:
    - AidWear: Full body IMU with Hand/Foot sensors
    - RevalExo: Lower body IMU with Lower_Leg/Foot sensors
    - Direct: Uses first 12 columns directly (for pre-processed data)
    """
    def __init__(
        self,
        column_names: Optional[List[str]] = None,
        target_length: int = 320,
        target_height: int = 128,
        norm_mean: float = -54.33,
        norm_std: float = 26.04,
        n_fft: int = 256,  # Original EVI-MAE parameter
        win_length: int = 24,  # Original EVI-MAE parameter
        hop_length: int = 1,
        sampling_rate: int = 60,
        target_sampling_rate: int = 125,  # EVI-MAE expects 125Hz
        expected_signal_length: int = 250,  # EVI-MAE expects 250 samples
        use_aidwear_mapping: bool = False,  # Use predefined AidWear mapping
        use_revalexo_mapping: bool = False,  # Use predefined RevalExo mapping
        use_direct_mapping: bool = False,   # Use columns 0-11 directly
        dataset_type: Optional[str] = None  # Alternative: specify dataset type
    ):
        super().__init__()

        # Determine dataset type from parameters
        if dataset_type:
            use_aidwear_mapping = dataset_type.lower() == 'aidwear'
            use_revalexo_mapping = dataset_type.lower() == 'revalexo'
            use_direct_mapping = dataset_type.lower() == 'direct'

        # For AidWear dataset, use predefined mapping based on the column structure
        if use_aidwear_mapping:
            # Based on the column order provided:
            # acc columns are indices 0-50 (17 body parts × 3 axes)
            # We want: Left_Hand (30-32), Right_Hand (18-20), Left_Foot (48-50), Right_Foot (39-41)
            channel_mapping = {
                'left_arm': [30, 31, 32],    # acc_Left_Hand_X/Y/Z
                'right_arm': [18, 19, 20],   # acc_Right_Hand_X/Y/Z
                'left_leg': [48, 49, 50],    # acc_Left_Foot_X/Y/Z
                'right_leg': [39, 40, 41]    # acc_Right_Foot_X/Y/Z
            }
            print(f"Using AidWear channel mapping: {channel_mapping}")
        elif use_revalexo_mapping:
            # RevalExo dataset has lower body sensors only
            # Map to EVI-MAE's 4-limb structure using available sensors:
            # - left_arm -> Left_Lower_Leg acc (proxy for left side)
            # - right_arm -> Right_Lower_Leg acc (proxy for right side)
            # - left_leg -> Left_Foot acc
            # - right_leg -> Right_Foot acc
            #
            # Column order based on patterns in revalexo config:
            # acc_Pelvis (0-2), acc_Right_Upper_Leg (3-5), acc_Right_Lower_Leg (6-8),
            # acc_Right_Foot (9-11), acc_Left_Upper_Leg (12-14), acc_Left_Lower_Leg (15-17),
            # acc_Left_Foot (18-20)
            channel_mapping = {
                'left_arm': [15, 16, 17],    # acc_Left_Lower_Leg_X/Y/Z
                'right_arm': [6, 7, 8],      # acc_Right_Lower_Leg_X/Y/Z
                'left_leg': [18, 19, 20],    # acc_Left_Foot_X/Y/Z
                'right_leg': [9, 10, 11]     # acc_Right_Foot_X/Y/Z
            }
            print(f"Using RevalExo channel mapping: {channel_mapping}")
        elif use_direct_mapping:
            # Use first 12 columns directly (assumes pre-processed data in correct order)
            channel_mapping = {
                'left_arm': [0, 1, 2],
                'right_arm': [3, 4, 5],
                'left_leg': [6, 7, 8],
                'right_leg': [9, 10, 11]
            }
            print(f"Using direct channel mapping (columns 0-11): {channel_mapping}")
        elif column_names:
            # Create adaptive mapping if column names provided
            adapter = AdaptiveIMUChannelMapping(column_names)
            channel_mapping = adapter.get_mapping()
            print(f"Created adaptive channel mapping: {channel_mapping}")
        else:
            # Default mapping (direct 0-11)
            channel_mapping = {
                'left_arm': [0, 1, 2],
                'right_arm': [3, 4, 5],
                'left_leg': [6, 7, 8],
                'right_leg': [9, 10, 11]
            }
            print(f"Using default channel mapping (columns 0-11): {channel_mapping}")

        self.target_height = target_height
        self.target_length = target_length

        # Initialize spectrogram transform with EVI-MAE parameters
        self.spectrogram_transform = IMUToSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            target_length=target_length,
            target_height=target_height,
            norm_mean=norm_mean,
            norm_std=norm_std,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_sampling_rate,
            channel_mapping=channel_mapping,
            expected_signal_length=expected_signal_length
        )

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Transform raw IMU data to EVI-MAE compatible spectrograms.

        Args:
            imu_data: Raw IMU data [B, T, C] or [B, C, T]

        Returns:
            Spectrograms [B, 12, H, W]
        """
        spectrograms = self.spectrogram_transform(imu_data)

        # Reshape from [B, 4, 3, H, W] to [B, 12, H, W]
        batch_size = spectrograms.shape[0]
        if spectrograms.dim() == 5 and spectrograms.shape[1] == 4 and spectrograms.shape[2] == 3:
            # Flatten the 4 IMUs × 3 axes into 12 channels
            spectrograms = spectrograms.reshape(batch_size, 12, self.target_height, self.target_length)

        return spectrograms


# Helper function to get AidWear column names
def get_aidwear_column_names() -> List[str]:
    """Return the standard AidWear IMU column names in order."""
    body_parts = [
        'Pelvis', 'T8', 'Head',
        'Right_Shoulder', 'Right_Upper_Arm', 'Right_Forearm', 'Right_Hand',
        'Left_Shoulder', 'Left_Upper_Arm', 'Left_Forearm', 'Left_Hand',
        'Right_Upper_Leg', 'Right_Lower_Leg', 'Right_Foot',
        'Left_Upper_Leg', 'Left_Lower_Leg', 'Left_Foot'
    ]
    
    columns = []
    # Accelerometer columns
    for part in body_parts:
        for axis in ['X', 'Y', 'Z']:
            columns.append(f'acc_{part}_{axis}')
    
    # Gyroscope columns
    for part in body_parts:
        for axis in ['X', 'Y', 'Z']:
            columns.append(f'gyro_{part}_{axis}')
    
    # Magnetometer columns
    for part in body_parts:
        for axis in ['X', 'Y', 'Z']:
            columns.append(f'mag_{part}_{axis}')
    
    # Additional columns
    columns.extend(['timestamp', 'time_from_start_s'])
    
    return columns