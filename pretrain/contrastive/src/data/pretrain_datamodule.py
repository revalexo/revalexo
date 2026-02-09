# src/data/pretrain_datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import sys
import os
from typing import Optional, Dict, Any, List

# Add parent directory to path to import from existing codebase
from data.multimodaldataset import MultimodalSensorDataset

class PretrainDataModule(pl.LightningDataModule):
    """
    Data module for IMU2CLIP pretraining using existing MultimodalSensorDataset.
    """
    
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 4,
        modalities: List[str] = ["raw_imu", "video"],
        window_size: float = 2.0,
        video_model_frame_size: int = 16,
        train_subjects: Optional[List[str]] = None,
        val_subjects: Optional[List[str]] = None,
        train_transforms: Optional[Dict] = None,
        val_transforms: Optional[Dict] = None,
        sample_multiplier: int = 10,
        use_frames: bool = False,
        prediction_horizons: List[float] = [0]
    ):
        """
        Initialize data module.
        
        Args:
            dataset_config: Configuration dictionary for dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            modalities: List of modalities to load
            window_size: Window size in seconds
            video_model_frame_size: Number of frames for video models
            train_subjects: List of subjects for training
            val_subjects: List of subjects for validation
            train_transforms: Dictionary of transforms for training
            val_transforms: Dictionary of transforms for validation
            sample_multiplier: Number of samples per video clip
            use_frames: Whether to use extracted frames instead of video
            prediction_horizons: Prediction horizons (use [0] for pretraining)
        """
        super().__init__()
        
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modalities = modalities
        self.window_size = window_size
        self.video_model_frame_size = video_model_frame_size
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.train_transforms = train_transforms or {}
        self.val_transforms = val_transforms or {}
        self.sample_multiplier = sample_multiplier
        self.use_frames = use_frames
        self.prediction_horizons = prediction_horizons
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = MultimodalSensorDataset(
                dataset_config=self.dataset_config,
                modalities=self.modalities,
                window_size=self.window_size,
                video_model_frame_size=self.video_model_frame_size,
                split='train',
                transforms=self.train_transforms,
                subjects=self.train_subjects,
                sample_multiplier=self.sample_multiplier,
                use_frames=self.use_frames,
                exclude_background=False,
                prediction_horizons=self.prediction_horizons,
                base_seed=42
            )
            
            # Validation dataset
            self.val_dataset = MultimodalSensorDataset(
                dataset_config=self.dataset_config,
                modalities=self.modalities,
                window_size=self.window_size,
                video_model_frame_size=self.video_model_frame_size,
                split='val',
                transforms=self.val_transforms,
                subjects=self.val_subjects,
                sample_multiplier=1,  # No multiplication for validation
                use_frames=self.use_frames,
                exclude_background=False,
                prediction_horizons=self.prediction_horizons,
                eval_stride=0.5,  # Deterministic evaluation
                base_seed=42
            )
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True  # Important for contrastive learning
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )
    
    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Called before batch is transferred to device."""
        # Set epoch for reproducible random sampling
        if hasattr(self, 'trainer') and self.trainer is not None:
            if dataloader_idx == 0:  # Training dataloader
                self.train_dataset.set_epoch(self.trainer.current_epoch)
        return batch