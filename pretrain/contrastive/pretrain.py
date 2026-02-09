#!/usr/bin/env python3
# pretrain.py

"""
Main script for contrastive pretraining on RevalExo dataset.
Trains encoders using contrastive learning between IMU and visual modalities.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src.modules.contrastive_module import IMU2CLIPModule
from src.data.pretrain_datamodule import PretrainDataModule

# Import models from existing codebase
from src.models.deepconvlstm import DeepConvLSTM
from src.models.resnet_models import ResNet18_Image
from src.models.pytorchvideo_models import X3D_Video, MViT_Video

# Import transforms from existing codebase
import src.transforms as transform_utils


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_transforms(transform_config: dict):
    """Build transform pipeline from configuration."""
    transforms = {}
    
    for modality, transform_list in transform_config.items():
        if transform_list:
            transform_instances = []
            for transform_spec in transform_list:
                transform_class = getattr(transform_utils, transform_spec['name'])
                transform_params = transform_spec.get('params', {})
                
                # Convert lists to tuples for size/crop_size parameters
                if 'size' in transform_params and isinstance(transform_params['size'], list):
                    transform_params['size'] = tuple(transform_params['size'])
                if 'crop_size' in transform_params and isinstance(transform_params['crop_size'], list):
                    transform_params['crop_size'] = tuple(transform_params['crop_size'])
                
                transform_instances.append(transform_class(**transform_params))
            
            transforms[modality] = transform_utils.Compose(transform_instances)
        else:
            transforms[modality] = None
    
    return transforms


def build_imu_encoder(config: dict) -> DeepConvLSTM:
    """Build IMU encoder from configuration."""
    model_params = config['models']['imu_encoder']['params'].copy()
    
    # Create DeepConvLSTM model
    encoder = DeepConvLSTM(
        channels=model_params['channels'],
        num_classes=11,  # Dummy value, not used in pretraining
        window_size=model_params['window_size'],
        conv_kernels=model_params['conv_kernels'],
        conv_kernel_size=model_params['conv_kernel_size'],
        lstm_units=model_params['lstm_units'],
        lstm_layers=model_params['lstm_layers'],
        dropout=model_params['dropout'],
        feature_dim=model_params.get('feature_dim', 512),
        prediction_horizons=[0]  # Single horizon for pretraining
    )
    
    return encoder


def build_visual_encoder(config: dict):
    """Build visual encoder from configuration."""
    encoder_name = config['models']['visual_encoder']['name']
    model_params = config['models']['visual_encoder']['params'].copy()
    
    if encoder_name == 'ResNet18_Image':
        encoder = ResNet18_Image(
            num_classes=11,  # Dummy value, not used in pretraining
            pretrained=model_params.get('pretrained', True),
            feature_dim=model_params.get('feature_dim', 512),
            freeze_backbone=model_params.get('freeze_backbone', False),
            freeze_early_layers=model_params.get('freeze_early_layers', False),
            prediction_horizons=[0]  # Single horizon for pretraining
        )
    elif encoder_name == 'X3D_Video':
        encoder = X3D_Video(
            num_classes=11,  # Dummy value, not used in pretraining
            pretrained=model_params.get('pretrained', True),
            model_size=model_params.get('model_size', 'xs'),
            feature_dim=model_params.get('feature_dim', 512),
            freeze_backbone=model_params.get('freeze_backbone', False),
            prediction_horizons=[0]  # Single horizon for pretraining
        )
    elif encoder_name == 'MViT_Video':
        encoder = MViT_Video(
            num_classes=11,  # Dummy value, not used in pretraining
            pretrained=model_params.get('pretrained', True),
            variant=model_params.get('variant', '16x4'),
            feature_dim=model_params.get('feature_dim', 512),
            freeze_backbone=model_params.get('freeze_backbone', False),
            prediction_horizons=[0]  # Single horizon for pretraining
        )
    elif encoder_name == 'CLIPVisualEncoder':
        from src.models.clip_encoder import CLIPVisualEncoder
        encoder = CLIPVisualEncoder(
            clip_model_name=model_params.get('clip_model_name', 'ViT-B/32'),
            feature_dim=model_params.get('feature_dim', 512),
            freeze=model_params.get('freeze', True),
            prediction_horizons=[0]
        )
    else:
        raise ValueError(f"Unknown visual encoder: {encoder_name}")
    
    return encoder


def main(args):
    """Main pretraining function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up paths
    checkpoint_dir = Path(config['training']['save_dir']) / config['experiment_name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['logging']['save_dir']) / config['experiment_name']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to checkpoint directory
    config_save_path = checkpoint_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # Build transforms
    train_transforms = build_transforms(config['transforms']['train'])
    val_transforms = build_transforms(config['transforms']['val'])
    
    # Determine modalities based on visual encoder type
    visual_encoder_name = config['models']['visual_encoder']['name']
    if 'Image' in visual_encoder_name or 'CLIP' in visual_encoder_name:
        modalities = ['raw_imu', 'image']
    else:
        modalities = ['raw_imu', 'video']
    
    # Create data module
    data_module = PretrainDataModule(
        dataset_config=config['dataset'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        modalities=modalities,
        window_size=config['dataset']['default_window_size'],
        video_model_frame_size=config['training'].get('video_model_frame_size', 16),
        train_subjects=config['dataset']['train_subjects'],
        val_subjects=config['dataset']['val_subjects'],
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        sample_multiplier=config['training']['sample_multiplier'],
        use_frames=False,
        prediction_horizons=[0]  # Single horizon for pretraining
    )
    
    # Build encoders
    print("Building encoders...")
    imu_encoder = build_imu_encoder(config)
    visual_encoder = build_visual_encoder(config)
    
    # Create contrastive learning module
    print("Creating IMU2CLIP module...")
    model = IMU2CLIPModule(
        imu_encoder=imu_encoder,
        visual_encoder=visual_encoder,
        feature_dim=config['models']['imu_encoder']['params'].get('feature_dim', 512),
        temperature=config['training']['temperature'],
        learn_temperature=config['training']['learn_temperature'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        symmetric_loss=config['training']['symmetric_loss'],
        freeze_visual=config['training'].get('freeze_visual', False)
    )
    
    # Set optimizer type if specified
    if config['training'].get('use_adagrad', False):
        model.use_adagrad = True
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor=config['training']['monitor'],
        mode=config['training']['mode'],
        save_top_k=config['training']['save_top_k'],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Custom callback to save encoders separately when checkpoint is saved
    class SaveEncodersCallback(pl.Callback):
        def __init__(self, checkpoint_dir):
            self.checkpoint_dir = checkpoint_dir
            self.best_val_loss = float('inf')
            
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            # This is called whenever a checkpoint is saved
            epoch = trainer.current_epoch
            val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Save best encoders
                imu_path = self.checkpoint_dir / f'imu_encoder_best.pt'
                visual_path = self.checkpoint_dir / f'visual_encoder_best.pt'
                torch.save(pl_module.imu_encoder.state_dict(), imu_path)
                torch.save(pl_module.visual_encoder.state_dict(), visual_path)
                print(f"Saved best encoders (val_loss={val_loss:.4f})")
            
            # Always save current epoch encoders
            imu_path = self.checkpoint_dir / f'imu_encoder_epoch{epoch:02d}.pt'
            visual_path = self.checkpoint_dir / f'visual_encoder_epoch{epoch:02d}.pt'
            torch.save(pl_module.imu_encoder.state_dict(), imu_path)
            torch.save(pl_module.visual_encoder.state_dict(), visual_path)
            print(f"Saved epoch {epoch} encoders separately")
    
    callbacks.append(SaveEncodersCallback(checkpoint_dir))
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Early stopping (optional)
    if args.early_stop:
        early_stop = EarlyStopping(
            monitor=config['training']['monitor'],
            mode=config['training']['mode'],
            patience=10,
            verbose=True
        )
        callbacks.append(early_stop)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=config['logging']['name'],
        version=datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else 1,
        strategy='ddp' if args.gpus > 1 else 'auto',
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        precision=16 if args.mixed_precision else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
    )
    
    # Train model
    print(f"Starting training for {config['experiment_name']}...")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    trainer.fit(model, data_module, ckpt_path=args.resume)
    
    # Save final encoders separately
    print("Saving trained encoders...")
    
    # Save IMU encoder
    imu_encoder_path = checkpoint_dir / 'imu_encoder_final.pt'
    torch.save(model.imu_encoder.state_dict(), imu_encoder_path)
    print(f"IMU encoder saved to: {imu_encoder_path}")
    
    # Save visual encoder
    visual_encoder_path = checkpoint_dir / 'visual_encoder_final.pt'
    torch.save(model.visual_encoder.state_dict(), visual_encoder_path)
    print(f"Visual encoder saved to: {visual_encoder_path}")
    
    # Save projection heads
    projections_path = checkpoint_dir / 'projections_final.pt'
    torch.save({
        'imu_projection': model.imu_projection.state_dict(),
        'visual_projection': model.visual_projection.state_dict()
    }, projections_path)
    print(f"Projection heads saved to: {projections_path}")
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMU2CLIP Pretraining')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to use'
    )
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Use mixed precision training'
    )
    parser.add_argument(
        '--early-stop',
        action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (e.g., checkpoints/experiment/last.ckpt)'
    )

    args = parser.parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    main(args)