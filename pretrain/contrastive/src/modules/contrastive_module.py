# src/modules/contrastive_module.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional
import torch.nn.functional as F


class IMU2CLIPModule(pl.LightningModule):
    """
    PyTorch Lightning module for IMU2CLIP contrastive learning.
    Adapts the IMU2CLIP approach for custom encoders.
    """
    
    def __init__(
        self,
        imu_encoder: nn.Module,
        visual_encoder: nn.Module,
        feature_dim: int = 512,
        temperature: float = 0.1,
        learn_temperature: bool = True,
        learning_rate: float = 0.01,
        weight_decay: float = 0.1,
        symmetric_loss: bool = True,
        freeze_visual: bool = False
    ):
        """
        Initialize IMU2CLIP module.
        
        Args:
            imu_encoder: IMU encoder model (e.g., DeepConvLSTM)
            visual_encoder: Visual encoder model (ResNet18 or X3D)
            feature_dim: Dimension of the joint embedding space
            temperature: Temperature parameter for InfoNCE loss
            learn_temperature: Whether to learn temperature as a parameter
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            symmetric_loss: Whether to use symmetric contrastive loss
            freeze_visual: Whether to freeze visual encoder (for CLIP-like pretraining)
        """
        super().__init__()
        
        self.imu_encoder = imu_encoder
        self.visual_encoder = visual_encoder
        self.feature_dim = feature_dim
        self.symmetric_loss = symmetric_loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Temperature parameter
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        
        # Projection heads to map to joint embedding space
        self.imu_projection = self._build_projection_head(
            self.imu_encoder.feature_dim, feature_dim
        )
        self.visual_projection = self._build_projection_head(
            self.visual_encoder.feature_dim, feature_dim
        )
        
        # Freeze visual encoder if specified
        if freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['imu_encoder', 'visual_encoder'])
    
    def _build_projection_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build a projection head for mapping to joint embedding space."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, imu_data: torch.Tensor, visual_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both encoders.
        
        Args:
            imu_data: IMU sensor data
            visual_data: Visual data (images or video)
            
        Returns:
            Dictionary containing normalized embeddings
        """
        # Get encoder features
        imu_features = self.imu_encoder.encode_features(imu_data)
        visual_features = self.visual_encoder.encode_features(visual_data)
        
        # Project to joint embedding space
        imu_embeddings = self.imu_projection(imu_features)
        visual_embeddings = self.visual_projection(visual_features)
        
        # L2 normalize embeddings
        imu_embeddings = F.normalize(imu_embeddings, p=2, dim=1)
        visual_embeddings = F.normalize(visual_embeddings, p=2, dim=1)
        
        return {
            'imu': imu_embeddings,
            'visual': visual_embeddings
        }
    
    def compute_loss(self, imu_embeds: torch.Tensor, visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            imu_embeds: Normalized IMU embeddings [B, D]
            visual_embeds: Normalized visual embeddings [B, D]
            
        Returns:
            Contrastive loss value
        """
        batch_size = imu_embeds.shape[0]
        device = imu_embeds.device
        
        # Compute similarity matrix
        similarity = torch.matmul(imu_embeds, visual_embeds.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=device)
        
        # Compute loss
        if self.symmetric_loss:
            # IMU -> Visual loss
            loss_i2v = F.cross_entropy(similarity, labels)
            # Visual -> IMU loss
            loss_v2i = F.cross_entropy(similarity.T, labels)
            loss = (loss_i2v + loss_v2i) / 2
        else:
            loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        imu_data, visual_data, _, _ = batch  # Unpack batch (ignoring labels and transition flags)
        
        # Forward pass
        embeddings = self(imu_data, visual_data)
        
        # Compute loss
        loss = self.compute_loss(embeddings['imu'], embeddings['visual'])
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        if isinstance(self.temperature, nn.Parameter):
            self.log('temperature', self.temperature.item(), prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        imu_data, visual_data, _, _ = batch
        
        # Forward pass
        embeddings = self(imu_data, visual_data)
        
        # Compute loss
        loss = self.compute_loss(embeddings['imu'], embeddings['visual'])
        
        # Compute retrieval metrics
        metrics = self._compute_retrieval_metrics(embeddings['imu'], embeddings['visual'])
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, prog_bar=False, sync_dist=True)
        
        return loss
    
    def _compute_retrieval_metrics(self, imu_embeds: torch.Tensor, visual_embeds: torch.Tensor) -> Dict[str, float]:
        """
        Compute retrieval metrics (R@1, R@5, R@10).
        
        Args:
            imu_embeds: IMU embeddings
            visual_embeds: Visual embeddings
            
        Returns:
            Dictionary of retrieval metrics
        """
        batch_size = imu_embeds.shape[0]
        
        # Compute similarity matrix
        similarity = torch.matmul(imu_embeds, visual_embeds.T)
        
        # Get rankings
        _, indices = similarity.topk(k=min(10, batch_size), dim=1)
        
        # Compute recall metrics
        correct_at_1 = (indices[:, 0] == torch.arange(batch_size, device=indices.device)).float().mean()
        correct_at_5 = (indices[:, :5] == torch.arange(batch_size, device=indices.device).unsqueeze(1)).any(dim=1).float().mean()
        correct_at_10 = (indices == torch.arange(batch_size, device=indices.device).unsqueeze(1)).any(dim=1).float().mean()
        
        return {
            'R@1': correct_at_1.item(),
            'R@5': correct_at_5.item(),
            'R@10': correct_at_10.item()
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Check if we should use Adagrad (as in the paper)
        use_adagrad = getattr(self, 'use_adagrad', False)
        
        if use_adagrad:
            # Use Adagrad as in the paper
            optimizer = torch.optim.Adagrad(
                self.parameters(),
                lr=self.learning_rate,
                eps=1e-8,
                weight_decay=self.weight_decay
            )
            
            # Learning rate scheduler for Adagrad
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            # Use AdamW (better for most cases)
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Cosine annealing for AdamW
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=50,  # Should be passed from config
                eta_min=1e-6
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, sync_dist=True)