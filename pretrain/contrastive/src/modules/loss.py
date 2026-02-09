# src/modules/loss.py
"""
Loss functions for IMU2CLIP contrastive learning.
Adapted from the original IMU2CLIP implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCE(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    Implementation based on IMU2CLIP paper.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
        symmetric: bool = True,
        learn_temperature: bool = False
    ):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling similarities
            reduction: How to reduce the loss ('mean' or 'sum')
            symmetric: Whether to use symmetric loss
            learn_temperature: Whether to learn temperature as a parameter
        """
        super().__init__()
        
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        
        self.reduction = reduction
        self.symmetric = symmetric
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings [B, D]
            positive_key: Positive key embeddings [B, D]
            negative_keys: Optional negative keys (if None, uses in-batch negatives)
            
        Returns:
            Loss value
        """
        return info_nce(
            query, 
            positive_key, 
            negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
            symmetric=self.symmetric
        )


def info_nce(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negative_keys: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    reduction: str = "mean",
    symmetric: bool = True
) -> torch.Tensor:
    """
    Compute InfoNCE loss function.
    
    Args:
        query: Query embeddings [B, D]
        positive_key: Positive embeddings [B, D]
        negative_keys: Negative embeddings (optional)
        temperature: Temperature for scaling
        reduction: Reduction method
        symmetric: Use symmetric loss
        
    Returns:
        Loss value
    """
    # Normalize embeddings
    query = F.normalize(query, p=2, dim=1)
    positive_key = F.normalize(positive_key, p=2, dim=1)
    
    batch_size = query.shape[0]
    device = query.device
    
    if negative_keys is None:
        # Use in-batch negatives
        # Compute all pairwise similarities
        logits = torch.matmul(query, positive_key.T) / temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=device)
        
        if symmetric:
            # Query -> Key loss
            loss_qk = F.cross_entropy(logits, labels, reduction=reduction)
            # Key -> Query loss  
            loss_kq = F.cross_entropy(logits.T, labels, reduction=reduction)
            loss = (loss_qk + loss_kq) / 2
        else:
            loss = F.cross_entropy(logits, labels, reduction=reduction)
    else:
        # Use provided negative keys
        negative_keys = F.normalize(negative_keys, p=2, dim=1)
        
        # Positive logits
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True) / temperature
        
        # Negative logits
        negative_logits = torch.matmul(query, negative_keys.T) / temperature
        
        # Concatenate logits
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        
        # Labels (positive is at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, labels, reduction=reduction)
    
    return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Alternative formulation of contrastive loss.
    """
    
    def __init__(self, temperature: float = 0.1, learn_temperature: bool = False):
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z_i: First view embeddings [B, D]
            z_j: Second view embeddings [B, D]
            
        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create mask for positive pairs
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=device)
        mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool)
        mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool)
        
        # Extract positive and negative pairs
        positives = similarity_matrix[mask].view(batch_size * 2, 1)
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
        
        # Compute logits
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        
        # Labels (positive pair is at index 0)
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss