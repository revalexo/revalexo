# losses/distillation.py
"""
Knowledge Distillation loss functions.

Supports:
- Vanilla KD (Hinton et al., 2015): Logit-based soft target distillation
- FitNets (Romero et al., 2015): Feature-based distillation with adapter
- CRD (Tian et al., 2020): Contrastive Representation Distillation
- NKD (Yang et al., ICCV 2023): Normalized Knowledge Distillation (target/non-target decomposition)
- Multi-horizon models: Distillation across multiple prediction horizons

# Some taken and adapted from https://github.com/yzd-v/cls_KD

Usage in config:
    distillation:
      enabled: true
      teacher_checkpoint: "path/to/teacher/best_model.pt"
      teacher_config: "path/to/teacher/config.yaml"
      method: "vanilla"  # or "fitnets", "crd", "nkd"
      temperature: 4.0
      alpha: 0.5  # weight for task loss (1-alpha for KD loss)

      # FitNets specific
      fitnets:
        beta: 100.0  # feature matching weight

      # CRD specific
      crd:
        embed_dim: 128
        temperature: 0.07
        use_memory_bank: false
        memory_size: 16384

      # NKD specific
      nkd:
        temperature: 1.0  # for non-target distillation
        gamma: 1.5  # weight for non-target loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union
import math


class VanillaKDLoss(nn.Module):
    """
    Vanilla Knowledge Distillation loss (Hinton et al., 2015).

    Computes KL divergence between softened teacher and student logits.

    Loss = KL_div(log_softmax(student/T), softmax(teacher/T)) * T^2

    Args:
        temperature: Temperature for softening logits (default: 4.0)
            Higher temperature = softer probability distribution
        reduction: Reduction method ('batchmean', 'sum', 'none')
    """

    def __init__(self, temperature: float = 4.0, reduction: str = 'batchmean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KD loss between student and teacher logits.

        Args:
            student_logits: Student model outputs [batch_size, num_classes]
            teacher_logits: Teacher model outputs [batch_size, num_classes]

        Returns:
            KD loss scalar
        """
        # Soften logits with temperature
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence loss
        loss = F.kl_div(soft_student, soft_teacher, reduction=self.reduction)

        # Scale by temperature squared (as per Hinton et al.)
        loss = loss * (self.temperature ** 2)

        return loss


class MultiHorizonKDLoss(nn.Module):
    """
    Knowledge Distillation loss for multi-horizon prediction models.

    Applies KD loss to each prediction horizon and combines them.

    Args:
        temperature: Temperature for softening logits
        horizon_weights: Optional weights for each horizon's KD loss
            If None, uses equal weights
    """

    def __init__(
        self,
        temperature: float = 4.0,
        horizon_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.temperature = temperature
        self.kd_loss = VanillaKDLoss(temperature=temperature)
        self.horizon_weights = horizon_weights

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compute KD loss for multi-horizon outputs.

        Args:
            student_outputs: List of student outputs for each horizon
            teacher_outputs: List of teacher outputs for each horizon

        Returns:
            Dictionary containing total loss and per-horizon losses
        """
        if len(student_outputs) != len(teacher_outputs):
            raise ValueError(
                f"Number of student horizons ({len(student_outputs)}) must match "
                f"teacher horizons ({len(teacher_outputs)})"
            )

        num_horizons = len(student_outputs)

        # Set horizon weights
        if self.horizon_weights is None:
            weights = [1.0 / num_horizons] * num_horizons
        else:
            weights = self.horizon_weights

        # Compute KD loss for each horizon
        individual_losses = []
        for student_out, teacher_out in zip(student_outputs, teacher_outputs):
            loss = self.kd_loss(student_out, teacher_out)
            individual_losses.append(loss)

        # Combine losses with weights
        total_loss = sum(w * loss for w, loss in zip(weights, individual_losses))

        return {
            'total_loss': total_loss,
            'individual_losses': individual_losses,
            'horizon_losses': {
                f'horizon_{i}': loss.item()
                for i, loss in enumerate(individual_losses)
            }
        }


class DistillationLoss(nn.Module):
    """
    Combined distillation loss: task loss + KD loss.

    Total Loss = alpha * task_loss + (1 - alpha) * kd_loss

    Where:
    - task_loss: Standard cross-entropy with ground truth labels
    - kd_loss: KL divergence with teacher's soft predictions
    - alpha: Balancing weight (0.5 = equal weight for both)

    Args:
        task_loss_fn: Base task loss function (e.g., CrossEntropyLoss)
        temperature: Temperature for KD softening (default: 4.0)
        alpha: Weight for task loss, (1-alpha) for KD loss (default: 0.5)
        prediction_horizons: List of prediction horizons (for multi-horizon models)
        horizon_loss_weights: Optional weights for each horizon
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        prediction_horizons: Optional[List[float]] = None,
        horizon_loss_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.temperature = temperature
        self.alpha = alpha
        self.prediction_horizons = prediction_horizons or [0]
        self.num_horizons = len(self.prediction_horizons)

        # KD loss component
        self.kd_loss = MultiHorizonKDLoss(
            temperature=temperature,
            horizon_weights=horizon_loss_weights
        )

        # Set horizon weights for task loss
        if horizon_loss_weights is None:
            self.horizon_weights = torch.ones(self.num_horizons) / self.num_horizons
        else:
            self.horizon_weights = torch.tensor(horizon_loss_weights, dtype=torch.float32)

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compute combined distillation loss.

        Args:
            student_outputs: List of student model outputs for each horizon
            teacher_outputs: List of teacher model outputs for each horizon (detached)
            targets: List of ground truth labels for each horizon

        Returns:
            Dictionary containing:
                - total_loss: Combined loss for backprop
                - task_loss: Task loss component
                - kd_loss: KD loss component
                - task_losses_per_horizon: Per-horizon task losses
                - kd_losses_per_horizon: Per-horizon KD losses
        """
        device = student_outputs[0].device
        horizon_weights = self.horizon_weights.to(device)

        # Compute task loss for each horizon
        task_losses = []
        for student_out, target in zip(student_outputs, targets):
            loss = self.task_loss_fn(student_out, target)
            task_losses.append(loss)

        total_task_loss = sum(w * loss for w, loss in zip(horizon_weights, task_losses))

        # Compute KD loss
        kd_result = self.kd_loss(student_outputs, teacher_outputs)
        total_kd_loss = kd_result['total_loss']

        # Combined loss
        total_loss = self.alpha * total_task_loss + (1 - self.alpha) * total_kd_loss

        return {
            'total_loss': total_loss,
            'task_loss': total_task_loss,
            'kd_loss': total_kd_loss,
            'task_losses_per_horizon': {
                f'horizon_{i}': loss.item()
                for i, loss in enumerate(task_losses)
            },
            'kd_losses_per_horizon': kd_result['horizon_losses']
        }


# =============================================================================
# FitNets: Feature-based Knowledge Distillation (Romero et al., 2015)
# =============================================================================

class FitNetsAdapter(nn.Module):
    """
    Adapter network to project student features to teacher feature space.

    Used when student and teacher have different feature dimensions.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = (student_dim + teacher_dim) // 2

        self.adapter = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, teacher_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class FitNetsLoss(nn.Module):
    """
    FitNets: Hints for Thin Deep Nets (Romero et al., 2015)

    Feature-based distillation that matches intermediate representations
    between student and teacher using an adapter layer.

    Loss = MSE(adapter(student_features), teacher_features)

    Args:
        student_dim: Dimension of student features
        teacher_dim: Dimension of teacher features
        beta: Weight for the feature matching loss (default: 100.0)
        normalize_features: Whether to L2-normalize features before matching
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        beta: float = 100.0,
        normalize_features: bool = True
    ):
        super().__init__()
        self.beta = beta
        self.normalize_features = normalize_features

        # Create adapter if dimensions don't match
        if student_dim != teacher_dim:
            self.adapter = FitNetsAdapter(student_dim, teacher_dim)
        else:
            self.adapter = nn.Identity()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute FitNets feature matching loss.

        Args:
            student_features: Student model features [batch_size, student_dim]
            teacher_features: Teacher model features [batch_size, teacher_dim]

        Returns:
            Feature matching loss (MSE)
        """
        # Project student features to teacher space
        student_proj = self.adapter(student_features)

        # Optionally normalize features
        if self.normalize_features:
            student_proj = F.normalize(student_proj, p=2, dim=1)
            teacher_features = F.normalize(teacher_features, p=2, dim=1)

        # MSE loss between projected student and teacher features
        loss = F.mse_loss(student_proj, teacher_features)

        return self.beta * loss


# =============================================================================
# CRD: Contrastive Representation Distillation (Tian et al., 2020)

# Simplified version
# Differences from the original RepDistiller implementation:
# - Single-direction contrastive loss (original uses symmetric student-anchor + teacher-anchor)
# - In-batch negatives or simple FIFO memory bank (original uses momentum-updated memory sized to full dataset with AliasMethod sampling)
# =============================================================================

class CRDProjectionHead(nn.Module):
    """
    Projection head for CRD.

    Projects features to a lower-dimensional embedding space for contrastive learning.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), p=2, dim=1)


class CRDLoss(nn.Module):
    """
    Contrastive Representation Distillation (Tian et al., 2020)

    Uses contrastive learning to transfer knowledge from teacher to student.
    Positive pairs: (student_i, teacher_i) from same sample
    Negative pairs: (student_i, teacher_j) from different samples

    Args:
        student_dim: Dimension of student features
        teacher_dim: Dimension of teacher features
        embed_dim: Dimension of projection space (default: 128)
        temperature: Temperature for contrastive loss (default: 0.07)
        use_memory_bank: Whether to use memory bank for negatives (default: False)
        memory_size: Size of memory bank if used (default: 16384)
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        embed_dim: int = 128,
        temperature: float = 0.07,
        use_memory_bank: bool = False,
        memory_size: int = 16384
    ):
        super().__init__()
        self.temperature = temperature
        self.use_memory_bank = use_memory_bank
        self.embed_dim = embed_dim

        # Projection heads
        self.student_proj = CRDProjectionHead(student_dim, embed_dim)
        self.teacher_proj = CRDProjectionHead(teacher_dim, embed_dim)

        # Memory bank for teacher embeddings (optional)
        if use_memory_bank:
            self.register_buffer(
                'memory_bank',
                torch.randn(memory_size, embed_dim)
            )
            self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
            self.memory_size = memory_size
            # Normalize memory bank
            self.memory_bank = F.normalize(self.memory_bank, p=2, dim=1)

    @torch.no_grad()
    def _update_memory_bank(self, teacher_embeddings: torch.Tensor):
        """Update memory bank with new teacher embeddings."""
        if not self.use_memory_bank:
            return

        batch_size = teacher_embeddings.size(0)
        ptr = int(self.memory_ptr)

        # Handle wrap-around
        if ptr + batch_size > self.memory_size:
            # Fill remaining space
            remaining = self.memory_size - ptr
            self.memory_bank[ptr:] = teacher_embeddings[:remaining]
            # Wrap to beginning
            overflow = batch_size - remaining
            self.memory_bank[:overflow] = teacher_embeddings[remaining:]
            ptr = overflow
        else:
            self.memory_bank[ptr:ptr + batch_size] = teacher_embeddings
            ptr = (ptr + batch_size) % self.memory_size

        self.memory_ptr[0] = ptr

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CRD contrastive loss.

        Args:
            student_features: [batch_size, student_dim]
            teacher_features: [batch_size, teacher_dim]

        Returns:
            Dictionary with total loss and additional info
        """
        batch_size = student_features.size(0)

        # Project to embedding space
        z_student = self.student_proj(student_features)  # [B, embed_dim]
        z_teacher = self.teacher_proj(teacher_features)  # [B, embed_dim]

        if self.use_memory_bank:
            # Use memory bank for negatives
            # Positive: similarity with corresponding teacher
            pos_sim = (z_student * z_teacher).sum(dim=1, keepdim=True)  # [B, 1]

            # Negative: similarity with memory bank (clone to avoid in-place modification issue)
            neg_sim = torch.mm(z_student, self.memory_bank.clone().t())  # [B, memory_size]

            # Combine: [B, 1 + memory_size]
            logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature

            # Labels: positive is at index 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

            # Update memory bank
            self._update_memory_bank(z_teacher.detach())
        else:
            # In-batch negatives (InfoNCE)
            # Similarity matrix: [B, B]
            similarity = torch.mm(z_student, z_teacher.t()) / self.temperature

            # Labels: diagonal elements are positives
            labels = torch.arange(batch_size, device=similarity.device)
            logits = similarity

        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy for monitoring
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

        return {
            'total_loss': loss,
            'contrastive_accuracy': accuracy
        }


# =============================================================================
# NKD: Normalized Knowledge Distillation (Yang et al., ICCV 2023)
# =============================================================================

class NKDLoss(nn.Module):
    """
    Normalized Knowledge Distillation (NKD) Loss.

    From "From Knowledge Distillation to Self-Knowledge Distillation:
    A Unified Approach with Normalized Loss and Customized Soft Labels" (ICCV 2023)

    Key insight: Decomposes KD into target and non-target components.
    - Target loss: Weighted cross-entropy on target class (no temperature)
    - Non-target loss: KL divergence on normalized non-target logits (with temperature)

    The "normalization" refers to removing the target logit and re-applying softmax
    to the remaining logits, ensuring proper probability distribution over non-targets.

    Args:
        temperature: Temperature for non-target distillation (default: 1.0)
        gamma: Weight for non-target loss (default: 1.5)
    """

    def __init__(self, temperature: float = 1.0, gamma: float = 1.5):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NKD loss.

        Args:
            student_logits: Student model raw logits [batch_size, num_classes]
            teacher_logits: Teacher model raw logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size] or [batch_size, 1]

        Returns:
            NKD loss scalar
        """
        # Handle label shape
        if len(labels.size()) > 1:
            label = labels.view(labels.size(0), 1)
        else:
            label = labels.view(-1, 1)

        B, C = student_logits.shape

        # ===== Target class loss (no temperature) =====
        # log_softmax for student, softmax for teacher
        s_i = F.log_softmax(student_logits, dim=1)
        t_i = F.softmax(teacher_logits, dim=1)

        # Gather target class probabilities
        s_t = torch.gather(s_i, 1, label)  # log(p_student) for target class
        t_t = torch.gather(t_i, 1, label).detach()  # p_teacher for target class

        # Weighted cross-entropy on target: -t_t * log(s_t)
        loss_target = -(t_t * s_t).mean()

        # ===== Non-target class loss (with temperature) =====
        # Create mask to remove target class from logits
        mask = torch.ones_like(student_logits).scatter_(1, label, 0).bool()

        # Extract non-target logits [B, C-1]
        logit_s_nt = student_logits[mask].reshape(B, -1)
        logit_t_nt = teacher_logits[mask].reshape(B, -1)

        # Apply softmax to non-target logits (this is the "normalization")
        S_i = F.log_softmax(logit_s_nt / self.temperature, dim=1)
        T_i = F.softmax(logit_t_nt / self.temperature, dim=1)

        # Cross-entropy on non-targets: -sum(T * log(S))
        loss_nontarget = -(T_i * S_i).sum(dim=1).mean()
        loss_nontarget = self.gamma * (self.temperature ** 2) * loss_nontarget

        return loss_target + loss_nontarget


class MultiHorizonNKDLoss(nn.Module):
    """
    NKD loss for multi-horizon prediction models.

    Args:
        temperature: Temperature for non-target distillation
        gamma: Weight for non-target loss
        horizon_weights: Optional weights for each horizon's loss
    """

    def __init__(
        self,
        temperature: float = 1.0,
        gamma: float = 1.5,
        horizon_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.nkd_loss = NKDLoss(temperature=temperature, gamma=gamma)
        self.horizon_weights = horizon_weights

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compute NKD loss for multi-horizon outputs.

        Args:
            student_outputs: List of student logits for each horizon
            teacher_outputs: List of teacher logits for each horizon
            targets: List of labels for each horizon

        Returns:
            Dictionary containing total loss and per-horizon losses
        """
        num_horizons = len(student_outputs)

        if self.horizon_weights is None:
            weights = [1.0 / num_horizons] * num_horizons
        else:
            weights = self.horizon_weights

        individual_losses = []
        for s_out, t_out, target in zip(student_outputs, teacher_outputs, targets):
            loss = self.nkd_loss(s_out, t_out, target)
            individual_losses.append(loss)

        total_loss = sum(w * loss for w, loss in zip(weights, individual_losses))

        return {
            'total_loss': total_loss,
            'individual_losses': individual_losses,
            'horizon_losses': {
                f'horizon_{i}': loss.item()
                for i, loss in enumerate(individual_losses)
            }
        }


class NKDDistillationLoss(nn.Module):
    """
    Combined NKD distillation loss: task loss + NKD loss.

    Total Loss = alpha * task_loss + (1 - alpha) * nkd_loss

    Args:
        task_loss_fn: Base task loss function (e.g., CrossEntropyLoss)
        temperature: Temperature for NKD non-target distillation (default: 1.0)
        gamma: Weight for NKD non-target loss (default: 1.5)
        alpha: Weight for task loss, (1-alpha) for NKD loss (default: 0.5)
        prediction_horizons: List of prediction horizons
        horizon_loss_weights: Optional weights for each horizon
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        temperature: float = 1.0,
        gamma: float = 1.5,
        alpha: float = 0.5,
        prediction_horizons: Optional[List[float]] = None,
        horizon_loss_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.alpha = alpha
        self.prediction_horizons = prediction_horizons or [0]
        self.num_horizons = len(self.prediction_horizons)

        # NKD loss component
        self.nkd_loss = MultiHorizonNKDLoss(
            temperature=temperature,
            gamma=gamma,
            horizon_weights=horizon_loss_weights
        )

        # Set horizon weights for task loss
        if horizon_loss_weights is None:
            self.horizon_weights = torch.ones(self.num_horizons) / self.num_horizons
        else:
            self.horizon_weights = torch.tensor(horizon_loss_weights, dtype=torch.float32)

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compute combined NKD distillation loss.

        Args:
            student_outputs: List of student model outputs for each horizon
            teacher_outputs: List of teacher model outputs for each horizon
            targets: List of ground truth labels for each horizon

        Returns:
            Dictionary containing loss components
        """
        device = student_outputs[0].device
        horizon_weights = self.horizon_weights.to(device)

        # Compute task loss for each horizon
        task_losses = []
        for student_out, target in zip(student_outputs, targets):
            loss = self.task_loss_fn(student_out, target)
            task_losses.append(loss)

        total_task_loss = sum(w * loss for w, loss in zip(horizon_weights, task_losses))

        # Compute NKD loss
        nkd_result = self.nkd_loss(student_outputs, teacher_outputs, targets)
        total_nkd_loss = nkd_result['total_loss']

        # Combined loss
        total_loss = self.alpha * total_task_loss + (1 - self.alpha) * total_nkd_loss

        return {
            'total_loss': total_loss,
            'task_loss': total_task_loss,
            'kd_loss': total_nkd_loss,  # Use 'kd_loss' for compatibility
            'nkd_loss': total_nkd_loss,
            'task_losses_per_horizon': {
                f'horizon_{i}': loss.item()
                for i, loss in enumerate(task_losses)
            },
            'kd_losses_per_horizon': nkd_result['horizon_losses']
        }


# =============================================================================
# Combined Feature-based Distillation Loss
# =============================================================================

class FeatureDistillationLoss(nn.Module):
    """
    Combined feature-based distillation loss.

    Combines task loss with feature-based KD methods (FitNets, or CRD).
    Optionally can also include vanilla KD loss.

    Total Loss = alpha * task_loss + (1 - alpha) * feature_kd_loss + gamma * vanilla_kd_loss

    Args:
        task_loss_fn: Base task loss function
        method: KD method ("fitnets", or "crd")
        student_dim: Student feature dimension
        teacher_dim: Teacher feature dimension
        alpha: Weight for task loss (default: 0.5)
        include_vanilla_kd: Whether to also include vanilla KD (default: False)
        vanilla_kd_weight: Weight for vanilla KD if included (default: 0.5)
        temperature: Temperature for vanilla KD (default: 4.0)
        prediction_horizons: List of prediction horizons
        **method_kwargs: Additional arguments for the specific KD method
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        method: str,
        student_dim: int,
        teacher_dim: int,
        alpha: float = 0.5,
        include_vanilla_kd: bool = False,
        vanilla_kd_weight: float = 0.5,
        temperature: float = 4.0,
        prediction_horizons: Optional[List[float]] = None,
        horizon_loss_weights: Optional[List[float]] = None,
        **method_kwargs
    ):
        super().__init__()

        self.task_loss_fn = task_loss_fn
        self.method = method.lower()
        self.alpha = alpha
        self.include_vanilla_kd = include_vanilla_kd
        self.vanilla_kd_weight = vanilla_kd_weight
        self.prediction_horizons = prediction_horizons or [0]
        self.num_horizons = len(self.prediction_horizons)

        # Set horizon weights for task loss
        if horizon_loss_weights is None:
            self.horizon_weights = torch.ones(self.num_horizons) / self.num_horizons
        else:
            self.horizon_weights = torch.tensor(horizon_loss_weights, dtype=torch.float32)

        # Create feature-based KD loss
        if self.method == "fitnets":
            beta = method_kwargs.get('beta', 100.0)
            normalize = method_kwargs.get('normalize_features', True)
            self.feature_kd_loss = FitNetsLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                beta=beta,
                normalize_features=normalize
            )
        elif self.method == "crd":
            embed_dim = method_kwargs.get('embed_dim', 128)
            crd_temperature = method_kwargs.get('crd_temperature', 0.07)
            use_memory_bank = method_kwargs.get('use_memory_bank', False)
            memory_size = method_kwargs.get('memory_size', 16384)
            self.feature_kd_loss = CRDLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                embed_dim=embed_dim,
                temperature=crd_temperature,
                use_memory_bank=use_memory_bank,
                memory_size=memory_size
            )
        else:
            raise ValueError(f"Unknown feature-based KD method: {method}")

        # Optional vanilla KD
        if include_vanilla_kd:
            self.vanilla_kd_loss = MultiHorizonKDLoss(
                temperature=temperature,
                horizon_weights=horizon_loss_weights
            )

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        targets: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compute combined feature-based distillation loss.

        Args:
            student_outputs: List of student logits for each horizon
            teacher_outputs: List of teacher logits for each horizon
            student_features: Student feature tensor [batch_size, student_dim]
            teacher_features: Teacher feature tensor [batch_size, teacher_dim]
            targets: List of ground truth labels for each horizon

        Returns:
            Dictionary containing loss components
        """
        device = student_outputs[0].device
        horizon_weights = self.horizon_weights.to(device)

        # Compute task loss for each horizon
        task_losses = []
        for student_out, target in zip(student_outputs, targets):
            loss = self.task_loss_fn(student_out, target)
            task_losses.append(loss)

        total_task_loss = sum(w * loss for w, loss in zip(horizon_weights, task_losses))

        # Compute feature-based KD loss
        feature_kd_result = self.feature_kd_loss(student_features, teacher_features)

        if isinstance(feature_kd_result, dict):
            feature_kd_loss = feature_kd_result['total_loss']
        else:
            feature_kd_loss = feature_kd_result
            feature_kd_result = {'total_loss': feature_kd_loss}

        # Combined loss
        total_loss = self.alpha * total_task_loss + (1 - self.alpha) * feature_kd_loss

        result = {
            'total_loss': total_loss,
            'task_loss': total_task_loss,
            'feature_kd_loss': feature_kd_loss,
            'task_losses_per_horizon': {
                f'horizon_{i}': loss.item()
                for i, loss in enumerate(task_losses)
            }
        }

        # Add method-specific metrics
        if self.method == "crd":
            result['crd_accuracy'] = feature_kd_result.get('contrastive_accuracy', torch.tensor(0.0))

        # Optional vanilla KD
        if self.include_vanilla_kd:
            vanilla_result = self.vanilla_kd_loss(student_outputs, teacher_outputs)
            vanilla_kd_loss = vanilla_result['total_loss']
            total_loss = total_loss + self.vanilla_kd_weight * vanilla_kd_loss
            result['total_loss'] = total_loss
            result['vanilla_kd_loss'] = vanilla_kd_loss

        return result
