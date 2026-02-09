# losses/__init__.py
"""
Custom loss functions for training.
"""

from .distillation import (
    VanillaKDLoss,
    MultiHorizonKDLoss,
    DistillationLoss,
    NKDLoss,
    MultiHorizonNKDLoss,
    NKDDistillationLoss,
    FitNetsLoss,
    CRDLoss,
    FeatureDistillationLoss
)

__all__ = [
    # Standard KD
    'VanillaKDLoss',
    'MultiHorizonKDLoss',
    'DistillationLoss',
    # NKD
    'NKDLoss',
    'MultiHorizonNKDLoss',
    'NKDDistillationLoss',
    # Feature-based KD
    'FitNetsLoss',
    'CRDLoss',
    'FeatureDistillationLoss'
]
