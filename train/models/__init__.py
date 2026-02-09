# models/__init__.py

from .base_models import BaseEncoder, MultiHorizonClassifier
from .fusion import FusionModel
from .deepconvlstm import DeepConvLSTM
from .pytorchvideo_models import X3D_Video, MViT_Video, R3D_18_Video
from .resnet_models import ResNet_Image, ResNet18_Image, ResNet34_Image, ResNet50_Image

from .mobilenet_models import (
    MobileNet_Image,
    MobileNetV2_Image,
    MobileNetV3Small_Image,
    MobileNetV3Large_Image
)
from .sftik_model import SFTIK, SFTIK_IMUEncoder, SFTIK_VideoEncoder
from .sftik_fusion import SFTIK_Fusion

try:
    from .evi_mae_fusion import EVI_MAE_Encoder, EVI_MAE_Fusion
    HAS_DGL = True
except (ImportError, FileNotFoundError, OSError) as e:
    # DGL may raise FileNotFoundError if C++ library is missing
    print(f"Could not import EVI_MAE (DGL not available): {e}")
    HAS_DGL = False

from .kifnet_mlp import KIFNetMLP, KIFNetMLPSeparateDecoders
from .mobileone_image import MobileOneS0_Image

__all__ = [
    # Base classes
    'BaseEncoder',
    'MultiHorizonClassifier',
    'FusionModel',
    # IMU models
    'DeepConvLSTM',
    'SFTIK_IMUEncoder',
    # Image models - ResNet
    'ResNet_Image',
    'ResNet18_Image',
    'ResNet34_Image',
    'ResNet50_Image',
    # Image models - MobileNet
    'MobileNet_Image',
    'MobileNetV2_Image',
    'MobileNetV3Small_Image',
    'MobileNetV3Large_Image',
    # Video models
    'X3D_Video',
    'MViT_Video',
    'R3D_18_Video',
    'SFTIK_VideoEncoder',
    # SFTIK models
    'SFTIK',
    'SFTIK_Fusion',
    # MobileOne models
    'MobileOneS0_Image',
]

if HAS_DGL:
    __all__.extend(['EVI_MAE_Encoder', 'EVI_MAE_Fusion'])