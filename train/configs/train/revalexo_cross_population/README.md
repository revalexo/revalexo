# Cross-Population Generalization Configs

Configurations for the **cross-population generalization** benchmark. Models are trained on healthy older adults (N=7) and tested on stroke survivors (N=6) using multimodal data (synchronized egocentric video and lower-body IMU).

## Available Configurations

### IMU-Only Models

| Config | Modality | Model |
|--------|----------|-------|
| `deepconvlstm_acc.yaml` | IMU (acc) | DeepConvLSTM |
| `deepconvlstm_acc_gyro.yaml` | IMU (acc+gyro) | DeepConvLSTM |
| `deepconvlstm_acc_gyro_contrastive_pretrained.yaml` | IMU (acc+gyro) | DeepConvLSTM (contrastive pretrained) |

### Image/Video Models (Fine-tuned)

| Config | Modality | Model |
|--------|----------|-------|
| `resnet18.yaml` | Image | ResNet-18 |
| `resnet50.yaml` | Image | ResNet-50 |
| `mobilenetv3_small.yaml` | Image | MobileNet-v3 Small |
| `mvit_adamW.yaml` | Video | MViT |
| `x3d_xs_adamW.yaml` | Video | X3D-XS |

### Image/Video Models (From Scratch)

| Config | Modality | Model |
|--------|----------|-------|
| `resnet18_scratch.yaml` | Image | ResNet-18 (from scratch) |
| `mvit_scratch.yaml` | Video | MViT (from scratch) |

### Image/Video Models (Frozen / Linear Probe)

| Config | Modality | Model |
|--------|----------|-------|
| `frozen/resnet18_linear_probe.yaml` | Image | ResNet-18 (linear probe) |
| `frozen/dinov3_linear_probe.yaml` | Image | DINOv3 (linear probe) |
| `frozen/mvit_linear_probe.yaml` | Video | MViT (linear probe) |
| `frozen/resnet18_dcl_fusion_concat_ln_linear_probe.yaml` | IMU+Image | ResNet-18 + DeepConvLSTM (linear probe) |

### Multimodal Fusion Models

| Config | Modality | Model |
|--------|----------|-------|
| `resnet18_dcl_fusion_concat_ln.yaml` | IMU+Image | ResNet-18 + DeepConvLSTM (concat) |
| `resnet50_dcl_fusion_concat_ln.yaml` | IMU+Image | ResNet-50 + DeepConvLSTM (concat) |
| `mobilenetv3_small_dcl_fusion_concat_ln.yaml` | IMU+Image | MobileNet-v3 + DeepConvLSTM (concat) |
| `mvit_dcl_fusion_concat_ln.yaml` | IMU+Video | MViT + DeepConvLSTM (concat) |
| `x3d_xs_dcl_fusion_concat_ln.yaml` | IMU+Video | X3D-XS + DeepConvLSTM (concat) |
| `kifnet_style_fusion.yaml` | IMU+Image | [KIFNet](https://github.com/Anvilondre/kifnet)-Style fusion |
| `sftik_fusion.yaml` | IMU+Image | [SFTIK](https://github.com/RuoqiZhao116/SFTIK) fusion |
| `evi_mae_fusion.yaml` | IMU+Video | [EVI-MAE](https://github.com/mf-zhang/IMU-Video-MAE) fusion |

## Results

See [RESULTS.md](RESULTS.md) for benchmark results.

## Usage

```bash
python train.py --config configs/train/revalexo_cross_population/<config_name>.yaml
```
