# Locomotion Mode Recognition (LOSO) Configs

Configurations for **leave-one-subject-out (LOSO) cross-validation** on the 13 subjects with synchronized egocentric video and lower-body IMU recordings (7 healthy older adults and 6 stroke survivors).

## Available Configurations

### IMU-Only

| Config | Model |
|--------|-------|
| `deepconvlstm_acc.yaml` | DeepConvLSTM (accelerometer only) |
| `deepconvlstm_acc_gyro.yaml` | DeepConvLSTM (accelerometer + gyroscope) |

### Video-Only

| Config | Model |
|--------|-------|
| `resnet18.yaml` | ResNet-18 |
| `resnet50.yaml` | ResNet-50 |
| `mobilenetv3_small.yaml` | MobileNet-v3 Small |
| `x3d_xs.yaml` | X3D-XS |
| `x3d_m.yaml` | X3D-M |
| `mvit.yaml` | MViT |

### Multimodal Fusion (IMU + Video)

| Config | Model |
|--------|-------|
| `resnet18_dcl_fusion_average.yaml` | ResNet-18 + DeepConvLSTM (average) |
| `resnet18_dcl_fusion_concat_ln.yaml` | ResNet-18 + DeepConvLSTM (concat) |
| `mobilenetv3_small_dcl_fusion_average.yaml` | MobileNet-v3 + DeepConvLSTM (average) |
| `mobilenetv3_small_dcl_fusion_concat_ln.yaml` | MobileNet-v3 + DeepConvLSTM (concat) |
| `x3d_xs_dcl_fusion_average.yaml` | X3D-XS + DeepConvLSTM (average) |
| `x3d_xs_dcl_fusion_concat_ln.yaml` | X3D-XS + DeepConvLSTM (concat) |
| `mvit_dcl_fusion_average_loso.yaml` | MViT + DeepConvLSTM (average) |
| `mvit_dcl_fusion_concat_ln_loso.yaml` | MViT + DeepConvLSTM (concat) |
| `kifnet_fusion_average.yaml` | [KIFNet](https://github.com/Anvilondre/kifnet) (average) |
| `kifnet_fusion_concat_layernorm.yaml` | [KIFNet](https://github.com/Anvilondre/kifnet) (concat + LayerNorm) |
| `kifnet_style_fusion.yaml` | [KIFNet](https://github.com/Anvilondre/kifnet)-Style fusion |
| `sftik_sandwich_loso_lowerbody.yaml` | [SFTIK](https://github.com/RuoqiZhao116/SFTIK) sandwich fusion |
| `evi_mae_fusion.yaml` | [EVI-MAE](https://github.com/mf-zhang/IMU-Video-MAE) fusion |

## Usage

```bash
python loso_multi_horizon.py --base-config configs/loso/revalexo/<config_name>.yaml
```
