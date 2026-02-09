# Cross-Population Generalization Configs

Configurations for the **cross-population generalization** benchmark. Models are trained on healthy older adults (N=7) and tested on stroke survivors (N=6) using multimodal data (synchronized egocentric video and lower-body IMU).

## Available Configurations

| Config | Modality | Model |
|--------|----------|-------|
| `deepconvlstm_acc.yaml` | IMU (acc) | DeepConvLSTM |
| `deepconvlstm_acc_gyro.yaml` | IMU (acc+gyro) | DeepConvLSTM |
| `resnet18.yaml` | Video | ResNet-18 |
| `resnet50.yaml` | Video | ResNet-50 |
| `mobilenetv3_small.yaml` | Video | MobileNet-v3 Small |
| `mvit_adamW.yaml` | Video | MViT |
| `x3d_xs_adamW.yaml` | Video | X3D-XS |
| `resnet18_dcl_fusion_concat_ln.yaml` | IMU+Video | ResNet-18 + DeepConvLSTM (concat) |
| `resnet50_dcl_fusion_concat_ln.yaml` | IMU+Video | ResNet-50 + DeepConvLSTM (concat) |
| `mobilenetv3_small_dcl_fusion_concat_ln.yaml` | IMU+Video | MobileNet-v3 + DeepConvLSTM (concat) |
| `mvit_dcl_fusion_concat_ln.yaml` | IMU+Video | MViT + DeepConvLSTM (concat) |
| `x3d_xs_dcl_fusion_concat_ln.yaml` | IMU+Video | X3D-XS + DeepConvLSTM (concat) |
| `kifnet_style_fusion.yaml` | IMU+Video | [KIFNet](https://github.com/Anvilondre/kifnet)-Style fusion |
| `sftik_fusion.yaml` | IMU+Video | [SFTIK](https://github.com/RuoqiZhao116/SFTIK) fusion |
| `evi_mae_fusion.yaml` | IMU+Video | [EVI-MAE](https://github.com/mf-zhang/IMU-Video-MAE) fusion |

## Usage

```bash
python train.py --config configs/train/revalexo_cross_population/<config_name>.yaml
```
