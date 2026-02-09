# Vision-Guided Knowledge Transfer Configs (Healthy &rarr; Sarcopenic)

Configurations for the **vision-guided knowledge transfer** benchmark. Models are trained on healthy older adults (N=7) and tested on older adults with probable sarcopenia (N=10) using IMU-only data at inference, with video available only during training.

## Available Configurations

| Config | Method |
|--------|--------|
| `deepconvlstm_acc_gyro.yaml` | Baseline (IMU-only, no transfer) |
| `deepconvlstm_acc_gyro_contrastive_pretrained.yaml` | Contrastive pretraining |
| `deepconvlstm_acc_gyro_kd_resnet50_dcl.yaml` | Knowledge distillation (vanilla KD) |
| `deepconvlstm_acc_gyro_kd_fitnets.yaml` | Knowledge distillation (FitNets) |
| `deepconvlstm_acc_gyro_kd_crd_membank.yaml` | Knowledge distillation (CRD with memory bank) |
| `deepconvlstm_acc_gyro_kd_nkd.yaml` | Knowledge distillation (NKD) |

## Setup

**Contrastive pretraining** requires a pretrained IMU encoder checkpoint. Override the path in the config:

```yaml
models:
  raw_imu_model:
    pretrained_checkpoint: "pretrained/contrastive/deepconvlstm_imu_encoder.pt"
```

**Knowledge distillation** requires a trained multimodal teacher checkpoint. Override the teacher config and checkpoint path:

```yaml
distillation:
  teacher_config: "configs/train/revalexo_cross_population/resnet50_dcl_fusion_concat_ln.yaml"
  teacher_checkpoint: "path/to/teacher/checkpoint.pt"
```

## Usage

```bash
python train.py --config configs/train/revalexo_healthy_sarcopenic_cross_modality/<config_name>.yaml
```
