# Vision-Guided Knowledge Transfer Configs (Healthy &rarr; Sarcopenic)

Configurations for the **vision-guided knowledge transfer** benchmark. Models are trained on healthy older adults (N=7) and tested on older adults with probable sarcopenia (N=10) using IMU-only data at inference, with video available only during training.

## Available Configurations

| Config | Method |
|--------|--------|
| `deepconvlstm_acc_gyro.yaml` | Baseline (IMU-only, no transfer) |
| `deepconvlstm_acc_gyro_contrastive_pretrained.yaml` | Contrastive pretraining (CP) |
| `deepconvlstm_acc_gyro_kd_resnet50_dcl.yaml` | Knowledge distillation (vanilla KD) |
| `deepconvlstm_acc_gyro_kd_fitnets.yaml` | Knowledge distillation (FitNets) |
| `deepconvlstm_acc_gyro_kd_crd_membank.yaml` | Knowledge distillation (CRD with memory bank) |
| `deepconvlstm_acc_gyro_kd_nkd.yaml` | Knowledge distillation (NKD) |
| `deepconvlstm_acc_gyro_contrastive_pretrained_kd_fitnets.yaml` | Contrastive pretraining + Knowledge distillation (FitNets) |

## Results

See [RESULTS.md](RESULTS.md) for benchmark results.

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

### 5-seed training and evaluation

The benchmark numbers in [RESULTS.md](RESULTS.md) are averaged over 5 seeds (0, 1, 2, 3, 42). Example scripts that loop over all configs and seeds live in [`hpc_scripts/cross_modality/`](../../../hpc_scripts/cross_modality):

- Train (this cohort): [`hpc_scripts/cross_modality/train_sarcopenic/train_sarcopenic_5seed.sh`](../../../hpc_scripts/cross_modality/train_sarcopenic/train_sarcopenic_5seed.sh)
- Evaluate (this cohort): [`hpc_scripts/cross_modality/eval_sarcopenic/eval_sarcopenic_5seed.sh`](../../../hpc_scripts/cross_modality/eval_sarcopenic/eval_sarcopenic_5seed.sh)

Each script runs all configs by default, or a single config if you pass its name, e.g.:

```bash
bash hpc_scripts/cross_modality/train_sarcopenic/train_sarcopenic_5seed.sh deepconvlstm_acc_gyro_contrastive_pretrained_kd_fitnets
```
