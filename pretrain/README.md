# Pretraining

This directory contains the pretraining pipelines for self-supervised models used in RevalExo:

- [**EVI-MAE**](#evi-mae-imu-video-masked-autoencoder) - Masked autoencoder pretraining (IMU + video reconstruction)
- [**Contrastive (IMU2CLIP)**](#contrastive-pretraining-imu2clip) - Cross-modal contrastive alignment (IMU ↔ video)

## EVI-MAE (IMU-Video Masked Autoencoder)

EVI-MAE performs multimodal masked autoencoder pretraining on synchronized IMU and egocentric video data. The pretrained encoder is then used for downstream locomotion mode recognition in the [`train/`](../train/) directory.

The code is adapted from [IMU-Video-MAE](https://github.com/mf-zhang/IMU-Video-MAE) (ECCV 2024). See [`imu-video-mae/ATTRIBUTION.md`](imu-video-mae/ATTRIBUTION.md) for details on modifications.

### Prerequisites

1. **Conda environment** (separate from the training environment):

   ```bash
   conda create -n evi-mae python=3.11
   conda activate evi-mae
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
   conda install -c dglteam/label/th24_cu124 dgl
   pip install decord timm==0.4.5 einops scikit-learn pandas
   ```
2. **VideoMAE checkpoint**: Download the adapted VideoMAE encoder checkpoint:

   > [[Download Checkpoint]](https://drive.google.com/file/d/1JxQtmgoxIxqFdY-3CAjUlZvs2JTn_l3A/view?usp=share_link)
   >

   Place it at:

   ```
   imu-video-mae/data_release/videomae_adapt_ckpt/video_imu_beforepretrain_model_small_only_encoder.pth
   ```
3. **RevalExo dataset**: Download from:

   > [[Download RevalExo Dataset]](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/OWJOID)
   >

   Download and extract `trimmed.zip`.

### Step 1: Convert RevalExo Data to EVI-MAE Format

The RevalExo dataset has 7 lower-body IMU sensors, but EVI-MAE expects 4 body parts (12 channels). The migration script selects 4 sensors and maps them to the expected format:

| EVI-MAE Slot | RevalExo Sensor |
| :----------- | :-------------- |
| left_arm     | Left_Foot       |
| right_arm    | Right_Foot      |
| left_leg     | Left_Upper_Leg  |
| right_leg    | Right_Upper_Leg |

Run the conversion:

```bash
cd imu-video-mae
python3 migrate_revalexo_data.py --source_dir /path/to/RevalExoDataset --target_dir ./data_release/revalexo-release
```

This produces:

- 2,279 synchronized IMU-video pairs (1,823 train / 456 test)
- IMU statistics file for normalization (mean: -1.22, std: 7.99)
- JSON label files for pretraining and fine-tuning splits

### Step 2: Run Pretraining

```bash
cd imu-video-mae/egs/release
bash pretrain_revalexo.sh
```

Key hyperparameters:

| Parameter             | Value          |
| :-------------------- | :------------- |
| Batch size            | 16             |
| Epochs                | 300            |
| IMU channels          | 12             |
| IMU masking ratio     | 0.75           |
| Video masking ratio   | 0.90           |
| IMU encoder           | 768d, 11L, 12H |
| Video encoder (ViT-S) | 384d, 12L, 6H  |
| Graph neural network  | GIN            |

Step 3: Use Pretrained Weights for Fine-tuning

After pretraining completes, copy the best checkpoint to the training directory:

```bash
cp imu-video-mae/data_release/revalexo-release/evi-mae-exp/[TIMESTAMP]/models/best_evi_model.pth \
   ../train/pretrained/EVI-MAE/best_evi_model.pth
```

Then run fine-tuning from the `train/` directory (see [`train/README.md`](../train/README.md) for details):

```bash
cd ../train
python loso_multi_horizon.py --base-config configs/loso/revalexo/evi_mae_fusion.yaml
```

### Pre-computed Checkpoint

To skip pretraining, download the pre-computed checkpoint and place it directly in the training directory:

> [[Download Pretrained Models]](https://kuleuven-my.sharepoint.com/:f:/g/personal/diwas_lamsal_kuleuven_be/IgDAiB2SZR5RS7hXB8x3UZNqAaVs7xccU_jvEpNV42VMlcM?e=7SefJQ)

```
train/pretrained/EVI-MAE/best_evi_model.pth
```

---

## Contrastive Pretraining (IMU2CLIP)

Contrastive pretraining aligns IMU and video representations in a shared embedding space using InfoNCE loss. The pretrained IMU encoder (DeepConvLSTM) is then used for vision-guided knowledge transfer in the [`train/`](../train/) directory.

The code is adapted from [IMU2CLIP](https://github.com/facebookresearch/imu2clip) (Meta Research, 2022). See [`contrastive/ATTRIBUTION.md`](contrastive/ATTRIBUTION.md) for details on modifications.

### Prerequisites

1. **Conda environment**: Uses the same environment as training:

   ```bash
   conda env create -f ../train/environment-cuda.yml
   conda activate multimodal-env-cuda
   ```
2. **RevalExo dataset**: Download from:

   > [[Download RevalExo Dataset]](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/OWJOID)
   >

   Download and extract `trimmed.zip`.

### Step 1: Configure Dataset Path

Edit the config file to set the path to the RevalExo dataset:

```yaml
# In configs/pretrain_deepconv_lower_x3d_revalexo.yaml
dataset:
  root_path: "/path/to/RevalExoDataset"
```

### Step 2: Run Pretraining

```bash
cd contrastive
bash run_training.sh
```

Or run directly with a specific config:

```bash
cd contrastive
python3 pretrain.py --config configs/pretrain_deepconv_lower_x3d_revalexo.yaml --gpus 1
```

Two visual encoder variants are available:

| Parameter            | X3D-XS          | MViT-B (16x4)    |
| :------------------- | :-------------- | :--------------- |
| Config               | `..._x3d_...` | `..._mvit_...` |
| Batch size           | 64              | 8                |
| IMU channels         | 42              | 42               |
| Joint embedding dim  | 256             | 512              |
| Video frames sampled | 4               | 16               |
| Epochs               | 50              | 50               |

### Step 3: Use Pretrained IMU Encoder for Fine-tuning

After pretraining, copy the IMU encoder checkpoint:

```bash
cp contrastive/checkpoints/[EXPERIMENT]/imu_encoder_best.pt \
   ../train/pretrained/contrastive/deepconvlstm_imu_encoder.pt
```

Then run vision-guided knowledge transfer from the `train/` directory:

```bash
cd ../train
python train.py --config configs/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_contrastive_pretrained.yaml
```

### Pre-computed Checkpoint

To skip pretraining, download the pre-computed checkpoint:

> [[Download Pretrained Models]](https://kuleuven-my.sharepoint.com/:f:/g/personal/diwas_lamsal_kuleuven_be/IgDAiB2SZR5RS7hXB8x3UZNqAaVs7xccU_jvEpNV42VMlcM?e=7SefJQ)

```
train/pretrained/contrastive/deepconvlstm_imu_encoder.pt
```
