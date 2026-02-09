# RevalExo: A Multimodal Dataset and Benchmark for Locomotion Mode Recognition Across Healthy and Clinical Populations

This is the official codebase for the paper *"RevalExo: A Multimodal Dataset and Benchmark for Locomotion Mode Recognition Across Healthy and Clinical Populations"*.

[[Project Page]](https://revalexo.github.io/)

## Setup

The code can be run under any environment with Python 3.12 and above.

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment using one of the provided environment files:

**CPU only:**

```bash
conda env create -f environment-cpu.yml
conda activate multimodal-env-cpu
```

**CUDA (GPU):**

```bash
conda env create -f environment-cuda.yml
conda activate multimodal-env-cuda
```

**CUDA with DGL** (required for EVI-MAE with graph neural network support):

```bash
conda env create -f environment-cuda-dgl.yml
conda activate multimodal-env-cuda-dgl
```

## Configuration

### Dataset

Download the RevalExo dataset from the following link:

> [[Download RevalExo Dataset]](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/OWJOID)

Download and extract `trimmed.zip` from the link above. Then, set the path to the `RevalExoDataset` folder in `configs/datasets/revalexodataset.yaml`:

```yaml
root_path: "/path/to/RevalExoDataset"
```

### Running Experiments

After preparing the dataset, you can use the provided configs to run training. The codebase supports three benchmark challenges from the paper.

<details>
<summary><b>Locomotion Mode Recognition (LOSO)</b></summary>

Leave-one-subject-out cross-validation on the 13 subjects with synchronized IMU and egocentric video. Run using `loso_multi_horizon.py` with the desired config:

```bash
python loso_multi_horizon.py --base-config configs/loso/revalexo/deepconvlstm_acc_gyro.yaml
```

Other examples:

```bash
# Video-only (ResNet18)
python loso_multi_horizon.py --base-config configs/loso/revalexo/resnet18.yaml

# Multimodal fusion (ResNet18 + DeepConvLSTM, average fusion)
python loso_multi_horizon.py --base-config configs/loso/revalexo/resnet18_dcl_fusion_average.yaml
```

See the config files in `configs/loso/revalexo/` for all available model configurations and guidance on setting up config values.

</details>

<details>
<summary><b>Cross-Population Generalization</b></summary>

Models are trained on healthy older adults (N=7) and tested on stroke survivors (N=6) with multimodal data. Run using `train.py`:

```bash
python train.py --config configs/train/revalexo_cross_population/deepconvlstm_acc_gyro.yaml
```

This trains the model on healthy subjects and evaluates on stroke subjects. See [`configs/train/revalexo_cross_population/README.md`](configs/train/revalexo_cross_population/README.md) for more details on the available configurations.

</details>

<details>
<summary><b>Vision-Guided Knowledge Transfer</b></summary>

Vision-guided transfer leverages video during training to improve an IMU-only student model. Two strategies are supported: (1) contrastive pretraining, which aligns IMU and video representations in a shared latent space, and (2) knowledge distillation, where a frozen multimodal teacher guides an IMU-only student.

Two cross-population transfer settings are available:

- `configs/train/revalexo_healthy_sarcopenic_cross_modality/` — Healthy &rarr; Sarcopenic
- `configs/train/revalexo_healthy_stroke_cross_modality/` — Healthy &rarr; Stroke

**Contrastive pretraining:**

Set the path to the contrastively pretrained IMU encoder checkpoint in the config:

```yaml
models:
  raw_imu_model:
    pretrained_checkpoint: "pretrained/contrastive/deepconvlstm_imu_encoder.pt"
```

Then run:

```bash
python train.py --config configs/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_contrastive_pretrained.yaml
```

**Knowledge distillation:**

First, train a multimodal teacher model (e.g., ResNet50+DCL fusion) and save the checkpoint. Then, set the teacher config and checkpoint path in the distillation config:

```yaml
distillation:
  teacher_config: "configs/train/revalexo_cross_population/resnet50_dcl_fusion_concat_ln.yaml"
  teacher_checkpoint: "path/to/teacher/checkpoint.pt"
```

Then run:

```bash
python train.py --config configs/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_kd_fitnets.yaml
```

</details>

<details>
<summary><b>EVI-MAE</b></summary>

[EVI-MAE](https://github.com/mf-zhang/IMU-Video-MAE) (Efficient Video-IMU Masked Autoencoder) is a multimodal fusion model that uses pretrained vision and IMU transformer encoders with optional graph neural network support for body skeleton encoding.

> **Pretraining from scratch**: To reproduce the EVI-MAE pretrained checkpoint, see [`../pretrain/README.md`](../pretrain/README.md).

The pretrained EVI-MAE checkpoint must be set in the config:

```yaml
models:
  fusion_model:
    name: "EVI_MAE_Fusion"
    pretrained_checkpoint: "pretrained/EVI-MAE/best_evi_model.pth"
```

Then run:

```bash
# LOSO evaluation
python loso_multi_horizon.py --base-config configs/loso/revalexo/evi_mae_fusion.yaml

# Cross-population generalization
python train.py --config configs/train/revalexo_cross_population/evi_mae_fusion.yaml
```

> **Note:** EVI-MAE with `enable_graph: true` requires DGL. Use the `environment-cuda-dgl.yml` environment for this.

</details>

### Pretrained Models

The `pretrained/` directory has the following structure:

```
pretrained/
├── contrastive/
│   └── deepconvlstm_imu_encoder.pt    # IMU encoder pretrained with X3D-XS via contrastive alignment (adapted from IMU2CLIP)
└── EVI-MAE/
    └── best_evi_model.pth              # EVI-MAE pretrained encoder
```

- `deepconvlstm_imu_encoder.pt` — DeepConvLSTM IMU encoder contrastively pretrained together with an X3D-XS video encoder, adapting the [IMU2CLIP](https://github.com/facebookresearch/imu2clip) approach to align IMU and video representations in a shared latent space.
- `best_evi_model.pth` — Pretrained [EVI-MAE](https://github.com/mf-zhang/IMU-Video-MAE) encoder.

Download the pretrained checkpoints from the link below and place them in the `pretrained/` directory, preserving the folder structure above:

> [[Download Pretrained Models]](https://kuleuven-my.sharepoint.com/:f:/g/personal/diwas_lamsal_kuleuven_be/IgDAiB2SZR5RS7hXB8x3UZNqAaVs7xccU_jvEpNV42VMlcM?e=7SefJQ)

## Citation

Coming soon.
