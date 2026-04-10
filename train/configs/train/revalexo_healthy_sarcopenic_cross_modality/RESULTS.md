# Vision-Guided Knowledge Transfer Results (Healthy &rarr; Sarcopenic)

Vision-guided cross-modal knowledge transfer performance at &tau;=0.0 s and &tau;=0.5 s. Models are trained on healthy older adults (N=7) and tested on older adults with probable sarcopenia (N=10). Results are reported as mean &pm; SD Macro F1 (%) across subjects.

## Results at &tau;=0.0 s and &tau;=0.5 s

| Method | &tau;=0.0 s | | &tau;=0.5 s | | Logs |
|---|---|---|---|---|---|
| | Overall | Trans. | Overall | Trans. | |
| [Baseline](deepconvlstm_acc_gyro.yaml) | 42.1 &pm; 20.1 | 22.5 &pm; 8.7 | 36.1 &pm; 17.9 | 22.3 &pm; 11.3 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_sarcopenic_cross_modality/deepconvlstm_acc_gyro/eval_20260409_171833_Subject04_to_Subject30/) |
| [CP](deepconvlstm_acc_gyro_contrastive_pretrained.yaml) | 46.4 &pm; 22.6 | 24.5 &pm; 10.9 | 37.3 &pm; 19.2 | 20.7 &pm; 11.2 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_sarcopenic_cross_modality/deepconvlstm_acc_gyro_contrastive_pretrained/20260409_051702_RevalExoDataset_raw_imu_DeepConvLSTM/) |
| [KD](deepconvlstm_acc_gyro_kd_resnet50_dcl.yaml) | 38.4 &pm; 20.0 | 19.9 &pm; 9.5 | 34.3 &pm; 18.6 | 20.4 &pm; 11.6 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_sarcopenic_cross_modality/deepconvlstm_acc_gyro_kd_resnet50_dcl/20260409_054627_RevalExoDataset_raw_imu-image_DeepConvLSTM/) |
| [FitNets](deepconvlstm_acc_gyro_kd_fitnets.yaml) | 45.2 &pm; 22.4 | 23.3 &pm; 10.5 | 35.8 &pm; 19.6 | 21.2 &pm; 12.1 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_sarcopenic_cross_modality/deepconvlstm_acc_gyro_kd_fitnets/20260409_065052_RevalExoDataset_raw_imu-image_DeepConvLSTM/) |
| [CRD](deepconvlstm_acc_gyro_kd_crd_membank.yaml) | 43.4 &pm; 21.0 | 24.1 &pm; 10.0 | 35.5 &pm; 18.4 | 21.9 &pm; 10.9 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_sarcopenic_cross_modality/deepconvlstm_acc_gyro_kd_crd_membank/20260409_051702_RevalExoDataset_raw_imu-image_DeepConvLSTM/) |
| [NKD](deepconvlstm_acc_gyro_kd_nkd.yaml) | 42.6 &pm; 20.2 | 22.1 &pm; 8.8 | 36.9 &pm; 17.9 | 22.5 &pm; 10.7 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_sarcopenic_cross_modality/deepconvlstm_acc_gyro_kd_nkd/20260409_062036_RevalExoDataset_raw_imu-image_DeepConvLSTM/) |

> Baseline: DCL (acc+gyro); CP: Contrastive Pretraining; KD: Vanilla Knowledge Distillation (ResNet-50+DCL teacher); CRD: Contrastive Representation Distillation (memory bank).

## Analysis

Full analysis outputs (plots, per-class breakdowns, per-subject reports) are available in:
```
outputs/train/revalexo_healthy_sarcopenic_cross_modality/analysis_results/
```
