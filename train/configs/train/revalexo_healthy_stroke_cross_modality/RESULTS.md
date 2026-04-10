# Vision-Guided Knowledge Transfer Results (Healthy &rarr; Stroke)

Vision-guided cross-modal knowledge transfer performance at &tau;=0.0 s and &tau;=0.5 s. Models are trained on healthy older adults (N=7) and tested on stroke survivors (N=10). Results are reported as mean &pm; SD Macro F1 (%) across subjects.

## Results at &tau;=0.0 s and &tau;=0.5 s

| Method | &tau;=0.0 s | | &tau;=0.5 s | | Logs |
|---|---|---|---|---|---|
| | Overall | Trans. | Overall | Trans. | |
| [Baseline](deepconvlstm_acc_gyro.yaml) | 59.9 &pm; 19.4 | 33.2 &pm; 13.9 | 52.4 &pm; 17.3 | 32.2 &pm; 12.3 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro/eval_20260409_181356_Subject05_to_Subject26/) |
| [CP](deepconvlstm_acc_gyro_contrastive_pretrained.yaml) | 62.5 &pm; 20.9 | 34.5 &pm; 13.3 | 53.6 &pm; 18.4 | 32.4 &pm; 11.5 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_contrastive_pretrained/eval_20260409_181234_Subject05_to_Subject26/) |
| [KD](deepconvlstm_acc_gyro_kd_resnet50_dcl.yaml) | 55.7 &pm; 20.2 | 32.1 &pm; 13.9 | 50.2 &pm; 18.9 | 31.2 &pm; 12.8 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_kd_resnet50_dcl/eval_20260409_170644_Subject05_to_Subject26/) |
| [FitNets](deepconvlstm_acc_gyro_kd_fitnets.yaml) | 62.9 &pm; 21.5 | 36.1 &pm; 13.0 | 55.0 &pm; 19.3 | 33.5 &pm; 13.1 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_kd_fitnets/eval_20260409_170925_Subject05_to_Subject26/) |
| [CRD](deepconvlstm_acc_gyro_kd_crd_membank.yaml) | 58.4 &pm; 20.2 | 33.0 &pm; 13.2 | 50.0 &pm; 17.9 | 30.6 &pm; 12.0 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_kd_crd_membank/eval_20260409_171203_Subject05_to_Subject26/) |
| [NKD](deepconvlstm_acc_gyro_kd_nkd.yaml) | 57.1 &pm; 20.2 | 31.0 &pm; 12.6 | 51.5 &pm; 18.8 | 31.2 &pm; 12.2 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/train/revalexo_healthy_stroke_cross_modality/deepconvlstm_acc_gyro_kd_nkd/eval_20260409_171440_Subject05_to_Subject26/) |

> Baseline: DCL (acc+gyro); CP: Contrastive Pretraining; KD: Vanilla Knowledge Distillation (ResNet-50+DCL teacher); CRD: Contrastive Representation Distillation (memory bank).

## Analysis

Full analysis outputs (plots, per-class breakdowns, per-subject reports) are available in:
```
outputs/train/revalexo_healthy_stroke_cross_modality/analysis_results/
```
