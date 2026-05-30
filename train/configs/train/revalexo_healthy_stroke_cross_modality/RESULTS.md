# Vision-Guided Knowledge Transfer Results (Healthy &rarr; Stroke)

Vision-guided cross-modal knowledge transfer performance at &tau;=0.0 s and &tau;=0.5 s. Models are trained on healthy older adults (N=7) and tested on stroke survivors (N=10). Results are reported as 5-seed average Macro F1 (%) across subjects.

## Results at &tau;=0.0 s and &tau;=0.5 s

| Method | &tau;=0.0 s | | &tau;=0.5 s | |
|---|---|---|---|---|
| | Overall | Trans. | Overall | Trans. |
| [Baseline](deepconvlstm_acc_gyro.yaml) | 57.7 | 35.0 | 51.2 | 32.9 |
| [NKD](deepconvlstm_acc_gyro_kd_nkd.yaml) | 59.2 | 34.8 | 51.9 | 32.7 |
| [KD](deepconvlstm_acc_gyro_kd_resnet50_dcl.yaml) | 58.0 | 35.1 | 51.8 | 32.8 |
| [FitNets](deepconvlstm_acc_gyro_kd_fitnets.yaml) | 60.5 | 36.3 | 53.9 | 34.0 |
| [CRD](deepconvlstm_acc_gyro_kd_crd_membank.yaml) | 60.3 | 36.5 | 52.6 | 32.9 |
| [CP](deepconvlstm_acc_gyro_contrastive_pretrained.yaml) | 63.6 | 35.5 | 54.5 | 32.4 |
| [CP+FitNets](deepconvlstm_acc_gyro_contrastive_pretrained_kd_fitnets.yaml) | 64.6 | 36.7 | 55.2 | 32.3 |

> Baseline: DCL (acc+gyro); CP: Contrastive Pretraining; KD: Vanilla Knowledge Distillation (ResNet-50+DCL teacher); NKD: Normalized Knowledge Distillation; FitNets: FitNet feature distillation; CRD: Contrastive Representation Distillation (memory bank); CP+FitNets: CP initialization followed by FitNets feature distillation.

## Analysis

Full analysis outputs (plots, per-class breakdowns, per-subject reports) are available in:
```
outputs/train/revalexo_healthy_stroke_cross_modality/analysis_results/
```
