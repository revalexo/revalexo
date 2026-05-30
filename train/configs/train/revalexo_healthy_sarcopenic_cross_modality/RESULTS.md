# Vision-Guided Knowledge Transfer Results (Healthy &rarr; Sarcopenic)

Vision-guided cross-modal knowledge transfer performance at &tau;=0.0 s and &tau;=0.5 s. Models are trained on healthy older adults (N=7) and tested on older adults with probable sarcopenia (N=10). Results are reported as 5-seed average Macro F1 (%) across subjects. Training logs for all models are available [here](https://huggingface.co/datasets/wearablehar/train-logs/tree/main/train/revalexo_healthy_sarcopenic_cross_modality).

## Results at &tau;=0.0 s and &tau;=0.5 s

| Method                                                                  | &tau;=0.0 s |        | &tau;=0.5 s |        |
| ----------------------------------------------------------------------- | ----------- | ------ | ----------- | ------ |
|                                                                         | Overall     | Trans. | Overall     | Trans. |
| [Baseline](deepconvlstm_acc_gyro.yaml)                                     | 39.9        | 22.1   | 34.8        | 21.5   |
| [NKD](deepconvlstm_acc_gyro_kd_nkd.yaml)                                   | 41.4        | 22.4   | 35.7        | 21.5   |
| [KD](deepconvlstm_acc_gyro_kd_resnet50_dcl.yaml)                           | 41.5        | 23.2   | 36.3        | 22.4   |
| [FitNets](deepconvlstm_acc_gyro_kd_fitnets.yaml)                           | 41.8        | 23.7   | 35.9        | 21.5   |
| [CRD](deepconvlstm_acc_gyro_kd_crd_membank.yaml)                           | 42.3        | 24.3   | 35.2        | 21.7   |
| [CP](deepconvlstm_acc_gyro_contrastive_pretrained.yaml)                    | 46.4        | 25.1   | 38.1        | 22.1   |
| [CP+FitNets](deepconvlstm_acc_gyro_contrastive_pretrained_kd_fitnets.yaml) | 47.3        | 25.3   | 38.5        | 21.5   |

> Baseline: DCL (acc+gyro); CP: Contrastive Pretraining; KD: Vanilla Knowledge Distillation (ResNet-50+DCL teacher); NKD: Normalized Knowledge Distillation; FitNets: FitNet feature distillation; CRD: Contrastive Representation Distillation (memory bank); CP+FitNets: CP initialization followed by FitNets feature distillation.
