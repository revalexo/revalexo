# Locomotion Mode Recognition (LOSO) Results

Leave-one-subject-out cross-validation performance at &tau;=0.0 s and &tau;=0.5 s. Results are reported as mean &pm; SD Macro F1 (%) across subjects (N=13).

## Results at &tau;=0.0 s and &tau;=0.5 s

|                   | Method                                                   | &tau;=0.0 s  |               | &tau;=0.5 s  |              | Logs                                                                                                                                             |
| ----------------- | -------------------------------------------------------- | ------------ | ------------- | ------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
|                   |                                                          | Overall      | Trans.        | Overall      | Trans.       |                                                                                                                                                  |
| **IMU**     | [DCL (acc)](deepconvlstm_acc.yaml)                          | 78.6&pm; 8.6 | 50.6&pm; 9.8  | 70.9&pm; 9.2 | 46.4&pm; 8.1 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/deepconvlstm_acc_loso/20260407_232230/)                     |
|                   | [DCL (acc+gyro)](deepconvlstm_acc_gyro.yaml)                | 80.8&pm; 8.0 | 53.4&pm; 10.1 | 74.2&pm; 8.1 | 47.7&pm; 8.3 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/deepconvlstm_acc_gyro_loso/20260408_062751/)                |
| **Img**     | [MNV3-S](mobilenetv3_small.yaml)                            | 80.2&pm; 2.6 | 52.1&pm; 3.3  | 81.5&pm; 2.3 | 50.5&pm; 3.7 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/mobilenetv3_small_loso/20260408_153622/)                    |
|                   | [R18](resnet18.yaml)                                        | 82.8&pm; 2.1 | 53.8&pm; 3.0  | 83.2&pm; 1.7 | 52.2&pm; 3.2 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/resnet18_loso/20260407_232431/)                             |
|                   | [R50](resnet50.yaml)                                        | 82.8&pm; 1.8 | 54.2&pm; 3.1  | 83.6&pm; 2.0 | 52.5&pm; 3.3 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/resnet50_loso/20260409_061226/)                             |
| **Vid**     | [X3D-XS](x3d_xs.yaml)                                       | 86.1&pm; 3.0 | 47.8&pm; 6.9  | 82.4&pm; 4.6 | 40.0&pm; 7.9 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/x3d_xs_loso/20260408_134318/)                               |
|                   | [MViT](mvit.yaml)                                           | 90.9&pm; 2.2 | 62.9&pm; 3.6  | 87.8&pm; 2.2 | 58.0&pm; 3.5 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/mvit_loso/20260408_143840/)                                 |
| **IMU+Img** | [KIFNet (A)](kifnet_fusion_average.yaml)                    | 87.5&pm; 2.7 | 55.2&pm; 7.9  | 83.7&pm; 2.1 | 49.3&pm; 6.4 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/kifnet_average_loso/20260407_233803/)                       |
|                   | [KIFNet (C)](kifnet_fusion_concat_layernorm.yaml)           | 87.0&pm; 3.0 | 55.5&pm; 6.1  | 83.6&pm; 2.0 | 50.8&pm; 5.8 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/kifnet_concat_layernorm_loso/20260408_192025/)              |
|                   | [KIFNet-Style](kifnet_style_fusion.yaml)                    | 84.7&pm; 2.5 | 51.5&pm; 7.0  | 80.8&pm; 2.8 | 47.8&pm; 6.7 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/kifnet_style_loso/20260408_160421/)                         |
|                   | [MNV3-DCL (A)](mobilenetv3_small_dcl_fusion_average.yaml)   | 91.4&pm; 1.7 | 65.1&pm; 6.1  | 88.1&pm; 2.2 | 57.8&pm; 5.6 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/mobilenetv3_small_dcl_fusion_average_loso/20260409_004726/) |
|                   | [MNV3-DCL (C)](mobilenetv3_small_dcl_fusion_concat_ln.yaml) | 91.3&pm; 1.6 | 65.8&pm; 4.0  | 87.9&pm; 2.4 | 59.4&pm; 4.7 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/mobilenetv3_small_dcl_fusion_concat_ln_loso/20260409_175625/)                                                                                                                                                  |
|                   | [R18-DCL (A)](resnet18_dcl_fusion_average.yaml)             | 92.3&pm; 1.7 | 65.9&pm; 5.3  | 88.5&pm; 2.0 | 58.7&pm; 4.2 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/resnet18_dcl_fusion_average_loso/20260407_232904/)          |
|                   | [R18-DCL (C)](resnet18_dcl_fusion_concat_ln.yaml)           | 92.1&pm; 1.7 | 66.2&pm; 5.0  | 88.6&pm; 2.1 | 59.3&pm; 5.5 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/resnet18_dcl_fusion_concat_ln_loso/20260409_171700/)        |
| **IMU+Vid** | [SFTIK](sftik_sandwich_loso_lowerbody.yaml)                 | -            | -             | -            | -            |                                                                                                                                                  |
|                   | [EVI-MAE](evi_mae_fusion.yaml)                              | 90.3&pm; 2.0 | 61.8&pm; 4.8  | 88.0&pm; 2.4 | 57.5&pm; 5.8 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/evi_mae_fusion_loso/20260408_220920/)                                                                                                                                                  |
|                   | [X3D-XS-DCL (A)](x3d_xs_dcl_fusion_average.yaml)            | 90.1&pm; 3.8 | 56.2&pm; 7.7  | 83.2&pm; 5.6 | 41.3&pm; 9.0 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/x3d_xs_dcl_fusion_average_loso/20260411_213432/)            |
|                   | [X3D-XS-DCL (C)](x3d_xs_dcl_fusion_concat_ln.yaml)          | -            | -             | -            | -            |                                                                                                                                                  |
|                   | [MViT-DCL (A)](mvit_dcl_fusion_average_loso.yaml)           | 92.7&pm; 1.7 | 67.6&pm; 4.6  | 88.8&pm; 2.5 | 61.0&pm; 5.2 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/mvit_dcl_fusion_average_loso_loso/20260408_151513/)         |
|                   | [MViT-DCL (C)](mvit_dcl_fusion_concat_ln_loso.yaml)         | 92.8&pm; 2.0 | 68.2&pm; 5.4  | 89.6&pm; 2.3 | 61.9&pm; 4.5 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/mvit_dcl_fusion_concat_ln_loso_loso/20260408_144245/)       |

> DCL: DeepConvLSTM; MNV3: MobileNetV3; R18: ResNet-18; R50: ResNet-50; A: Feature average; C: Feature concatenation. `-` indicates results not yet available.

## From-Scratch Models

|               | Method                              | &tau;=0.0 s  |              | &tau;=0.5 s  |              | Logs                                                                                                                         |
| ------------- | ----------------------------------- | ------------ | ------------ | ------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
|               |                                     | Overall      | Trans.       | Overall      | Trans.       |                                                                                                                              |
| **Img** | [R18 (scratch)](resnet18_scratch.yaml) | 80.0&pm; 2.8 | 52.7&pm; 3.5 | 80.9&pm; 2.7 | 51.0&pm; 3.8 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/resnet18_scratch_loso/20260409_014705/) |

## Frozen / Linear Probe Models

|                   | Method                                                              | &tau;=0.0 s  |              | &tau;=0.5 s  |              | Logs                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------- | ------------ | ------------ | ------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                   |                                                                     | Overall      | Trans.       | Overall      | Trans.       |                                                                                                                                                               |
| **Img**     | [R18 (LP)](frozen/resnet18_linear_probe.yaml)                          | 77.3&pm; 2.8 | 49.8&pm; 3.8 | 78.8&pm; 3.1 | 48.6&pm; 4.2 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/frozen/resnet18_linear_probe_loso/20260408_023159/)                      |
|                   | [DINOv3 (LP)](frozen/dinov3_linear_probe.yaml)                         | 79.1&pm; 2.5 | 52.5&pm; 2.7 | 81.0&pm; 2.1 | 51.6&pm; 2.7 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/frozen/dinov3_linear_probe_loso/20260409_004200/)                        |
| **IMU+Img** | [R18-DCL (LP)](frozen/resnet18_dcl_fusion_concat_ln_linear_probe.yaml) | 89.3&pm; 3.0 | 61.8&pm; 7.5 | 85.8&pm; 2.7 | 56.6&pm; 5.8 | [&#128194;](https://huggingface.co/datasets/revalexo/train-logs/tree/main/loso/revalexo/frozen/resnet18_dcl_fusion_concat_ln_linear_probe_loso/20260408_173055/) |

> LP: Linear Probe (backbone frozen, only classification head trained).

## Analysis

Full analysis outputs (plots, per-class breakdowns, per-subject reports) are available in:

```
outputs/loso/revalexo/analysis_results/
```
