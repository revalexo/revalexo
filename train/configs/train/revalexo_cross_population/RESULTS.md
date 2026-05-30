# Cross-Population Generalization Results

Cross-population generalization performance at &tau;=0.0 s and &tau;=0.5 s. Models are trained on healthy older adults (N=7) and tested on stroke survivors (N=6) with multimodal data. Results are reported as mean &pm; SD Macro F1 (%) across subjects. Training logs for all models are available [here](https://huggingface.co/datasets/wearablehar/train-logs/tree/main/train/revalexo_cross_population).

## Results at &tau;=0.0 s and &tau;=0.5 s

|                   | Method                                                               | &tau;=0.0 s   |               | &tau;=0.5 s  |              |
| ----------------- | -------------------------------------------------------------------- | ------------- | ------------- | ------------ | ------------ |
|                   |                                                                      | Overall       | Trans.        | Overall      | Trans.       |
| **IMU**     | [DCL (acc)](deepconvlstm_acc.yaml)                                      | 59.9&pm; 10.3 | 34.7&pm; 10.7 | 53.5&pm; 9.7 | 33.5&pm; 8.9 |
|                   | [DCL (acc+gyro)](deepconvlstm_acc_gyro.yaml)                            | 65.6&pm; 9.5  | 36.8&pm; 11.8 | 59.3&pm; 8.7 | 37.9&pm; 9.9 |
|                   | [DCL (acc+gyro, CP)](deepconvlstm_acc_gyro_contrastive_pretrained.yaml) | 72.8&pm; 7.7  | 38.3&pm; 13.2 | 62.1&pm; 8.5 | 36.6&pm; 9.7 |
| **Img**     | [MNV3-S](mobilenetv3_small.yaml)                                        | 81.9&pm; 2.5  | 54.3&pm; 2.1  | 79.5&pm; 3.1 | 50.5&pm; 1.0 |
|                   | [R18](resnet18.yaml)                                                    | 82.0&pm; 2.5  | 52.6&pm; 1.1  | 80.6&pm; 2.8 | 51.2&pm; 1.3 |
|                   | [R50](resnet50.yaml)                                                    | 81.8&pm; 2.6  | 52.5&pm; 2.2  | 80.1&pm; 2.1 | 50.5&pm; 1.5 |
| **Vid**     | [X3D-XS](x3d_xs_adamW.yaml)                                             | 80.5&pm; 2.7  | 46.5&pm; 4.5  | 81.8&pm; 1.5 | 39.8&pm; 6.0 |
|                   | [MViT](mvit_adamW.yaml)                                                 | 86.7&pm; 2.8  | 57.6&pm; 3.6  | 84.6&pm; 3.1 | 54.3&pm; 3.5 |
| **IMU+Img** | [KIFNet-Style](kifnet_style_fusion.yaml)                                | 67.2&pm; 5.7  | 39.0&pm; 8.4  | 68.3&pm; 4.7 | 43.3&pm; 5.7 |
|                   | [MNV3-DCL (C)](mobilenetv3_small_dcl_fusion_concat_ln.yaml)             | 87.1&pm; 1.9  | 56.5&pm; 6.1  | 83.5&pm; 1.2 | 52.3&pm; 3.6 |
|                   | [R18-DCL (C)](resnet18_dcl_fusion_concat_ln.yaml)                       | 87.1&pm; 2.5  | 54.3&pm; 7.6  | 83.6&pm; 1.1 | 50.2&pm; 5.3 |
|                   | [R50-DCL (C)](resnet50_dcl_fusion_concat_ln.yaml)                       | 88.6&pm; 1.5  | 58.8&pm; 4.7  | 85.4&pm; 0.6 | 52.8&pm; 3.4 |
| **IMU+Vid** | [SFTIK](sftik_fusion.yaml)                                              | 84.0&pm; 3.1  | 52.3&pm; 3.8  | 82.1&pm; 3.6 | 53.9&pm; 2.6 |
|                   | [EVI-MAE](evi_mae_fusion.yaml)                                          | 86.5&pm; 2.8  | 55.4&pm; 6.1  | 85.2&pm; 2.3 | 53.8&pm; 3.8 |
|                   | [X3D-XS-DCL (C)](x3d_xs_dcl_fusion_concat_ln.yaml)                      | 85.4&pm; 3.7  | 43.9&pm; 9.8  | 76.2&pm; 3.3 | 31.3&pm; 8.3 |
|                   | [MViT-DCL (C)](mvit_dcl_fusion_concat_ln.yaml)                          | 88.1&pm; 2.4  | 56.1&pm; 5.7  | 85.4&pm; 1.4 | 53.5&pm; 3.7 |

> DCL: DeepConvLSTM; MNV3: MobileNetV3; R18: ResNet-18; R50: ResNet-50; CP: Contrastive Pretrained; C: Feature concatenation. `-` indicates results not yet available.

## Frozen / Linear Probe Models

|                   | Method                                                              | &tau;=0.0 s  |               | &tau;=0.5 s  |              |
| ----------------- | ------------------------------------------------------------------- | ------------ | ------------- | ------------ | ------------ |
|                   |                                                                     | Overall      | Trans.        | Overall      | Trans.       |
| **Img**     | [R18 (LP)](frozen/resnet18_linear_probe.yaml)                          | 73.7&pm; 3.3 | 49.2&pm; 2.2  | 74.5&pm; 3.4 | 47.4&pm; 2.9 |
|                   | [DINOv3 (LP)](frozen/dinov3_linear_probe.yaml)                         | 76.6&pm; 3.1 | 52.5&pm; 3.0  | 77.4&pm; 2.4 | 51.6&pm; 1.6 |
| **Vid**     | [MViT (LP)](frozen/mvit_linear_probe.yaml)                             | 75.4&pm; 4.4 | 49.2&pm; 3.9  | 76.0&pm; 4.1 | 47.4&pm; 2.6 |
| **IMU+Img** | [R18-DCL (LP)](frozen/resnet18_dcl_fusion_concat_ln_linear_probe.yaml) | 79.8&pm; 5.0 | 48.9&pm; 11.7 | 76.6&pm; 3.4 | 44.3&pm; 8.9 |

> LP: Linear Probe (backbone frozen, only classification head trained).

## Analysis

Per-class breakdowns, per-subject reports, and summary plots can be regenerated from the training logs with [`tools/analyze_train_results.py`](../../../tools/analyze_train_results.py):

```
python3 tools/analyze_train_results.py \
    --base-dir outputs/train/revalexo_cross_population \
    --output-dir analysis_results
```
