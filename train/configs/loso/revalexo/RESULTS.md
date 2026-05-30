# Locomotion Mode Recognition (LOSO) Results

Leave-one-subject-out cross-validation performance at &tau;=0.0 s and &tau;=0.5 s. Results are reported as mean &pm; SD Macro F1 (%) across subjects (N=13). Training logs for all models are available [here](https://huggingface.co/datasets/wearablehar/train-logs/tree/main/loso/revalexo).

## Results at &tau;=0.0 s and &tau;=0.5 s

|                   | Method                                                   | &tau;=0.0 s  |               | &tau;=0.5 s  |              |
| ----------------- | -------------------------------------------------------- | ------------ | ------------- | ------------ | ------------ |
|                   |                                                          | Overall      | Trans.        | Overall      | Trans.       |
| **IMU**     | [DCL (acc)](deepconvlstm_acc.yaml)                          | 78.6&pm; 8.6 | 50.6&pm; 9.8  | 70.9&pm; 9.2 | 46.4&pm; 8.1 |
|                   | [DCL (acc+gyro)](deepconvlstm_acc_gyro.yaml)                | 80.8&pm; 8.0 | 53.4&pm; 10.1 | 74.2&pm; 8.1 | 47.7&pm; 8.3 |
| **Img**     | [MNV3-S](mobilenetv3_small.yaml)                            | 80.2&pm; 2.6 | 52.1&pm; 3.3  | 81.5&pm; 2.3 | 50.5&pm; 3.7 |
|                   | [R18](resnet18.yaml)                                        | 82.8&pm; 2.1 | 53.8&pm; 3.0  | 83.2&pm; 1.7 | 52.2&pm; 3.2 |
|                   | [R50](resnet50.yaml)                                        | 82.8&pm; 1.8 | 54.2&pm; 3.1  | 83.6&pm; 2.0 | 52.5&pm; 3.3 |
| **Vid**     | [X3D-XS](x3d_xs.yaml)                                       | 86.1&pm; 3.0 | 47.8&pm; 6.9  | 82.4&pm; 4.6 | 40.0&pm; 7.9 |
|                   | [MViT](mvit.yaml)                                           | 90.9&pm; 2.2 | 62.9&pm; 3.6  | 87.8&pm; 2.2 | 58.0&pm; 3.5 |
| **IMU+Img** | [KIFNet (A)](kifnet_fusion_average.yaml)                    | 87.5&pm; 2.7 | 55.2&pm; 7.9  | 83.7&pm; 2.1 | 49.3&pm; 6.4 |
|                   | [KIFNet (C)](kifnet_fusion_concat_layernorm.yaml)           | 87.0&pm; 3.0 | 55.5&pm; 6.1  | 83.6&pm; 2.0 | 50.8&pm; 5.8 |
|                   | [KIFNet-Style](kifnet_style_fusion.yaml)                    | 84.7&pm; 2.5 | 51.5&pm; 7.0  | 80.8&pm; 2.8 | 47.8&pm; 6.7 |
|                   | [MNV3-DCL (A)](mobilenetv3_small_dcl_fusion_average.yaml)   | 91.4&pm; 1.7 | 65.1&pm; 6.1  | 88.1&pm; 2.2 | 57.8&pm; 5.6 |
|                   | [MNV3-DCL (C)](mobilenetv3_small_dcl_fusion_concat_ln.yaml) | 91.3&pm; 1.6 | 65.8&pm; 4.0  | 87.9&pm; 2.4 | 59.4&pm; 4.7 |
|                   | [R18-DCL (A)](resnet18_dcl_fusion_average.yaml)             | 92.3&pm; 1.7 | 65.9&pm; 5.3  | 88.5&pm; 2.0 | 58.7&pm; 4.2 |
|                   | [R18-DCL (C)](resnet18_dcl_fusion_concat_ln.yaml)           | 92.1&pm; 1.7 | 66.2&pm; 5.0  | 88.6&pm; 2.1 | 59.3&pm; 5.5 |
| **IMU+Vid** | [SFTIK](sftik_sandwich_loso_lowerbody.yaml)                 | 89.2&pm; 3.6 | 59.8&pm; 5.0  | 86.9&pm; 3.0 | 58.0&pm; 4.6 |
|                   | [EVI-MAE](evi_mae_fusion.yaml)                              | 90.3&pm; 2.0 | 61.8&pm; 4.8  | 88.0&pm; 2.4 | 57.5&pm; 5.8 |
|                   | [X3D-XS-DCL (A)](x3d_xs_dcl_fusion_average.yaml)            | 90.1&pm; 3.8 | 56.2&pm; 7.7  | 83.2&pm; 5.6 | 41.3&pm; 9.0 |
|                   | [X3D-XS-DCL (C)](x3d_xs_dcl_fusion_concat_ln.yaml)          | 91.1&pm; 2.9 | 60.2&pm; 6.9  | 85.7&pm; 4.0 | 47.3&pm; 7.1 |
|                   | [MViT-DCL (A)](mvit_dcl_fusion_average_loso.yaml)           | 92.7&pm; 1.7 | 67.6&pm; 4.6  | 88.8&pm; 2.5 | 61.0&pm; 5.2 |
|                   | [MViT-DCL (C)](mvit_dcl_fusion_concat_ln_loso.yaml)         | 92.8&pm; 2.0 | 68.2&pm; 5.4  | 89.6&pm; 2.3 | 61.9&pm; 4.5 |

> DCL: DeepConvLSTM; MNV3: MobileNetV3; R18: ResNet-18; R50: ResNet-50; A: Feature average; C: Feature concatenation. `-` indicates results not yet available.

## Analysis

Per-class breakdowns, per-subject reports, and summary plots can be regenerated from the training logs with [`tools/analyze_loso_results.py`](../../../tools/analyze_loso_results.py):

```
python3 tools/analyze_loso_results.py \
    --base-dir outputs/loso/revalexo \
    --output-dir analysis_results
```
