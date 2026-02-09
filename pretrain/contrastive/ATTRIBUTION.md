# Attribution

This directory contains code adapted from **IMU2CLIP**.

## Original Repository

- **Repository**: https://github.com/facebookresearch/imu2clip
- **Paper**: "IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text"
- **Authors**: Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Alireza Dirafzoon, Aparajita Saraf, Amy Bearman, Babak Damavandi
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA 4.0) (see [LICENSE.md](LICENSE.md))

## Nature of Modifications

This codebase is modified from the original IMU2CLIP. The core contrastive learning paradigm (aligning IMU and visual representations via InfoNCE loss) is preserved, but most components have been rewritten or replaced

### What is adapted from the original

- **Contrastive learning framework**: The cross-modal alignment approach using InfoNCE loss with learnable temperature
- **Training paradigm**: Joint encoder training with projection heads to a shared embedding space
- **Evaluation approach**: Retrieval metrics (R@1, R@5, R@10) for cross-modal evaluation

### What is new or replaced

- **IMU encoder**: Replaced the original `AttentionPooledIMUEncoder` with `DeepConvLSTM` (Ordonez & Roggen architecture) to match downstream task models
- **Visual encoder**: Replaced frozen CLIP visual encoder with trainable X3D-XS and MViT-B video encoders from PyTorchVideo
- **Dataset**: Replaced EGO4D dataloader with `MultimodalSensorDataset` for RevalExo's synchronized IMU + egocentric video data
- **Configuration**: Replaced Hydra configs with standalone YAML configuration system
- **Training framework**: Built on PyTorch Lightning with custom callbacks, checkpointing, and logging
- **Data augmentation**: Custom IMU transforms (magnitude warping, time warping, channel dropout) and video transforms
- **Model architecture**: Added `BaseEncoder` abstract class, multi-horizon prediction support, and feature projection pathways
- **Downstream utilities**: Custom `load_pretrained.py` for loading contrastively pretrained encoders into downstream tasks

## Citation

```bibtex
@article{moon2022imu2clip,
  title={IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text},
  author={Moon, Seungwhan and Madotto, Andrea and Lin, Zhaojiang and Dirafzoon, Alireza and Saraf, Aparajita and Bearman, Amy and Damavandi, Babak},
  journal={arXiv preprint arXiv:2210.14395},
  year={2022}
}
```
