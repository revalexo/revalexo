# Attribution

This directory contains code adapted from **IMU-Video-MAE** (ECCV 2024).

## Original Repository

- **Repository**: https://github.com/mf-zhang/IMU-Video-MAE
- **Commit**: `af2a8a4` (release)
- **Paper**: "Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition"
- **Authors**: Mingfang Zhang, Yifei Huang, Ruicong Liu, Yoichi Sato
- **License**: BSD 2-Clause License (see [LICENSE](LICENSE))

## Modifications for RevalExo

### New files (not in original repository)

- **`migrate_revalexo_data.py`** - Converts RevalExo's 7-sensor lower-body IMU data to EVI-MAE's 4-body-part format (12 channels). Maps Left_Foot, Right_Foot, Left_Upper_Leg, and Right_Upper_Leg to the model's expected left_arm, right_arm, left_leg, right_leg slots.
- **`egs/release/pretrain_revalexo.sh`** - Pretraining launch script configured for RevalExo dataset parameters (12 IMU channels, auto-loaded statistics).

### Modified files

- **`src/dataloader.py`**
  - Added `'revalexo'` and `'aidwear'` to the dataset name detection condition (line 135) so that RevalExo data paths are recognized as WEAR-format datasets.
  - Added conditional IMU channel selection (lines 247-252): when the data path contains `'revalexo'`, uses `list(range(12))` to read all 12 pre-mapped channels directly, instead of the original WEAR index remapping.

- **`src/run_evimae_pretrain.py`**
  - Added shape-mismatch checking during checkpoint loading (lines 185-199). When loading pretrained weights, layers with mismatched shapes (due to different IMU channel configurations) are skipped instead of causing a RuntimeError. Skipped keys are logged for transparency.

### Unmodified files (copied as-is from original)

- `src/traintest_evimae.py` - Training/evaluation loop
- `src/transforms.py` - Video data augmentation transforms
- `src/models/` - All model architecture files (evi_mae.py, gin.py, gat.py, etc.)
- `src/utilities/` - Utility functions (stats.py, util.py)

## Citation

```bibtex
@inproceedings{zhang2024masked,
  title={Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition},
  author={Zhang, Mingfang and Huang, Yifei and Liu, Ruicong and Sato, Yoichi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
