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
- **`src/run_evimae_pretrain.py`**

## Citation

```bibtex
@inproceedings{zhang2024masked,
  title={Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition},
  author={Zhang, Mingfang and Huang, Yifei and Liu, Ruicong and Sato, Yoichi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
