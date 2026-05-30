# RevalExo

Official codebase for *"RevalExo: A Functional Daily-Activity Benchmark for Inertial and Visual Locomotion Mode Recognition in Older Adults and Clinical Cohorts"*.

[[Project Page]](https://revalexo.github.io/) | [[Dataset]](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/OWJOID)

## Repository Structure

| Path | Description |
| :--- | :---------- |
| [`pretrain/`](pretrain/) | Self-supervised pretraining pipelines (EVI-MAE masked autoencoder, contrastive IMU-video alignment) |
| [`train/`](train/) | Training, evaluation, and benchmarking for locomotion mode recognition (LOSO, cross-population, vision-guided transfer) |
