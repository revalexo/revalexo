# RevalExo

Official codebase for *"RevalExo: A Multimodal Dataset and Benchmark for Locomotion Mode Recognition Across Healthy and Clinical Populations"*.

[[Project Page]](https://revalexo.github.io/) | [[Dataset]](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/OWJOID)

## Repository Structure

| Path | Description |
| :--- | :---------- |
| [`pretrain/`](pretrain/) | Self-supervised pretraining pipelines (EVI-MAE masked autoencoder, contrastive IMU-video alignment) |
| [`train/`](train/) | Training, evaluation, and benchmarking for locomotion mode recognition (LOSO, cross-population, vision-guided transfer) |
