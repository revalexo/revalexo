# RevalExo

Official codebase for *"RevalExo: A Functional Daily-Activity Benchmark for Inertial and Visual Locomotion Mode Recognition in Older Adults and Clinical Cohorts"*.

[[Project Page]](https://revalexo.github.io/) | [[Dataset]](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/OWJOID)

## Repository Structure

| Path | Description |
| :--- | :---------- |
| [`pretrain/`](pretrain/) | Self-supervised pretraining pipelines (EVI-MAE masked autoencoder, contrastive IMU-video alignment) |
| [`train/`](train/) | Training, evaluation, and benchmarking for locomotion mode recognition (LOSO, cross-population, vision-guided transfer) |


<br>
<hr>


  <p>If you use this dataset or code, please cite our paper:</p>
  <pre tabindex="0" aria-label="BibTeX citation"
           style="white-space: pre; overflow-wrap: normal; overflow-x: auto;"><code class="language-bibtex">@misc{lamsal2026revalexo,
  title={RevalExo: A Functional Daily-Activity Benchmark for Inertial and Visual Locomotion Mode Recognition in Older Adults and Clinical Cohorts},
  author={Diwas Lamsal and Juha Carlon and Reinhard Claeys and Maxim Yudayev and Louis Flynn and Tom Verstraten and David Beckwée and Eva Swinnen and Mihai Bâce and Bart Vanrumste and Benjamin Filtjens},
  year={2026},
  eprint={2609.08090},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.08090},
}</code></pre>
