#  Detecting Anomalies in Industrial Machine Sounds


## Project Overview

Industrial machines generate distinct acoustic patterns during normal operation. When a fault occurs, these patterns change subtly. Manual monitoring is inefficient and cannot provide continuous supervision at scale.

This project develops a robust machine learning model for **unsupervised anomaly detection** in industrial machine sounds, with transfer learning to improve accuracy under domain shift conditions.

---

## Datasets

| Dataset | Purpose | URL |
|---------|---------|-----|
| **MIMII 2019** | Detect abnormal sounds when only normal sounds are available for training | [zenodo.org/records/3384388](https://zenodo.org/records/3384388) |
| **MIMII DUE 2021** | Evaluate performance when training and testing conditions differ | [zenodo.org/records/4740355](https://zenodo.org/records/4740355) |


---

## ML Pipeline

<!-- Save your pipeline image to the repo as 'pipeline.png' and it will display here -->
![ML Pipeline](pipeline.png)

---

## Methodology

### Stage 1 — Baseline Learning (MIMII 2019)

- Train an autoencoder **exclusively on normal sounds** from MIMII 2019 (Fan, 6 dB, id\_00)
- At test time: normal sounds → **low reconstruction error**, anomalous sounds → **high reconstruction error**
- No abnormal sounds used in training → reflects real-world conditions where fault data is rare
- **Expected AUC:** ~0.80 – 0.95

### Stage 2 — Domain Shift Testing (MIMII DUE 2021)

- Train a **fresh autoencoder** (randomly initialised) on MIMII DUE 2021 source + target domain normals
- Evaluate on target domain (different factory environment, different conditions)
- **Expected result:** AUC drops significantly — proving domain shift is a real, measurable problem

### Stage 3 — Transfer Learning / Fine-Tuning

- Load the **pretrained Stage 1 model** (already knows what normal sounds like)
- Fine-tune with a **lower learning rate** (1e-4) on 2021 source-domain normals
- Evaluate on target domain
- **Expected result:** AUC recovers — transfer learning bridges the domain gap

---

## Results

| Stage | Description | Expected AUC |
|-------|-------------|-------------|
| **Stage 1** | MIMII 2019 Baseline | High (~0.80 – 0.95) |
| **Stage 2** | MIMII DUE 2021 — Fresh model (Domain Shift) | Low (~0.47 – 0.52) |
| **Stage 3** | MIMII DUE 2021 — Fine-tuned (Transfer Learning) | 0.65 |

---

## References

1. Purohit, H. et al. (2019). *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection.* DCASE Workshop. [arXiv:1909.09347](https://arxiv.org/abs/1909.09347)

2. Tanabe, R. et al. (2021). *MIMII DUE: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection with Domain Shifts.* [Zenodo Record](https://zenodo.org/records/4740355)

3. MIMII Baseline Code: [github.com/MIMII-hitachi/mimii_baseline](https://github.com/MIMII-hitachi/mimii_baseline)
