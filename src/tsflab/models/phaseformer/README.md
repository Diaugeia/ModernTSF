---
name: "PhaseFormer"
summary: "PhaseFormer is an efficient time series forecasting model for standard univariate and multivariate prediction. It introduces a phase perspective for exploiting periodicity: instead of treating individual patches as tokens (which incurs large parameter counts), PhaseFormer groups time steps into compact phase embeddings aligned to the dominant period and uses a lightweight routing mechanism for cross-phase interaction, achieving state-of-the-art performance with approximately 1k parameters across benchmark datasets."
paper: "https://arxiv.org/abs/2510.04134"
paper_title: "PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/neumyor/PhaseFormer_TSL"
revision: "ed1db61c6abfa9326d5ca2a56c6c4ba53ea592ab"
license: "MIT"
---
# PhaseFormer

PhaseFormer is an efficient time series forecasting model for standard univariate and multivariate prediction. It introduces a phase perspective for exploiting periodicity: instead of treating individual patches as tokens (which incurs large parameter counts), PhaseFormer groups time steps into compact phase embeddings aligned to the dominant period and uses a lightweight routing mechanism for cross-phase interaction, achieving state-of-the-art performance with approximately 1k parameters across benchmark datasets.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.04134); title: PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/neumyor/PhaseFormer_TSL); revision: `ed1db61c6abfa9326d5ca2a56c6c4ba53ea592ab`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PhaseFormer.toml`](../../../../configs/models/PhaseFormer.toml).

## Differences

Pinned source inspection: `models/PhaseFormer.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This clean-room rewrite follows paper equations (5)--(11): circular phase
tokenization produces `[phase, period-index]` tokens, `CrossPhaseRouter` performs
phase-to-router aggregation and router-to-phase distribution, and one shared
linear predictor maps every phase to future periods before de-tokenization. The
linked repository is reference-only; its source was inspected at the pinned revision; no external source code was copied.

The paper estimates the dominant period by autocorrelation, whereas this
standalone runtime receives `period` as an explicit configuration value. The
default uses one routing layer and channel-independent processing; it does not
claim the paper's exact training recipe, learned period-selection pipeline, or
reported approximately-1k-parameter setting for every dataset.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=16`, `dropout=0.0`, `period=24`, `num_routers=4`, `num_layers=1`, `num_heads=1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.04134

## Abstract
Periodicity is a fundamental characteristic of time series data and has long played a central role in forecasting. Recent deep learning methods strengthen the exploitation of periodicity by treating patches as basic tokens, thereby improving predictive effectiveness. However, their efficiency remains a bottleneck due to large parameter counts and heavy computational costs. This paper provides, for the first time, a clear explanation of why patch-level processing is inherently inefficient, supported by strong evidence from real-world data. To address these limitations, we introduce a phase perspective for modeling periodicity and present an efficient yet effective solution, PhaseFormer. PhaseFormer features phase-wise prediction through compact phase embeddings and efficient cross-phase interaction enabled by a lightweight routing mechanism. Extensive experiments demonstrate that PhaseFormer achieves state-of-the-art performance with around 1k parameters, consistently across benchmark datasets. Notably, it excels on large-scale and complex datasets, where models with comparable efficiency often struggle. This work marks a significant step toward truly efficient and effective time series forecasting.

## Source and verification

Pinned source inspection: `models/PhaseFormer.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This clean-room rewrite follows paper equations (5)--(11): circular phase
tokenization produces `[phase, period-index]` tokens, `CrossPhaseRouter` performs
phase-to-router aggregation and router-to-phase distribution, and one shared
linear predictor maps every phase to future periods before de-tokenization. The
linked repository is reference-only; its source was inspected at the pinned revision; no external source code was copied.

The paper estimates the dominant period by autocorrelation, whereas this
standalone runtime receives `period` as an explicit configuration value. The
default uses one routing layer and channel-independent processing; it does not
claim the paper's exact training recipe, learned period-selection pipeline, or
reported approximately-1k-parameter setting for every dataset.

## In TSFLab
Default config: `configs/models/PhaseFormer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@misc{niu2025phaseformer,
  author        = {Yiming Niu and
                  Jinliang Deng and
                  Yongxin Tong},
  title         = {PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting},
  year          = {2025},
  eprint        = {2510.04134},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.04134}
}
```
