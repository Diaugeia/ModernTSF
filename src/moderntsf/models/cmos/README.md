---
name: "CMoS"
summary: "CMoS is a super-lightweight multivariate time series forecasting model for the standard time-series setting. Rather than learning shape embeddings, it directly models spatial correlations between different time-series chunks using a Correlation Mixing strategy that captures diverse channel dependencies with minimal parameters, and an optional Periodicity Injection technique for faster convergence — achieving competitive accuracy at up to 100x the parameter efficiency of DLinear."
paper: "https://arxiv.org/abs/2505.19090"
paper_title: "CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations"
venue: "ICML 2025"
year: 2025
code: "https://github.com/CSTCloudOps/CMoS"
revision: "b696a0c33b5ad8f03ad483d43b95fcb5564aa939"
license: "NOASSERTION"
---
# CMoS

CMoS is a super-lightweight multivariate time series forecasting model for the standard time-series setting. Rather than learning shape embeddings, it directly models spatial correlations between different time-series chunks using a Correlation Mixing strategy that captures diverse channel dependencies with minimal parameters, and an optional Periodicity Injection technique for faster convergence — achieving competitive accuracy at up to 100x the parameter efficiency of DLinear.

<!-- model-card:canonical:start -->
## Method overview

CMoS is a super-lightweight multivariate time series forecasting model for the standard time-series setting.

## Core architecture

Rather than learning shape embeddings, it directly models spatial correlations between different time-series chunks using a Correlation Mixing strategy that captures diverse channel dependencies with minimal parameters, and an optional Periodicity Injection technique for faster convergence — achieving competitive accuracy at up to 100x the parameter efficiency of DLinear.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.19090); title: CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations; venue/year: ICML 2025 / 2025
- [codebase](https://github.com/CSTCloudOps/CMoS); revision: `b696a0c33b5ad8f03ad483d43b95fcb5564aa939`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/CMoS.toml`](../../../../configs/models/CMoS.toml).

## Differences

Pinned source inspection: `model/CMoS/Model.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

**Local implementation: confirmed.** The linked repository has no explicit
license and is `reference-only`; its source was inspected at the pinned revision; no external source code was copied. The
local design maps paper Eqs. 3--5 to K shared chunk-correlation matrices,
channel-specific convolutional summaries, and a shared softmax allocator. The
optional `period` setting initializes the first matrix with Section 3.3's
periodic peaks; it is disabled unless the dataset justifies that prior. The
previous non-paper top-k router was removed. Official initialization details,
dataset recipes, and numerical reference comparison are not claimed.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `seg_size=4`, `num_map=3`, `kernel_size=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.19090

## Abstract
Recent advances in lightweight time series forecasting models suggest the inherent simplicity of time series forecasting tasks. In this paper, we present CMoS, a super-lightweight time series forecasting model. Instead of learning the embedding of the shapes, CMoS directly models the spatial correlations between different time series chunks. Additionally, we introduce a Correlation Mixing technique that enables the model to capture diverse spatial correlations with minimal parameters, and an optional Periodicity Injection technique to ensure faster convergence. Despite utilizing as low as 1% of the lightweight model DLinear's parameters count, experimental results demonstrate that CMoS outperforms existing state-of-the-art models across multiple datasets. Furthermore, the learned weights of CMoS exhibit great interpretability, providing practitioners with valuable insights into temporal structures within specific application scenarios.

## In ModernTSF
Default config: `configs/models/CMoS.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Source and verification

Pinned source inspection: `model/CMoS/Model.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

**Local implementation: confirmed.** The linked repository has no explicit
license and is `reference-only`; its source was inspected at the pinned revision; no external source code was copied. The
local design maps paper Eqs. 3--5 to K shared chunk-correlation matrices,
channel-specific convolutional summaries, and a shared softmax allocator. The
optional `period` setting initializes the first matrix with Section 3.3's
periodic peaks; it is disabled unless the dataset justifies that prior. The
previous non-paper top-k router was removed. Official initialization details,
dataset recipes, and numerical reference comparison are not claimed.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/SiPLPX25,
  author       = {Haotian Si and
                  Changhua Pei and
                  Jianhui Li and
                  Dan Pei and
                  Gaogang Xie},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise
                  Spatial Correlations},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/si25a.html},
  timestamp    = {Wed, 04 Feb 2026 17:22:46 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/SiPLPX25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
