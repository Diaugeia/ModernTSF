---
name: "APN"
summary: "APN (Adaptive Patching Network) is a general and efficient framework for forecasting irregular multivariate time series (IMTS) in a multivariate time-series forecasting setting. It introduces a Time-Aware Patch Aggregation (TAPA) module that learns dynamically adjustable patch boundaries and a time-aware weighted averaging strategy to transform raw irregular observations into high-quality regularized representations, avoiding the need for resampling or interpolation."
paper: "https://arxiv.org/abs/2505.11250"
paper_title: "Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/decisionintelligence/APN"
revision: "f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4"
license: "NOASSERTION"
---
# APN

APN (Adaptive Patching Network) is a general and efficient framework for forecasting irregular multivariate time series (IMTS) in a multivariate time-series forecasting setting. It introduces a Time-Aware Patch Aggregation (TAPA) module that learns dynamically adjustable patch boundaries and a time-aware weighted averaging strategy to transform raw irregular observations into high-quality regularized representations, avoiding the need for resampling or interpolation.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.11250); title: Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/decisionintelligence/APN); revision: `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/APN.toml`](../../../../configs/models/APN.toml).

## Differences

Pinned source inspection: `models/APN.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This is a local implementation derived from paper equations (2)–(10). It
implements learned soft temporal windows, normalized time-aware aggregation,
channel queries, and a query-time MLP decoder. The repository's dense tensor
contract uses regular timestamps unless explicit observation times are passed;
it does not reproduce APN's asynchronous ragged-data loader or missing-value
benchmark protocol. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `d_time=8`, `num_patches=8`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline
- **Venue**: AAAI 2026 (Oral)
- **Published**: 2026 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.11250

## Abstract
The forecasting of irregular multivariate time series (IMTS) is crucial in key areas such as healthcare, biomechanics, climate science, and astronomy. However, achieving accurate and practical predictions is challenging due to two main factors. First, the inherent irregularity and data missingness in irregular time series make modeling difficult. Second, most existing methods are typically complex and resource-intensive. In this study, we propose a general framework called APN to address these challenges. Specifically, we design a novel Time-Aware Patch Aggregation (TAPA) module that achieves adaptive patching. By learning dynamically adjustable patch boundaries and a time-aware weighted averaging strategy, TAPA transforms the original irregular sequences into high-quality, regularized representations in a channel-independent manner. Additionally, we use a simple query module to effectively integrate historical information while maintaining the model's efficiency. Finally, predictions are made by a shallow MLP. Experimental results on multiple real-world datasets show that APN outperforms existing state-of-the-art methods in both efficiency and accuracy.

## Source and verification

Pinned source inspection: `models/APN.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This is a local implementation derived from paper equations (2)–(10). It
implements learned soft temporal windows, normalized time-aware aggregation,
channel queries, and a query-time MLP decoder. The repository's dense tensor
contract uses regular timestamps unless explicit observation times are passed;
it does not reproduce APN's asynchronous ragged-data loader or missing-value
benchmark protocol. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## In TSFLab
Default config: `configs/models/APN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/LiuQWLGHY26,
  author       = {Xvyuan Liu and
                  Xiangfei Qiu and
                  Xingjian Wu and
                  Zhengyu Li and
                  Chenjuan Guo and
                  Jilin Hu and
                  Bin Yang},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Rethinking Irregular Time Series Forecasting: {A} Simple Yet Effective
                  Baseline},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {23873--23881},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i28.39563},
  doi          = {10.1609/AAAI.V40I28.39563},
  timestamp    = {Thu, 26 Mar 2026 16:46:49 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/LiuQWLGHY26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
