---
name: "HMformer"
summary: "HMformer is a Transformer-based multivariate time-series forecasting model that proposes a hierarchical multi-scale framework to overcome the limitations of the original Transformer architecture when applied to real-world time series with complex multi-scale periodicities. It employs a hierarchical cross-scale mixing mechanism, a scale-adaptive feature expansion design, and a multi-branch complementary prediction strategy to capture intricate multi-scale temporal dynamics while retaining the Transformer's strength in modeling long-range dependencies."
paper: "https://ojs.aaai.org/index.php/AAAI/article/view/39355"
paper_title: "HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/dantian123121/HMformer"
revision: "18e014934a7c7010c16822689f5332dd1ebd49a6"
license: "NOASSERTION"
---
# HMformer

HMformer is a Transformer-based multivariate time-series forecasting model that proposes a hierarchical multi-scale framework to overcome the limitations of the original Transformer architecture when applied to real-world time series with complex multi-scale periodicities. It employs a hierarchical cross-scale mixing mechanism, a scale-adaptive feature expansion design, and a multi-branch complementary prediction strategy to capture intricate multi-scale temporal dynamics while retaining the Transformer's strength in modeling long-range dependencies.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://ojs.aaai.org/index.php/AAAI/article/view/39355); title: HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/dantian123121/HMformer); revision: `18e014934a7c7010c16822689f5332dd1ebd49a6`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/HMformer.toml`](../../../../configs/models/HMformer.toml).

## Differences

Pinned source inspection: `Code/Long-term_Forecasting/models/HMformer.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This implementation was independently derived from the AAAI paper. It retains
channel-independent overlapping patch branches, SAFE's doubling of latent
width from fine to coarse scales, rotary self-attention, stride-two cross-scale
convolutions, and the sum of flattened branch forecasts in Eqs. (2)--(5). The
default preset is compact (`K=3`, one block per branch) rather than the complete
paper sweep, and branches that cannot fit the configured history are omitted.
Training recipes and reported weights are not reproduced. The reference-only
repository was inspected at the pinned revision; no external source code was copied. Strict evidence is in
`../../../verification/evidence/HMformer.json`.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `patch_len=16`, `stride=8`, `num_scales=3`, `depth=1`, `num_heads=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: HMformer: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling
- **Venue**: AAAI 2026
- **Published**: 2026
- **Proceedings**: https://ojs.aaai.org/index.php/AAAI/article/view/39355

## Abstract
Time series forecasting plays a critical role across a wide range of domains. Recently, an increasing number of Transformer-based forecasting models have emerged, achieving remarkably competitive performance. However, real-world time series data often exhibit complex multi-scale periodicities, which are not well-suited for modeling by the original Transformer architecture originally developed for NLP tasks. To address this limitation, we propose the Hierarchical Multi-scale Time Series Transformer (HMformer), employing a novel and sophisticated framework specifically designed for multi-scale time series forecasting. Specifically, HMformer incorporates a hierarchical cross-scale mixing mechanism that progressively aggregates temporal information from fine to coarse granularities, a scale-adaptive feature expansion design enhancing the extraction of high-level temporal semantics, and a multi-branch complementary prediction strategy for effectively integrating diverse temporal patterns. Collectively, these components enable HMformer to capture intricate, multi-scale temporal dynamics while retaining the Transformer's inherent strength in modeling long-range dependencies. Extensive experiments conducted on multiple real-world benchmark datasets—encompassing both long-term and short-term forecasting tasks—demonstrate that HMformer achieves state-of-the-art performance.

## Source and verification

Pinned source inspection: `Code/Long-term_Forecasting/models/HMformer.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This implementation was independently derived from the AAAI paper. It retains
channel-independent overlapping patch branches, SAFE's doubling of latent
width from fine to coarse scales, rotary self-attention, stride-two cross-scale
convolutions, and the sum of flattened branch forecasts in Eqs. (2)--(5). The
default preset is compact (`K=3`, one block per branch) rather than the complete
paper sweep, and branches that cannot fit the configured history are omitted.
Training recipes and reported weights are not reproduced. The reference-only
repository was inspected at the pinned revision; no external source code was copied. Strict evidence is in
`../../../verification/evidence/HMformer.json`.

## In TSFLab
Default config: `configs/models/HMformer.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@inproceedings{huang2026hmformer,
  author    = {Renjun Huang and Han Xiao and Bingqing Li and Baili Zhang and Jianhua Lyu},
  title     = {{HMformer}: Unleashing Transformer's Potential for Time Series Forecasting via Hierarchical Multi-Scale Modeling},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  url       = {https://doi.org/10.1609/aaai.v40i26.39355},
  doi       = {10.1609/aaai.v40i26.39355}
}
```
