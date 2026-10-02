---
name: "CoRA"
summary: "CoRA is a time series forecasting model that acts as a lightweight, plug-and-play correlation-aware adapter for multivariate forecasting. It augments time series foundation models (which typically use channel-independent modeling) by explicitly capturing three types of inter-channel correlations: time-varying dynamic correlations (via learnable polynomials), heterogeneous correlations (positive and negative), and partial correlations among subsets of channels (via a dual contrastive learning approach). The adapter requires only fine-tuning with the base foundation model and adds no extra complexity at inference time."
paper: "https://arxiv.org/abs/2603.21828"
paper_title: "CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/decisionintelligence/CoRA"
revision: "1292f7b114e26d477675291004acb018e81896ca"
license: "NOASSERTION"
---
# CoRA

CoRA is a time series forecasting model that acts as a lightweight, plug-and-play correlation-aware adapter for multivariate forecasting. It augments time series foundation models (which typically use channel-independent modeling) by explicitly capturing three types of inter-channel correlations: time-varying dynamic correlations (via learnable polynomials), heterogeneous correlations (positive and negative), and partial correlations among subsets of channels (via a dual contrastive learning approach). The adapter requires only fine-tuning with the base foundation model and adds no extra complexity at inference time.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2603.21828); title: CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/decisionintelligence/CoRA); revision: `1292f7b114e26d477675291004acb018e81896ca`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CoRA.toml`](../../../../configs/models/CoRA.toml).

## Differences

Pinned source inspection: `ts_benchmark/plugin/plugin.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This clean-room runtime implements the paper's polynomial time-varying factor,
time-invariant low-rank composition, Pearson prior, positive/negative channel
projections, fusion, and gated residual forecast. It deliberately substitutes a
small local linear forecaster for a pre-trained TSFM. The training-only H-PCorr
dual contrastive objective and foundation-model fine-tuning protocol are not
implemented. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `rank=4`, `polynomial_order=2`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2026-03)
- **arXiv**: https://arxiv.org/abs/2603.21828

## Abstract
Most existing Time Series Foundation Models (TSFMs) use channel independent modeling and focus on capturing and generalizing temporal dependencies, while neglecting the correlations among channels or overlooking the different aspects of correlations. However, these correlations play a vital role in Multivariate time series forecasting. To address this, we propose a CoRrelation-aware Adapter (CoRA), a lightweight plug-and-play method that requires only fine-tuning with TSFMs and is able to capture different types of correlations, so as to improve forecast performance. Specifically, to reduce complexity, we innovatively decompose the correlation matrix into low-rank Time-Varying and Time-Invariant components. For the Time-Varying component, we further design learnable polynomials to learn dynamic correlations by capturing trends or periodic patterns. To learn positive and negative correlations that appear only among some channels, we introduce a novel dual contrastive learning method that identifies correlations through projection layers, regulated by a Heterogeneous-Partial contrastive loss during training, without introducing additional complexity in the inference stage. Extensive experiments on 10 real-world datasets demonstrate that CoRA can improve TSFMs in multivariate forecasting performance.

## Source and verification

Pinned source inspection: `ts_benchmark/plugin/plugin.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This clean-room runtime implements the paper's polynomial time-varying factor,
time-invariant low-rank composition, Pearson prior, positive/negative channel
projections, fusion, and gated residual forecast. It deliberately substitutes a
small local linear forecaster for a pre-trained TSFM. The training-only H-PCorr
dual contrastive objective and foundation-model fine-tuning protocol are not
implemented. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## In ModernTSF
Default config: `configs/models/CoRA.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2603-21828,
  author       = {Hanyin Cheng and
                  Xingjian Wu and
                  Yang Shu and
                  Zhongwen Rao and
                  Lujia Pan and
                  Bin Yang and
                  Chenjuan Guo},
  title        = {CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting
                  through Correlation-aware Adapter},
  journal      = {CoRR},
  volume       = {abs/2603.21828},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2603.21828},
  doi          = {10.48550/ARXIV.2603.21828},
  eprinttype   = {arXiv},
  eprint       = {2603.21828},
  timestamp    = {Wed, 15 Apr 2026 11:01:56 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2603-21828.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
