---
name: "SegRNN"
summary: "SegRNN is an RNN-based model for long-term multivariate time-series forecasting that replaces the traditional point-wise recurrence with two complementary strategies: Segment-wise Iterations, which process fixed-length segments rather than individual time steps, and Parallel Multi-step Forecasting (PMF), which generates all future steps in a single parallel pass instead of autoregressively. Together these strategies drastically reduce the number of recurrent iterations, cutting runtime and memory by more than 78% compared to standard RNNs while outperforming Transformer-based competitors."
paper: "https://arxiv.org/abs/2308.11200"
paper_title: "SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting"
venue: "arXiv preprint"
year: 2023
code: "https://github.com/lss-1138/SegRNN"
revision: "8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f"
license: "Apache-2.0"
---
# SegRNN

SegRNN is an RNN-based model for long-term multivariate time-series forecasting that replaces the traditional point-wise recurrence with two complementary strategies: Segment-wise Iterations, which process fixed-length segments rather than individual time steps, and Parallel Multi-step Forecasting (PMF), which generates all future steps in a single parallel pass instead of autoregressively. Together these strategies drastically reduce the number of recurrent iterations, cutting runtime and memory by more than 78% compared to standard RNNs while outperforming Transformer-based competitors.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2308.11200); title: SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting; venue/year: arXiv preprint / 2023
- [codebase](https://github.com/lss-1138/SegRNN); revision: `8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SegRNN.toml`](../../../../configs/models/SegRNN.toml).

## Differences

**Paper-driven local implementation.** Histories are last-value centered, split into segments, projected and recurrently encoded by a shared GRU. Relative-horizon and channel identifiers form decoder inputs; each future segment is processed independently with the same encoded state, preserving Parallel Multi-step Forecasting rather than autoregression. Boundary padding is a local runtime extension. The external repository is reference-only and no source file was copied or adapted.

## Shared components

- [`last_value_center`](../_components/last_value_center/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `seg_len=24`
<!-- model-card:canonical:end -->

## Paper
- **Title**: SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-08)
- **arXiv**: https://arxiv.org/abs/2308.11200

## Abstract
RNN-based methods have faced challenges in the Long-term Time Series Forecasting (LTSF) domain when dealing with excessively long look-back windows and forecast horizons. Consequently, the dominance in this domain has shifted towards Transformer, MLP, and CNN approaches. The substantial number of recurrent iterations are the fundamental reasons behind the limitations of RNNs in LTSF. To address these issues, we propose two novel strategies to reduce the number of iterations in RNNs for LTSF tasks: Segment-wise Iterations and Parallel Multi-step Forecasting (PMF). RNNs that combine these strategies, namely SegRNN, significantly reduce the required recurrent iterations for LTSF, resulting in notable improvements in forecast accuracy and inference speed. Extensive experiments demonstrate that SegRNN not only outperforms SOTA Transformer-based models but also reduces runtime and memory usage by more than 78%. These achievements provide strong evidence that RNNs continue to excel in LTSF tasks and encourage further exploration of this domain with more RNN-based approaches.

## In TSFLab
Default config: `configs/models/SegRNN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Paper-driven local implementation.** Histories are last-value centered, split into segments, projected and recurrently encoded by a shared GRU. Relative-horizon and channel identifiers form decoder inputs; each future segment is processed independently with the same encoded state, preserving Parallel Multi-step Forecasting rather than autoregression. Boundary padding is a local runtime extension. The external repository is reference-only and no source file was copied or adapted.

## Citation

```bibtex
@article{DBLP:journals/iotj/LinLWZMZ26,
  author       = {Shengsheng Lin and
                  Weiwei Lin and
                  Wentai Wu and
                  Feiyu Zhao and
                  Ruichao Mo and
                  Haotong Zhang},
  title        = {SegRNN: Segment Recurrent Neural Network for Long-Term Time-Series
                  Forecasting},
  journal      = {{IEEE} Internet Things J.},
  volume       = {13},
  number       = {5},
  pages        = {9861--9871},
  year         = {2026},
  url          = {https://doi.org/10.1109/JIOT.2025.3647705},
  doi          = {10.1109/JIOT.2025.3647705},
  timestamp    = {Wed, 11 Mar 2026 08:24:56 +0100},
  biburl       = {https://dblp.org/rec/journals/iotj/LinLWZMZ26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
