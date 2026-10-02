---
name: "TimeKAN"
summary: "TimeKAN is a time series forecasting model that combines Kolmogorov-Arnold Networks (KANs) with multi-scale frequency decomposition. It decomposes a mixed-frequency input series into individual frequency bands via Cascaded Frequency Decomposition (CFD) blocks, learns band-specific temporal patterns with Multi-order KAN Representation Learning (M-KAN) blocks that exploit the flexibility of KANs, and recombines the bands via Frequency Mixing blocks to produce accurate multi-horizon predictions. The architecture achieves state-of-the-art results while remaining extremely lightweight."
paper: "https://arxiv.org/abs/2502.06910"
paper_title: "TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting"
venue: "arXiv preprint"
year: 2025
code: "https://github.com/huangst21/TimeKAN"
revision: "3a7c366a9e8547fd8840c5d27f25ee3e30615e33"
license: "Apache-2.0"
---
# TimeKAN

TimeKAN is a time series forecasting model that combines Kolmogorov-Arnold Networks (KANs) with multi-scale frequency decomposition. It decomposes a mixed-frequency input series into individual frequency bands via Cascaded Frequency Decomposition (CFD) blocks, learns band-specific temporal patterns with Multi-order KAN Representation Learning (M-KAN) blocks that exploit the flexibility of KANs, and recombines the bands via Frequency Mixing blocks to produce accurate multi-horizon predictions. The architecture achieves state-of-the-art results while remaining extremely lightweight.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2502.06910); title: TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting; venue/year: arXiv preprint / 2025
- [codebase](https://github.com/huangst21/TimeKAN); revision: `3a7c366a9e8547fd8840c5d27f25ee3e30615e33`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeKAN.toml`](../../../../configs/models/TimeKAN.toml).

## Differences

**Paper-driven local implementation.** The implementation follows Equations
(3)–(12): FFT-padding frequency upsampling, cascaded residual bands,
frequency-specific Chebyshev polynomial orders, depthwise temporal convolution,
and bottom-up frequency mixing. Timestamp covariates are not part of the paper's
active path. The external repository is reference-only; no source file was
copied or adapted.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=16`, `e_layers=1`, `down_sampling_window=2`, `down_sampling_layers=1`, `begin_order=0`, `moving_avg=25`, `dropout=0.1`, `embed='timeF'`, `freq='h'`, `use_norm=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-02)
- **arXiv**: https://arxiv.org/abs/2502.06910

## Abstract
Real-world time series often have multiple frequency components that are intertwined with each other, making accurate time series forecasting challenging. Decomposing the mixed frequency components into multiple single frequency components is a natural choice. However, the information density of patterns varies across different frequencies, and employing a uniform modeling approach for different frequency components can lead to inaccurate characterization. To address this challenges, inspired by the flexibility of the recent Kolmogorov-Arnold Network (KAN), we propose a KAN-based Frequency Decomposition Learning architecture (TimeKAN) to address the complex forecasting challenges caused by multiple frequency mixtures. Specifically, TimeKAN mainly consists of three components: Cascaded Frequency Decomposition (CFD) blocks, Multi-order KAN Representation Learning (M-KAN) blocks and Frequency Mixing blocks. CFD blocks adopt a bottom-up cascading approach to obtain series representations for each frequency band. Benefiting from the high flexibility of KAN, we design a novel M-KAN block to learn and represent specific temporal patterns within each frequency band. Finally, Frequency Mixing blocks is used to recombine the frequency bands into the original format. Extensive experimental results across multiple real-world time series datasets demonstrate that TimeKAN achieves state-of-the-art performance as an extremely lightweight architecture.

## In TSFLab
Default config: `configs/models/TimeKAN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Paper-driven local implementation.** The implementation follows Equations
(3)–(12): FFT-padding frequency upsampling, cascaded residual bands,
frequency-specific Chebyshev polynomial orders, depthwise temporal convolution,
and bottom-up frequency mixing. Timestamp covariates are not part of the paper's
active path. The external repository is reference-only; no source file was
copied or adapted.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/HuangZL025,
  author       = {Songtao Huang and
                  Zhen Zhao and
                  Can Li and
                  Lei Bai},
  title        = {TimeKAN: KAN-based Frequency Decomposition Learning Architecture for
                  Long-term Time Series Forecasting},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025, Singapore, April 24-28, 2025},
  publisher    = {OpenReview.net},
  year         = {2025},
  url          = {https://openreview.net/forum?id=wTLc79YNbh},
  timestamp    = {Sun, 02 Nov 2025 10:11:43 +0100},
  biburl       = {https://dblp.org/rec/conf/iclr/HuangZL025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
