---
name: "FeTS"
summary: "FeTS (Feature-Aware Framework for Time Series) is a multivariate time-series forecasting model accepted at AAAI 2026. It learns adaptive temporal importance weightings over input feature-time combinations to selectively emphasize the most informative dimensions, improving forecasting accuracy across standard benchmarks in the standard time-series forecasting setting."
paper: "https://doi.org/10.1609/aaai.v40i31.39838"
paper_title: "FeTS: A Feature-Aware Framework for Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/lllucky111/FeTS"
revision: "d908e434b70f3cf69065004e295db13cdb9790b2"
license: "NOASSERTION"
---
# FeTS

FeTS (Feature-Aware Framework for Time Series) is a multivariate time-series forecasting model accepted at AAAI 2026. It learns adaptive temporal importance weightings over input feature-time combinations to selectively emphasize the most informative dimensions, improving forecasting accuracy across standard benchmarks in the standard time-series forecasting setting.

<!-- model-card:canonical:start -->
## Method overview

FeTS (Feature-Aware Framework for Time Series) is a multivariate time-series forecasting model accepted at AAAI 2026.

## Core architecture

It learns adaptive temporal importance weightings over input feature-time combinations to selectively emphasize the most informative dimensions, improving forecasting accuracy across standard benchmarks in the standard time-series forecasting setting.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1609/aaai.v40i31.39838); title: FeTS: A Feature-Aware Framework for Time Series Forecasting; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/lllucky111/FeTS); revision: `d908e434b70f3cf69065004e295db13cdb9790b2`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/FeTS.toml`](../../../../configs/models/FeTS.toml).

## Differences

Pinned source inspection: `models/FeTS.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. The code is derived from equations
(2)--(14) in the AAAI paper: `FourierPolyMask` implements the Fourier/polynomial
basis and threshold mask, `adaptive_features()` implements mask-controlled local
aggregation, and the local/global branches implement DSFFN. The linked source is
reference-only; its implementation was inspected at the pinned revision; no external source code was copied.

The paper uses a non-differentiable binary threshold. This rewrite preserves the
exact binary forward mask while using a sigmoid straight-through gradient during
training. It uses one compact AdaFE/DSFFN block and does not reproduce the paper's
dataset-specific training schedule or hyperparameter sweep.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=32`, `patch_len=16`, `stride=8`, `fourier_order=2`, `polynomial_order=2`, `kernel_size=3`, `dropout=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: FeTS: A Feature-Aware Framework for Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
FeTS is a feature-aware forecasting framework for multivariate time series that learns adaptive importance weightings over input feature-time combinations. By selectively amplifying the most informative feature dimensions while suppressing irrelevant ones, FeTS improves forecasting accuracy across standard benchmarks. The framework is trained end-to-end and integrates with common backbone architectures, enabling efficient parameter utilization and competitive performance in long-term forecasting settings. (No arXiv preprint was found; this description is based on the AAAI 2026 acceptance and the official implementation repository at https://github.com/lllucky111/FeTS.)

## Source and verification

Pinned source inspection: `models/FeTS.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. The code is derived from equations
(2)--(14) in the AAAI paper: `FourierPolyMask` implements the Fourier/polynomial
basis and threshold mask, `adaptive_features()` implements mask-controlled local
aggregation, and the local/global branches implement DSFFN. The linked source is
reference-only; its implementation was inspected at the pinned revision; no external source code was copied.

The paper uses a non-differentiable binary threshold. This rewrite preserves the
exact binary forward mask while using a sigmoid straight-through gradient during
training. It uses one compact AdaFE/DSFFN block and does not reproduce the paper's
dataset-specific training schedule or hyperparameter sweep.

## In ModernTSF
Default config: `configs/models/FeTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/WangCL26,
  author       = {Le Wang and
                  Jianyong Chen and
                  Songbai Liu},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {FeTS: {A} Feature-Aware Framework for Time Series Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {26328--26336},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i31.39838},
  doi          = {10.1609/AAAI.V40I31.39838},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/WangCL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
