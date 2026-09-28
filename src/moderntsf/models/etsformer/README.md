---
name: "ETSformer"
summary: "ETSformer is a time series forecasting model that combines classical exponential smoothing principles with the Transformer architecture to address limitations of vanilla Transformers for long-term forecasting. It introduces two novel attention mechanisms—exponential smoothing attention (ESA) and frequency attention (FA)—to replace standard self-attention, and redesigns the Transformer with modular decomposition blocks that learn to separate time series into interpretable components: level, growth, and seasonality."
paper: "https://arxiv.org/abs/2202.01381"
paper_title: "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting"
venue: "arXiv preprint"
year: 2022
code: "https://github.com/thuml/Time-Series-Library"
revision: "230805fe9f451b61e34b96116d995b417e343ac0"
license: "MIT"
---
# ETSformer

ETSformer is a time series forecasting model that combines classical exponential smoothing principles with the Transformer architecture to address limitations of vanilla Transformers for long-term forecasting. It introduces two novel attention mechanisms—exponential smoothing attention (ESA) and frequency attention (FA)—to replace standard self-attention, and redesigns the Transformer with modular decomposition blocks that learn to separate time series into interpretable components: level, growth, and seasonality.

<!-- model-card:canonical:start -->
## Method overview

ETSformer is a time series forecasting model that combines classical exponential smoothing principles with the Transformer architecture to address limitations of vanilla Transformers for long-term forecasting.

## Core architecture

It introduces two novel attention mechanisms—exponential smoothing attention (ESA) and frequency attention (FA)—to replace standard self-attention, and redesigns the Transformer with modular decomposition blocks that learn to separate time series into interpretable components: level, growth, and seasonality.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2202.01381); title: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting; venue/year: arXiv preprint / 2022
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `230805fe9f451b61e34b96116d995b417e343ac0`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ETSformer.toml`](../../../../configs/models/ETSformer.toml).

## Differences

**Paper-driven local implementation.** The model implements the paper's
exponential-smoothing recurrence, top-amplitude Fourier seasonality extraction,
residual level/growth/seasonality stacks, growth damping, and additive decoder.
The paper explicitly avoids calendar covariates, so timestamp marks are accepted
by the common forecast signature but not consumed. The external repository is
reference-only; no source file was copied or adapted. Published benchmark
reproduction remains outside independent code validation.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_layers=2`, `d_ff=256`, `top_k=3`, `dropout=0.1`, `activation='sigmoid'`, `embed='timeF'`, `freq='h'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2022
- **arXiv**: https://arxiv.org/abs/2202.01381

## Abstract
Transformers have been actively studied for time-series forecasting in recent years. While often showing promising results in various scenarios, traditional Transformers are not designed to fully exploit the characteristics of time-series data and thus suffer some fundamental limitations, e.g., they generally lack of decomposition capability and interpretability, and are neither effective nor efficient for long-term forecasting. In this paper, we propose ETSFormer, a novel time-series Transformer architecture, which exploits the principle of exponential smoothing in improving Transformers for time-series forecasting. In particular, inspired by the classical exponential smoothing methods in time-series forecasting, we propose the novel exponential smoothing attention (ESA) and frequency attention (FA) to replace the self-attention mechanism in vanilla Transformers, thus improving both accuracy and efficiency. Based on these, we redesign the Transformer architecture with modular decomposition blocks such that it can learn to decompose the time-series data into interpretable time-series components such as level, growth and seasonality. Extensive experiments on various time-series benchmarks validate the efficacy and advantages of the proposed method. Code is available at this https URL.

## In ModernTSF
Default config: `configs/models/ETSformer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Paper-driven local implementation.** The model implements the paper's
exponential-smoothing recurrence, top-amplitude Fourier seasonality extraction,
residual level/growth/seasonality stacks, growth damping, and additive decoder.
The paper explicitly avoids calendar covariates, so timestamp marks are accepted
by the common forecast signature but not consumed. The external repository is
reference-only; no source file was copied or adapted. Published benchmark
reproduction remains outside independent code validation.

## Citation

```bibtex
@misc{woo2022etsformer,
  author        = {Gerald Woo and
                  Chenghao Liu and
                  Doyen Sahoo and
                  Akshat Kumar and
                  Steven Hoi},
  title         = {ETSformer: Exponential Smoothing Transformers for Time-series Forecasting},
  year          = {2022},
  eprint        = {2202.01381},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2202.01381}
}
```
