---
name: "Sonnet"
summary: "Sonnet (Spectral Operator Neural Network) is a time series forecasting model for multivariate prediction. It applies learnable wavelet transformations to the input and incorporates spectral analysis using the Koopman operator. The core of its predictive skill is Multivariable Coherence Attention (MVCA), which leverages spectral coherence among variables to model inter-variable dependencies in the frequency domain, avoiding the pitfalls of naive self-attention for time series."
paper: "https://arxiv.org/abs/2505.15312"
paper_title: "Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/ClaudiaShu/Sonnet"
revision: "bf3d4801d34c5e7261718490f287c6fb15cadfdb"
license: "NOASSERTION"
---
# Sonnet

Sonnet (Spectral Operator Neural Network) is a time series forecasting model for multivariate prediction. It applies learnable wavelet transformations to the input and incorporates spectral analysis using the Koopman operator. The core of its predictive skill is Multivariable Coherence Attention (MVCA), which leverages spectral coherence among variables to model inter-variable dependencies in the frequency domain, avoiding the pitfalls of naive self-attention for time series.

<!-- model-card:canonical:start -->
## Method overview

Sonnet (Spectral Operator Neural Network) is a time series forecasting model for multivariate prediction.

## Core architecture

It applies learnable wavelet transformations to the input and incorporates spectral analysis using the Koopman operator. The core of its predictive skill is Multivariable Coherence Attention (MVCA), which leverages spectral coherence among variables to model inter-variable dependencies in the frequency domain, avoiding the pitfalls of naive self-attention for time series.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.15312); title: Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/ClaudiaShu/Sonnet); revision: `bf3d4801d34c5e7261718490f287c6fb15cadfdb`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/Sonnet.toml`](../../../../configs/models/Sonnet.toml).

## Differences

Pinned source inspection: `sonnet/mts_model/models/Sonnet.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This clean-room rewrite maps paper equations (1)--(3) to
`LearnableWavelets` and `SpectralCoherence`. `StableKoopman` constructs the
unitary-eigenbasis operator `U diag(exp(i p)) U*`, and reconstruction multiplies
and sums the evolved states by their wavelet atoms before the three-layer
convolutional decoder. The linked repository is reference-only; its source was
inspected at the pinned revision; no external source code was copied.

The paper separates one endogenous target from exogenous variables using an
alpha-controlled joint embedding. ModernTSF's symmetric multivariate contract
instead embeds all channels jointly and forecasts all of them. The decoder uses
adaptive pooling for arbitrary configured horizons; dataset-specific target
selection, alpha splits, and the original training/evaluation harness remain
outside this local implementation.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=16`, `num_wavelets=4`, `dropout=0.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting
- **Venue**: AAAI 2026 (Oral)
- **Published**: 2026 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.15312

## Abstract
Multivariable time series forecasting methods can integrate information from exogenous variables, leading to significant prediction accuracy gains. The transformer architecture has been widely applied in various time series forecasting models due to its ability to capture long-range sequential dependencies. However, a naïve application of transformers often struggles to effectively model complex relationships among variables over time. To mitigate against this, we propose a novel architecture, termed Spectral Operator Neural Network (Sonnet). Sonnet applies learnable wavelet transformations to the input and incorporates spectral analysis using the Koopman operator. Its predictive skill relies on the Multivariable Coherence Attention (MVCA), an operation that leverages spectral coherence to model variable dependencies. Our empirical analysis shows that Sonnet yields the best performance on 34 out of 47 forecasting tasks with an average mean absolute error (MAE) reduction of 2.2% against the most competitive baseline. We further show that MVCA can remedy the deficiencies of naïve attention in various deep learning models, reducing MAE by 10.7% on average in the most challenging forecasting tasks.

## Source and verification

Pinned source inspection: `sonnet/mts_model/models/Sonnet.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This clean-room rewrite maps paper equations (1)--(3) to
`LearnableWavelets` and `SpectralCoherence`. `StableKoopman` constructs the
unitary-eigenbasis operator `U diag(exp(i p)) U*`, and reconstruction multiplies
and sums the evolved states by their wavelet atoms before the three-layer
convolutional decoder. The linked repository is reference-only; its source was
inspected at the pinned revision; no external source code was copied.

The paper separates one endogenous target from exogenous variables using an
alpha-controlled joint embedding. ModernTSF's symmetric multivariate contract
instead embeds all channels jointly and forecasts all of them. The decoder uses
adaptive pooling for arbitrary configured horizons; dataset-specific target
selection, alpha splits, and the original training/evaluation harness remain
outside this local implementation.

## In ModernTSF
Default config: `configs/models/Sonnet.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ShuL26,
  author       = {Yuxuan Shu and
                  Vasileios Lampos},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Sonnet: Spectral Operator Neural Network for Multivariable Time Series
                  Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25419--25427},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39736},
  doi          = {10.1609/AAAI.V40I30.39736},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ShuL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
