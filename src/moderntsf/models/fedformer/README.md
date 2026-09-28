---
name: "FEDformer"
summary: "FEDformer is a Transformer-based model for long-term multivariate and univariate time-series forecasting that combines seasonal-trend decomposition with a frequency-enhanced attention mechanism. The decomposition component captures the global profile of the series while Transformer blocks model finer-grained structure; exploiting the sparse Fourier representation of most time series yields linear complexity in sequence length, making FEDformer more efficient than standard Transformers."
paper: "https://proceedings.mlr.press/v162/zhou22g.html"
paper_title: "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
venue: "ICML 2022"
year: 2022
code: "https://github.com/MAZiqing/FEDformer"
revision: "c0f6b972def125691434d62be1ecadf710ae921a"
license: "MIT"
---
# FEDformer

FEDformer is a Transformer-based model for long-term multivariate and univariate time-series forecasting that combines seasonal-trend decomposition with a frequency-enhanced attention mechanism. The decomposition component captures the global profile of the series while Transformer blocks model finer-grained structure; exploiting the sparse Fourier representation of most time series yields linear complexity in sequence length, making FEDformer more efficient than standard Transformers.

<!-- model-card:canonical:start -->
## Method overview

FEDformer is a Transformer-based model for long-term multivariate and univariate time-series forecasting that combines seasonal-trend decomposition with a frequency-enhanced attention mechanism.

## Core architecture

The decomposition component captures the global profile of the series while Transformer blocks model finer-grained structure; exploiting the sparse Fourier representation of most time series yields linear complexity in sequence length, making FEDformer more efficient than standard Transformers.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.mlr.press/v162/zhou22g.html); title: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting; venue/year: ICML 2022 / 2022
- [codebase](https://github.com/MAZiqing/FEDformer); revision: `c0f6b972def125691434d62be1ecadf710ae921a`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/FEDformer.toml`](../../../../configs/models/FEDformer.toml).

## Differences

**Clean-room implementation: confirmed.** `FrequencyEnhancedBlock` maps paper
Eqs. (3)-(4), `FrequencyEnhancedAttention` maps Eqs. (6)-(7), and decoder
decomposition accumulates three trend updates per layer. Inputs are
`[B, seq_len, enc_in]` with optional six-column marks; outputs are
`[B, pred_len, c_out]`. Only Fourier mode is implemented, using head-local
complex kernels and deterministic random mode sets; wavelet, checkpoint, and
published-metric reference comparison are not claimed.

## Shared components

- [`forecast_embedding`](../_components/forecast_embedding/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `dec_in=7`, `c_out=7`, `d_model=512`, `n_heads=8`, `e_layers=2`, `d_layers=1`, `d_ff=2048`, `moving_avg=25`, `dropout=0.1`, `activation='gelu'`, `mode_select='random'`, `modes=32`
<!-- model-card:canonical:end -->

## Paper
- **Title**: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
- **Venue**: ICML 2022
- **Published**: 2022 (arXiv: 2022-01)
- **arXiv**: https://arxiv.org/abs/2201.12740

## Abstract
Although Transformer-based methods have significantly improved state-of-the-art results for long-term series forecasting, they are not only computationally expensive but more importantly, are unable to capture the global view of time series (e.g. overall trend). To address these problems, we propose to combine Transformer with the seasonal-trend decomposition method, in which the decomposition method captures the global profile of time series while Transformers capture more detailed structures. To further enhance the performance of Transformer for long-term prediction, we exploit the fact that most time series tend to have a sparse representation in well-known basis such as Fourier transform, and develop a frequency enhanced Transformer. Besides being more effective, the proposed method, termed as Frequency Enhanced Decomposed Transformer (FEDformer), is more efficient than standard Transformer with a linear complexity to the sequence length. Our empirical studies with six benchmark datasets show that compared with state-of-the-art methods, FEDformer can reduce prediction error by 14.8% and 22.6% for multivariate and univariate time series, respectively.

## In ModernTSF
Default config: `configs/models/FEDformer.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`. The linked MIT repository remains
`reference-only`; its source was not copied. Structural and runtime evidence is
generated by `uv run tsf verify model FEDformer`.

## Verification

**Clean-room implementation: confirmed.** `FrequencyEnhancedBlock` maps paper
Eqs. (3)-(4), `FrequencyEnhancedAttention` maps Eqs. (6)-(7), and decoder
decomposition accumulates three trend updates per layer. Inputs are
`[B, seq_len, enc_in]` with optional six-column marks; outputs are
`[B, pred_len, c_out]`. Only Fourier mode is implemented, using head-local
complex kernels and deterministic random mode sets; wavelet, checkpoint, and
published-metric reference comparison are not claimed.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/ZhouMWW0022,
  author       = {Tian Zhou and
                  Ziqing Ma and
                  Qingsong Wen and
                  Xue Wang and
                  Liang Sun and
                  Rong Jin},
  editor       = {Kamalika Chaudhuri and
                  Stefanie Jegelka and
                  Le Song and
                  Csaba Szepesv{\'{a}}ri and
                  Gang Niu and
                  Sivan Sabato},
  title        = {FEDformer: Frequency Enhanced Decomposed Transformer for Long-term
                  Series Forecasting},
  booktitle    = {International Conference on Machine Learning, {ICML} 2022, 17-23 July
                  2022, Baltimore, Maryland, {USA}},
  series       = {Proceedings of Machine Learning Research},
  pages        = {27268--27286},
  publisher    = {{PMLR}},
  year         = {2022},
  url          = {https://proceedings.mlr.press/v162/zhou22g.html},
  timestamp    = {Thu, 23 Jan 2025 19:51:39 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/ZhouMWW0022.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
