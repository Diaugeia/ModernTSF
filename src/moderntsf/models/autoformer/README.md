---
name: "Autoformer"
summary: "Autoformer is a Transformer-based model for long-term multivariate time series forecasting that replaces the standard self-attention mechanism with an Auto-Correlation mechanism and incorporates a progressive series decomposition block as a core inner component of the deep network rather than a pre-processing step."
paper: "https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html"
paper_title: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
venue: "NeurIPS 2021"
year: 2021
code: "https://github.com/thuml/Autoformer"
revision: "51c7d416ae120b805fd5beef2f4ccf7de496a6ff"
license: "MIT"
---
# Autoformer

Autoformer is a Transformer-based model for long-term multivariate time series forecasting that replaces the standard self-attention mechanism with an Auto-Correlation mechanism and incorporates a progressive series decomposition block as a core inner component of the deep network rather than a pre-processing step.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html); title: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting; venue/year: NeurIPS 2021 / 2021
- [codebase](https://github.com/thuml/Autoformer); revision: `51c7d416ae120b805fd5beef2f4ccf7de496a6ff`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Autoformer.toml`](../../../../configs/models/Autoformer.toml).

## Differences

**Clean-room implementation: confirmed.** `SeriesDecomposition` maps Eq. (1),
decoder initialization maps Eq. (2), encoder/decoder layers map Eqs. (3)-(4),
and FFT delay aggregation maps Eqs. (5)-(6). Inputs are `[B, seq_len, enc_in]`
with optional six-column marks; outputs are `[B, pred_len, c_out]`. This
forecast-only rewrite uses linear cross-context resizing and does not claim
checkpoint, training-recipe, or published-metric reference comparison.

## Shared components

- [`forecast_embedding`](../_components/forecast_embedding/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `dec_in=7`, `c_out=7`, `d_model=512`, `n_heads=8`, `e_layers=2`, `d_layers=1`, `d_ff=2048`, `moving_avg=25`, `factor=1.0`, `dropout=0.1`, `activation='gelu'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
- **Venue**: NeurIPS 2021
- **Published**: 2021 (arXiv: 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.13008

## Abstract
Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models. This design empowers Autoformer with progressive decomposition capacities for complex time series. Further, inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity, which conducts the dependencies discovery and representation aggregation at the sub-series level. Auto-Correlation outperforms self-attention in both efficiency and accuracy. In long-term forecasting, Autoformer yields state-of-the-art accuracy, with a 38% relative improvement on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease.

## In ModernTSF
Default config: `configs/models/Autoformer.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`. The linked MIT repository is retained as
`reference-only`; it was not copied for this rewrite. Structural and runtime
evidence is generated by `uv run tsf verify model Autoformer`.

## Verification

**Clean-room implementation: confirmed.** `SeriesDecomposition` maps Eq. (1),
decoder initialization maps Eq. (2), encoder/decoder layers map Eqs. (3)-(4),
and FFT delay aggregation maps Eqs. (5)-(6). Inputs are `[B, seq_len, enc_in]`
with optional six-column marks; outputs are `[B, pred_len, c_out]`. This
forecast-only rewrite uses linear cross-context resizing and does not claim
checkpoint, training-recipe, or published-metric reference comparison.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/WuXWL21,
  author       = {Haixu Wu and
                  Jiehui Xu and
                  Jianmin Wang and
                  Mingsheng Long},
  editor       = {Marc'Aurelio Ranzato and
                  Alina Beygelzimer and
                  Yann N. Dauphin and
                  Percy Liang and
                  Jennifer Wortman Vaughan},
  title        = {Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term
                  Series Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 34: Annual Conference
                  on Neural Information Processing Systems 2021, NeurIPS 2021, December
                  6-14, 2021, virtual},
  pages        = {22419--22430},
  year         = {2021},
  url          = {https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html},
  timestamp    = {Mon, 26 Jun 2023 20:41:56 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/WuXWL21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
