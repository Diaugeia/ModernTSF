---
name: "MQRNN"
summary: "MQRNN is a probabilistic direct multi-horizon forecaster: a shared LSTM encodes each series with historical temporal covariates, a global MLP jointly produces horizon-specific and horizon-agnostic contexts from the state and all known-future covariates, and one horizon-shared local MLP produces non-crossing quantiles."
paper: "https://arxiv.org/abs/1711.11053"
paper_title: "A Multi-Horizon Quantile Recurrent Forecaster"
venue: "NeurIPS 2017 Time Series Workshop"
year: 2017
---
# MQRNN

MQRNN is a probabilistic direct multi-horizon forecaster: a shared LSTM encodes
each series with historical temporal covariates, a global MLP jointly produces
horizon-specific and horizon-agnostic contexts from the state and all
known-future covariates, and one horizon-shared local MLP produces non-crossing
quantiles.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, quantiles]` quantile forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://arxiv.org/abs/1711.11053); title: A Multi-Horizon Quantile Recurrent Forecaster; venue/year: NeurIPS 2017 Time Series Workshop / 2017
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MQRNN.toml`](../../../../configs/models/MQRNN.toml).

## Differences

- Local implementation from the paper's Section 3.2 equations; no author implementation was inspected or copied.
- The local structure evaluates `(c_1,...,c_K,c_a)=m_G(h_t,x_future)` followed by `q_hat_k=m_L(c_k,c_a,x_future_k)`, with one local decoder shared across all horizons. Historical and known-future temporal covariates use `x_mark_enc` and `x_mark_dec`.
- ModernTSF uses its monotone `QuantileHead`, whereas the paper does not impose this parameterization. Static item covariates and the paper's forking-sequences training objective are not expressible by the repository's standard forecaster call and remain explicit experiment-layer limitations.

## Shared components

- [`quantile_head`](../_components/quantile_head/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `hidden_size=64`, `num_layers=1`, `context_size=32`, `decoder_hidden=64`, `future_covariate_size=6`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A Multi-Horizon Quantile Recurrent Forecaster
- **Authors**: Wen, Torkkola, Narayanaswamy, Madeka
- **Published**: 2017
- **arXiv**: https://arxiv.org/abs/1711.11053

## In ModernTSF
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/MQRNN.toml`; specification: `spec.py`; implementation:
`model.py`. Historical and known-future temporal covariates are passed through
`x_mark_enc` and `x_mark_dec`; `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.

## Source and verification

- Local implementation from the paper's Section 3.2 equations; no author implementation was inspected or copied.
- The local structure evaluates `(c_1,...,c_K,c_a)=m_G(h_t,x_future)` followed by `q_hat_k=m_L(c_k,c_a,x_future_k)`, with one local decoder shared across all horizons. Historical and known-future temporal covariates use `x_mark_enc` and `x_mark_dec`.
- ModernTSF uses its monotone `QuantileHead`, whereas the paper does not impose this parameterization. Static item covariates and the paper's forking-sequences training objective are not expressible by the repository's standard forecaster call and remain explicit experiment-layer limitations.
