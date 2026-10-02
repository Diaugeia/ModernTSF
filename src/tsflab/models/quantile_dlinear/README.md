---
name: "QuantileDLinear"
summary: "QuantileDLinear is a **probabilistic** TSFLab forecaster: it wraps the point DLinear backbone with the shared monotone `QuantileHead` (`src/tsflab/models/_components/quantile_head/README.md`) to emit a non-crossing grid of quantiles `(B, pred_len, C, Q)` instead of a single point. The head builds quantiles from a median anchor by adding/subtracting cumulative `softplus` offsets, so the predicted quantiles cannot cross by construction. It is trained with the pinball (`quantile`) loss and scored with CRPS / WQL / coverage."
paper: "https://arxiv.org/abs/2205.13504"
paper_title: "Are Transformers Effective for Time Series Forecasting? (DLinear backbone)"
venue: "AAAI 2023"
year: 2023
code: "https://github.com/cure-lab/LTSF-Linear"
revision: "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
license: "Apache-2.0"
---
# QuantileDLinear

QuantileDLinear is a **probabilistic** TSFLab forecaster: it wraps the point
DLinear backbone with the shared monotone `QuantileHead`
(`src/tsflab/models/_components/quantile_head/README.md`) to emit a non-crossing grid of quantiles
`(B, pred_len, C, Q)` instead of a single point. The head builds quantiles from a
median anchor by adding/subtracting cumulative `softplus` offsets, so the
predicted quantiles cannot cross by construction. It is trained with the pinball
(`quantile`) loss and scored with CRPS / WQL / coverage.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, quantiles]` quantile forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2205.13504); title: Are Transformers Effective for Time Series Forecasting? (DLinear backbone); venue/year: AAAI 2023 / 2023
- [codebase](https://github.com/cure-lab/LTSF-Linear); revision: `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/QuantileDLinear.toml`](../../../../configs/models/QuantileDLinear.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independently composed from verified shared components; no official reference source was copied.
- The probabilistic monotone head and pinball-loss protocol are TSFLab additions; this is not a model or result claimed by the DLinear paper.

## Shared components

- [`dlinear`](../_components/dlinear/README.md)
- [`quantile_head`](../_components/quantile_head/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `kernel_size=25`, `individual=False`
<!-- model-card:canonical:end -->

## Method
- **Backbone**: DLinear — trend + seasonal decomposition with two linear heads
  (Zeng et al., AAAI 2023, arXiv: 2205.13504).
- **Probabilistic head**: monotone quantile regression (pinball loss; Koenker &
  Bassett, 1978).

## In TSFLab
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/QuantileDLinear.toml`; specification: `spec.py`; implementation:
`model.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.

## Source and verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independently composed from verified shared components; no official reference source was copied.
- The probabilistic monotone head and pinball-loss protocol are TSFLab additions; this is not a model or result claimed by the DLinear paper.
