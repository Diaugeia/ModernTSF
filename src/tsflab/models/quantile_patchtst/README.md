---
name: "QuantilePatchTST"
summary: "QuantilePatchTST is a **probabilistic** TSFLab forecaster: it wraps the patch-based Transformer backbone PatchTST with the shared monotone `QuantileHead` (`src/tsflab/models/_components/quantile_head/README.md`) to emit a non-crossing quantile grid `(B, pred_len, C, Q)`. Quantiles are built from a median anchor via cumulative `softplus` offsets, so they cannot cross. Trained with the pinball (`quantile`) loss and scored with CRPS / WQL / coverage."
paper: "https://arxiv.org/abs/2211.14730"
paper_title: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST backbone)"
venue: "ICLR 2023"
year: 2023
code: "https://github.com/yuqinie98/PatchTST"
revision: "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
license: "Apache-2.0"
tagline: "PatchTST point backbone plus a monotone quantile head emitting non-crossing quantiles; pinball loss."
tags: ["transformer", "probabilistic", "quantile", "patching", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:patchtst", "channel=local:channel-independent-shared-weights", "head=component:flatten_forecast_head+component:quantile_head", "loss=loss:quantile"]
---
# QuantilePatchTST

## Key ideas

- `PatchTSTBackbone` (patch tokens, channel-independent Transformer encoder, flatten head, RevIN) produces the point forecast.
- `QuantileHead` turns each point into `Q` quantiles from a median anchor with cumulative softplus offsets, so quantiles cannot cross.
- `output_type = "quantile"` returns `(B, pred_len, C, Q)`; pair it with the pinball (`quantile`) loss.
- The head and pinball protocol are TSFLab additions, not part of the PatchTST paper.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, quantiles]` quantile forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2211.14730); title: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST backbone); venue/year: ICLR 2023 / 2023
- [codebase](https://github.com/yuqinie98/PatchTST); revision: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/QuantilePatchTST.toml`](../../../../configs/models/QuantilePatchTST.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independently composed from verified shared components; no official reference source was copied.
- The local PatchTST backbone is composed with TSFLab's monotone quantile head. The cited paper's point-forecast results do not validate this probabilistic composition.

## Shared components

- [`patchtst`](../_components/patchtst/README.md)
- [`quantile_head`](../_components/quantile_head/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `e_layers=3`, `d_model=128`, `n_heads=8`, `d_ff=256`
<!-- model-card:canonical:end -->

## Method
- **Backbone**: PatchTST — channel-independent patching + Transformer encoder
  (Nie et al., ICLR 2023, arXiv: 2211.14730).
- **Probabilistic head**: monotone quantile regression (pinball loss).

## In TSFLab
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/QuantilePatchTST.toml`; specification: `spec.py`; implementation:
`model.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.

## Source and verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independently composed from verified shared components; no official reference source was copied.
- The local PatchTST backbone is composed with TSFLab's monotone quantile head. The cited paper's point-forecast results do not validate this probabilistic composition.
