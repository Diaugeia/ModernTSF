---
name: "TimeMachine"
summary: "TimeMachine forecasts with four Mamba state-space blocks at two embedding scales. After RevIN, two linear embeddings (L to n1, n1 to n2) feed an outer Mamba pair on the n1 scale and an inner Mamba pair on the n2 scale; under channel independence one Mamba of each pair reads the embedding as the token axis and the other as the feature axis, and under channel mixing the tokens are the channels. Two residual links, a projection back to n1, concatenation with the outer pair, and a final projection give the horizon."
paper: "https://arxiv.org/abs/2403.09898"
paper_title: "TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting"
venue: "ECAI 2024"
year: 2024
code: "https://github.com/Atik-Ahamed/TimeMachine"
revision: "5bf17a728349c674eceafc28a90f22b904e38c2d"
license: "Apache-2.0"
tagline: "Four Mamba blocks over n1- and n2-scale embeddings, outer and inner pairs, with residual links and two projections."
tags: ["ssm", "mamba", "channel-independent", "channel-mixing", "normalization", "multi-scale", "time-series"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:mamba", "channel=local:channel-independence-axis-swap", "head=local:two-stage-projection", "loss=loss:mse"]
---
# TimeMachine

## Key ideas

- `embed1` and `embed2` map the lookback to an `n1` and then an `n2` embedding; `mamba3`/`mamba4` act on the `n1` scale (outer pair) and `mamba1`/`mamba2` on the `n2` scale (inner pair), all `MambaBlock` mixers from the `mamba` component.
- With `ch_ind=True` each variate is a separate batch item; `mamba4` and `mamba1` see the embedding as a length-`n` sequence of width 1 (global context) while `mamba3` and `mamba2` see one token of width `n` (local context). With `ch_ind=False` the channels are the tokens.
- Residual links add the `n2` embedding before `proj1` and the `n1` embedding after it; `proj2` maps the concatenation of that result and the outer-pair output to the horizon.
- RevIN (`revin` component) normalizes and denormalizes per variate.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2403.09898); title: TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting; venue/year: ECAI 2024 / 2024
- [codebase](https://github.com/Atik-Ahamed/TimeMachine); revision: `5bf17a728349c674eceafc28a90f22b904e38c2d`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeMachine.toml`](../../../../configs/models/TimeMachine.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`mamba`](../_components/mamba/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `n1=256`, `n2=128`, `d_state=256`, `d_conv=2`, `expand=1`, `dropout=0.05`, `revin=True`, `ch_ind=True`, `residual=True`
<!-- model-card:canonical:end -->
