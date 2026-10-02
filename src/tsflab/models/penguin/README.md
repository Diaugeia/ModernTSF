---
name: "PENGUIN"
summary: "PENGUIN is a channel-independent patch Transformer whose self-attention is a periodic-nested group attention: query heads are split into groups that each share one key/value head and add an ALiBi-style bias built from a triangle wave of the patch distance modulo that group's period, so one head group per known cycle (for example daily and weekly) sees periodic structure while a causal mask and a flatten-linear head produce the forecast."
paper: "https://arxiv.org/abs/2508.13773"
paper_title: "PENGUIN: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting"
venue: "AISTATS"
year: 2026
code: "https://github.com/ysygMhdxw/AISTATS2026_PENGUIN"
revision: "d9a2b84f9fa214c91c4c1c51e5a7f95fb451cefb"
license: "unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)"
tagline: "Grouped multi-query patch attention with a per-group periodic ALiBi bias, one head group per cycle length."
tags: ["transformer", "patching", "periodicity", "attention-variant", "channel-independent", "normalization", "time-series", "revin"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:periodic_alibi_bias+local:grouped-multi-query-causal-attention-encoder", "channel=local:channel-independent-shared-weights", "head=component:flatten_forecast_head", "loss=loss:mse"]
---
# PENGUIN

## Key ideas

- `PeriodicNestedGroupAttention` splits `n_heads` into one group per period; each group shares a single key/value head (grouped multi-query attention) and keeps its own relative-position bias.
- `periodic_alibi_bias` gives head `k` of a group the bias `-m_k * tri(|i - j| mod P_S)` with `m_k = 2^(-8k/n)`; periods are given in time steps and converted to patch units by dividing by `stride`.
- An encoder layer is attention then residual and norm, ReLU/GELU feed-forward then residual and norm, with causal masking over patch tokens and a final BatchNorm; `use_rmsnorm` swaps LayerNorm for RMSNorm.
- Inputs are RevIN-normalized (`revin`), embedded per channel by replicate-padded patching (`embed.PatchEmbedding`), and read out by `flatten_forecast_head`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2508.13773); title: PENGUIN: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting; venue/year: AISTATS / 2026
- [codebase](https://github.com/ysygMhdxw/AISTATS2026_PENGUIN); revision: `d9a2b84f9fa214c91c4c1c51e5a7f95fb451cefb`; license: `unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PENGUIN.toml`](../../../../configs/models/PENGUIN.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`periodic_alibi_bias`](../_components/periodic_alibi_bias/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_ff=256`, `n_heads=8`, `e_layers=2`, `patch_len=16`, `stride=8`, `dropout=0.1`, `periods=[24]`, `activation='relu'`, `use_rmsnorm=False`, `alibi=True`, `causal=True`
<!-- model-card:canonical:end -->
