---
name: "SAMBA"
summary: "SAMBA (SDE-Mamba) encodes patch embeddings with parallel activation-free Mamba stacks over time and over variates, fuses them with an FFN, and projects with a flatten head."
paper: "https://arxiv.org/abs/2408.12068"
paper_title: "SDE: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting"
venue: "KDD 2025"
year: 2025
code: "https://github.com/YukinoAsuna/SAMBA"
revision: "490257dac0806e16c00372226870850d39e8be65"
license: "MIT"
tagline: "Patch embeddings through parallel simplified-Mamba time and variate stacks, FFN fusion, flatten head."
tags: ["ssm", "mamba", "patching", "disentangled", "channel-mixing", "normalization", "time-series", "revin", "flatten_forecast_head"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:mamba", "channel=component:mamba", "head=component:flatten_forecast_head", "loss=none"]
---
# SAMBA

## Key ideas

- Simplification: `SimplifiedMambaMixer` reuses the `mamba` component's selective SSM (`ssm`) but feeds the depthwise convolution output to it without the SiLU activation (paper Sec. 5.1); the SiLU gate is kept.
- Disentangled dependency encoding: after patch embedding, a cross-time S-Mamba stack (per variate, over patches, with a learnable position term) and a cross-variate stack (per patch, over variates) run in parallel on the same embedding (Sec. 5.2).
- The two outputs are concatenated, mixed by an FFN, flattened and projected to the horizon with `flatten_forecast_head`; `revin` (no affine) wraps the whole model.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2408.12068); title: SDE: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting; venue/year: KDD 2025 / 2025
- [codebase](https://github.com/YukinoAsuna/SAMBA); revision: `490257dac0806e16c00372226870850d39e8be65`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SAMBA.toml`](../../../../configs/models/SAMBA.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_ff=128`, `e_layers=1`, `d_layers=1`, `d_state1=16`, `d_state2=16`, `patch_len=16`, `stride=8`, `dropout=0.1`, `use_act=False`
<!-- model-card:canonical:end -->
