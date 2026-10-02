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
tags: ["ssm", "mamba", "patching", "disentangled", "channel-mixing", "normalization", "revin"]
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

- Implementation: local rewrite from Sections 5.1 and 5.2. The official code (`YukinoAsuna/SAMBA`, MIT) was inspected at revision `490257dac0806e16c00372226870850d39e8be65` (`model/SAMBA.py`, `layers/Block.py`, `layers/mixer_seq_simple.py`, `layers/samba_simple.py`, `scripts/SAMBA/`) to resolve omissions; no source was copied.
- Resolved from the official code: replication padding of one stride before unfolding, a bias-carrying patch projection `Linear(patch_len, d_model)`, a learnable `[patches, 1]` position term added only on the time branch, `Add -> LayerNorm -> mixer -> Add -> LayerNorm -> gated MLP` blocks returning `(hidden, residual)`, the shared stack norm applied to the residual after every block and to `hidden + residual` at the end, the `Linear(2d, d) -> GELU -> Dropout -> Linear(d, d)` fusion, a single flatten head shared across variates, and the step-size initialization of the reference Mamba (`dt_proj`). The gated MLP follows `mamba_ssm==2.2.2`: SiLU gate and hidden width rounded up to a multiple of 128.
- Differences from the official code: the selective scan, depthwise convolution and gating run through the shared pure-PyTorch `mamba` component (a sequential Python-loop scan) instead of the CUDA `mamba_ssm` kernels and `causal_conv1d`, so it is portable but much slower; `use_act=True` applies SiLU after the convolution as the official kernel path does. Other weights use PyTorch defaults, as the official encoders are built without the `MixerModel` initializer. The preset is `d_model=128`, `d_ff=128`, `e_layers=1`, `d_layers=1`, whereas the official script for Electricity uses `d_model=512`, `d_ff=512`, `e_layers=1`, `d_layers=2`, learning rate 5e-4, batch size 16 (the `run.py` defaults are 512, 128, 2 and 1); per-dataset settings are not preset. `e_layers=0` or `d_layers=0` keeps a single branch without the fusion MLP, as in the official code. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure tests in `tests/test_samba_structure.py`; no training was run and the CUDA kernels were not compared numerically.

## Shared components

- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_ff=128`, `e_layers=1`, `d_layers=1`, `d_state1=16`, `d_state2=16`, `patch_len=16`, `stride=8`, `dropout=0.1`, `use_act=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: SDE: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting
- **Venue**: KDD 2025
- **Published**: 2025 (arXiv: 2024-08)
- **arXiv**: https://arxiv.org/abs/2408.12068
- **Naming**: the paper proposes the SDE framework and its instantiation SAMBA (SDE-Mamba); this entry implements SAMBA.

## Source and verification

- Implementation: local rewrite from Sections 5.1 and 5.2. The official code (`YukinoAsuna/SAMBA`, MIT) was inspected at revision `490257dac0806e16c00372226870850d39e8be65` (`model/SAMBA.py`, `layers/Block.py`, `layers/mixer_seq_simple.py`, `layers/samba_simple.py`, `scripts/SAMBA/`) to resolve omissions; no source was copied.
- Resolved from the official code: replication padding of one stride before unfolding, a bias-carrying patch projection `Linear(patch_len, d_model)`, a learnable `[patches, 1]` position term added only on the time branch, `Add -> LayerNorm -> mixer -> Add -> LayerNorm -> gated MLP` blocks returning `(hidden, residual)`, the shared stack norm applied to the residual after every block and to `hidden + residual` at the end, the `Linear(2d, d) -> GELU -> Dropout -> Linear(d, d)` fusion, a single flatten head shared across variates, and the step-size initialization of the reference Mamba (`dt_proj`). The gated MLP follows `mamba_ssm==2.2.2`: SiLU gate and hidden width rounded up to a multiple of 128.
- Differences from the official code: the selective scan, depthwise convolution and gating run through the shared pure-PyTorch `mamba` component (a sequential Python-loop scan) instead of the CUDA `mamba_ssm` kernels and `causal_conv1d`, so it is portable but much slower; `use_act=True` applies SiLU after the convolution as the official kernel path does. Other weights use PyTorch defaults, as the official encoders are built without the `MixerModel` initializer. The preset is `d_model=128`, `d_ff=128`, `e_layers=1`, `d_layers=1`, whereas the official script for Electricity uses `d_model=512`, `d_ff=512`, `e_layers=1`, `d_layers=2`, learning rate 5e-4, batch size 16 (the `run.py` defaults are 512, 128, 2 and 1); per-dataset settings are not preset. `e_layers=0` or `d_layers=0` keeps a single branch without the fusion MLP, as in the official code. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure tests in `tests/test_samba_structure.py`; no training was run and the CUDA kernels were not compared numerically.

## Citation

```bibtex
@inproceedings{weng2025sde,
  title         = {{SDE}: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting},
  author        = {Weng, Zixuan and Han, Jindong and Jiang, Wenzhao and Liu, Hao},
  booktitle     = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year          = {2025},
  eprint        = {2408.12068},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2408.12068}
}
```
