---
name: "TimePro"
summary: "TimePro patches each channel independently, then runs a stack of bidirectional selective-state-space blocks over the variate axis whose scalar hidden state is reshaped into a (time-patch, variate) grid and locally mixed before read-out. This variable- and time-aware hyper-state feeds a linear head that projects to the forecast horizon."
paper: "https://arxiv.org/abs/2505.20774"
paper_title: "TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state"
venue: "arXiv preprint"
year: 2025
tagline: "Bidirectional selective-SSM scan over the variate axis, with scalar state grid-mixed across time patches and variates."
tags: ["ssm", "patching", "channel-mixing", "normalization"]
composition: ["normalization=local:instance-standardization", "decomposition=none", "temporal=local:patch-embedding-folded-into-variate-token", "channel=component:hyper_state_scan+local:bidirectional-variate-axis-scan", "head=local:final-norm-linear-projector", "loss=loss:mse"]
---
# TimePro

## Key ideas

- Each channel is patched and embedded, then all patch embeddings of a channel are concatenated into one token per variate, so the scan runs over the variate axis.
- `ProMamba` runs a diagonal selective scan (`diagonal_selective_scan`, scalar state, `d_state=1`) and reshapes the state into a (time-patch, variate) grid that `GridStateMixer` mixes locally before read-out: the hyper-state. The paper's CUDA-only deformable conv is replaced by a portable depthwise conv.
- `ProBlock` runs forward and reversed scans plus an MLP; a LayerNorm and linear projector emit the horizon.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.20774); title: TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state; venue/year: arXiv preprint / 2025
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimePro.toml`](../../../../configs/models/TimePro.toml).

## Differences

The official implementation (`model/TimePro.py`, `layers/TimePro_EncDec.py`)
was inspected at the pinned revision
(`70a20e5a257b30eb026ee4316293cf4feeb92a1f`) to resolve the
patch/hidden-dimension relationship, the bidirectional block wiring, and the
exact reshape that turns the scan's hidden state into a `(patch_num, n_var)`
grid. The repository carries no `LICENSE` file at that revision, so this
card omits `code`/`revision`/`license` provenance fields;
`verification/models.toml` records the inspected files with pinned blob
URLs instead.

The official code depends on `selective_scan_cuda_oflex_rh` (a custom CUDA
selective-scan kernel) and `DCNv4` (a CUDA-only deformable convolution) for,
respectively, the scan itself and the hyper-state grid mixing; neither is
CPU-portable, so neither is used or imported here:

- **Selective scan**: replaced by
  `hyper_state_scan.diagonal_selective_scan`, a pure-PyTorch sequential
  recurrence over the same `d_state=1` (scalar-state) formulation the paper
  uses by default; other `d_state` values are rejected rather than silently
  approximated.
- **Hyper-state mixing**: `DCNv4` (an offset-based deformable convolution
  that samples its receptive field from learned per-position offsets) is
  replaced by `hyper_state_scan.GridStateMixer`, a fixed-receptive-field
  depthwise 2-D convolution over the same `(time-patch, variate)` grid. This
  keeps the "mix a state with its spatial neighbors" semantics but drops the
  learned sampling offsets, and is expected to reduce fidelity to the
  paper's reported numbers; it was chosen because DCNv4 requires a custom
  CUDA extension with no CPU build.
- Stochastic depth (`DropPath`) is omitted (always identity); it is a
  training-time regularizer, not part of the forecasting computation.

No source file was copied or adapted.

## Shared components

- [`hyper_state_scan`](../_components/hyper_state_scan/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `d_model=16`, `e_layers=2`, `d_state=1`, `d_conv=5`, `expand=1.0`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper

TimePro forecasts by patching each channel (as PatchTST does), concatenating
each channel's patch embeddings into one token per *variate*, and running a
selective-state-space ("Mamba"-style) mixer across the variate axis whose
hidden state is additionally propagated across a 2-D grid of (time-patch,
variate) positions before it is read out -- the paper's "hyper-state" (paper
Section 3, Figure 2).

## Source and verification

The official implementation (`model/TimePro.py`, `layers/TimePro_EncDec.py`)
was inspected at the pinned revision
(`70a20e5a257b30eb026ee4316293cf4feeb92a1f`) to resolve the
patch/hidden-dimension relationship, the bidirectional block wiring, and the
exact reshape that turns the scan's hidden state into a `(patch_num, n_var)`
grid. The repository carries no `LICENSE` file at that revision, so this
card omits `code`/`revision`/`license` provenance fields;
`verification/models.toml` records the inspected files with pinned blob
URLs instead.

The official code depends on `selective_scan_cuda_oflex_rh` (a custom CUDA
selective-scan kernel) and `DCNv4` (a CUDA-only deformable convolution) for,
respectively, the scan itself and the hyper-state grid mixing; neither is
CPU-portable, so neither is used or imported here:

- **Selective scan**: replaced by
  `hyper_state_scan.diagonal_selective_scan`, a pure-PyTorch sequential
  recurrence over the same `d_state=1` (scalar-state) formulation the paper
  uses by default; other `d_state` values are rejected rather than silently
  approximated.
- **Hyper-state mixing**: `DCNv4` (an offset-based deformable convolution
  that samples its receptive field from learned per-position offsets) is
  replaced by `hyper_state_scan.GridStateMixer`, a fixed-receptive-field
  depthwise 2-D convolution over the same `(time-patch, variate)` grid. This
  keeps the "mix a state with its spatial neighbors" semantics but drops the
  learned sampling offsets, and is expected to reduce fidelity to the
  paper's reported numbers; it was chosen because DCNv4 requires a custom
  CUDA extension with no CPU build.
- Stochastic depth (`DropPath`) is omitted (always identity); it is a
  training-time regularizer, not part of the forecasting computation.

No source file was copied or adapted.
