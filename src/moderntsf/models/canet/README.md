---
name: "CANet"
summary: "CANet runs one branch per patch size over the instance-normalized lookback window, spectrally filters each branch's patches, restores non-stationary per-sample style statistics with an adaptive instance normalization (NSAN) instead of a fixed affine transform, and mixes patches with a dual-path convolution before concatenating all branches into a linear forecast head."
paper: "https://arxiv.org/abs/2504.17913"
paper_title: "CANet: ChronoAdaptive Network for Enhanced Long-Term Time Series Forecasting under Non-Stationarity"
venue: "arXiv preprint"
year: 2025
code: "https://github.com/mertsonmezer/CANet"
revision: "8b5d9cbdbf091805de0d22fa57b41173c21df95e"
license: "MIT"

---
# CANet

<!-- model-card:canonical:start -->
## Method overview

CANet runs one branch per patch size over the instance-normalized lookback window, spectrally filters each branch's patches, restores non-stationary per-sample style statistics with an adaptive instance normalization (NSAN) instead of a fixed affine transform, and mixes patches with a dual-path convolution before concatenating all branches into a linear forecast head.

## Core architecture

CANet runs one branch per patch size over the instance-normalized lookback window, spectrally filters each branch's patches, restores non-stationary per-sample style statistics with an adaptive instance normalization (NSAN) instead of a fixed affine transform, and mixes patches with a dual-path convolution before concatenating all branches into a linear forecast head.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2504.17913); title: CANet: ChronoAdaptive Network for Enhanced Long-Term Time Series Forecasting under Non-Stationarity; venue/year: arXiv preprint / 2025
- [codebase](https://github.com/mertsonmezer/CANet); revision: `8b5d9cbdbf091805de0d22fa57b41173c21df95e`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/CANet.toml`](../../../../configs/models/CANet.toml).

## Differences

The official implementation (`canet/canet.py`,
`canet/modules/{canet_layer,asb,icb,nsan,patchifier,positional_encoding,
skp_layer}.py`) was inspected at the pinned revision to resolve the exact
patch-then-style-then-spectral-then-NSAN-then-ICB execution order, the
straight-through adaptive frequency mask, and the NSAN statistic-blending
equations.

Only the paper's default normalization pairing (`LayerNorm` before the
spectral block, NSAN after it) is implemented; the official code's
`BatchNorm1d`/`InstanceNorm1d`/DAIN ablation options for either slot are not
reproduced. The official `StackedKroneckerProductLayer` (a
parameter-efficient, Kronecker-factored approximation of the
branch-concatenation projection) is replaced by a plain dense `nn.Linear`;
this changes the parameter count of that one projection but not the
forecasting computation it performs. Stochastic depth (`DropPath`) on the
branch residual is omitted (always identity); it is a training-time
regularizer, not part of the forecasting computation. No source file was
copied or adapted.

## Shared components

- [`adain_style_norm`](../_components/adain_style_norm/README.md)
- [`positional_encoding`](../_components/positional_encoding/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `patch_sizes=[8, 64]`, `embed_dim=32`, `output_features=1024`, `dropout=0.5`, `blend_ratio=0.1`
<!-- model-card:canonical:end -->

## Paper

CANet targets non-stationary series by not discarding the distributional
shift that instance normalization removes: instead of a fixed learned affine
transform, each patch branch re-injects per-sample mean/standard-deviation
"style" statistics (derived from both the raw series and a parallel style
embedding) through an Adaptive Instance Normalization step, alongside
multi-scale patching and frequency-domain filtering (paper Section 3).

## Source and verification

The official implementation (`canet/canet.py`,
`canet/modules/{canet_layer,asb,icb,nsan,patchifier,positional_encoding,
skp_layer}.py`) was inspected at the pinned revision to resolve the exact
patch-then-style-then-spectral-then-NSAN-then-ICB execution order, the
straight-through adaptive frequency mask, and the NSAN statistic-blending
equations.

Only the paper's default normalization pairing (`LayerNorm` before the
spectral block, NSAN after it) is implemented; the official code's
`BatchNorm1d`/`InstanceNorm1d`/DAIN ablation options for either slot are not
reproduced. The official `StackedKroneckerProductLayer` (a
parameter-efficient, Kronecker-factored approximation of the
branch-concatenation projection) is replaced by a plain dense `nn.Linear`;
this changes the parameter count of that one projection but not the
forecasting computation it performs. Stochastic depth (`DropPath`) on the
branch residual is omitted (always identity); it is a training-time
regularizer, not part of the forecasting computation. No source file was
copied or adapted.
