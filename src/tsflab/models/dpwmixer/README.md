---
name: "DPWMixer"
summary: "DPWMixer replaces average-pooling multi-scale downsampling with a lossless Haar wavelet pyramid, forecasting every resolution with a dual-path (global-linear trend plus patch-MLP local) mixer and fusing the per-channel, per-scale forecasts with a learned softmax weighting."
paper: "https://arxiv.org/abs/2512.02070"
paper_title: "DPWMixer: Dual-Path Wavelet Mixer for Long-Term Time Series Forecasting"
venue: "arXiv"
year: 2025
code: "https://github.com/hit636/DPWMixer"
revision: "0a787be2aeba845914bd4ea88428a1ddb982abad"
license: "Apache-2.0"
tagline: "Lossless Haar wavelet pyramid replaces pooling; each scale gets linear-trend plus patch-MLP paths, softmax-fused."
tags: ["mlp", "wavelet", "multi-scale", "patching", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=component:wavelet", "temporal=local:dual-path-global-linear-and-patch-mlp-mixer", "channel=local:channel-independent-shared-weights", "head=local:softmax-multi-scale-fusion", "loss=loss:mse"]
---
# DPWMixer

## Key ideas

- `wavelet` (`DecimatedWaveletTransform`, Haar) is applied once per level, keeping the approximation signal to build a multi-resolution pyramid without average-pooling information loss.
- `DualPathTrendMixer` forecasts each resolution by combining a global linear trend path with a patch-embedding MLP path, mixed by two learnable scalars (`path_weight`).
- Per-scale, per-channel forecasts are fused by softmax over learned `fusion_weight` logits; `revin` wraps the model.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2512.02070); title: DPWMixer: Dual-Path Wavelet Mixer for Long-Term Time Series Forecasting; venue/year: arXiv / 2025
- [codebase](https://github.com/hit636/DPWMixer); revision: `0a787be2aeba845914bd4ea88428a1ddb982abad`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DPWMixer.toml`](../../../../configs/models/DPWMixer.toml).

## Differences

Clean-room implementation: confirmed. The wavelet split, dual-path mixer, and
adaptive fusion were re-derived from the paper's description and the pinned
official file's module boundaries and default hyperparameters; no source
lines were copied.

- The official `HaarWaveletSplit` is a fixed one-level Haar analysis filter
  applied iteratively; this is reused here as one call per level of the
  cataloged, general `DecimatedWaveletTransform` (which also supports other
  wavelets and an exact multi-level inverse, unused by this model).
- The official code's per-resolution `DualTrendMixer` patch stage always uses
  the configured `patch_len`/`stride` regardless of how short the downsampled
  resolution has become, then zero-pads short inputs; this implementation
  additionally clamps `patch_len`/`stride` to the resolution's length so the
  patch count is never degenerate for very short scales.
- Official defaults use `d_model=64` and no `patch_len`/`stride` CLI
  exposure beyond 16/8; the TSFLab preset keeps `patch_len=16`, `stride=8`,
  and raises `d_model` to 128 to match this catalog's other mixer presets.
- The fusion softmax is computed once per forward call over
  `(down_sampling_layers + 1, enc_in)` learned logits, matching the official
  `Softmax(dim=0)` over the same shape.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`wavelet`](../_components/wavelet/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `dropout=0.1`, `patch_len=16`, `stride=8`, `down_sampling_layers=2`
<!-- model-card:canonical:end -->

## Source and verification

Clean-room implementation: confirmed. The wavelet split, dual-path mixer, and
adaptive fusion were re-derived from the paper's description and the pinned
official file's module boundaries and default hyperparameters; no source
lines were copied.

- The official `HaarWaveletSplit` is a fixed one-level Haar analysis filter
  applied iteratively; this is reused here as one call per level of the
  cataloged, general `DecimatedWaveletTransform` (which also supports other
  wavelets and an exact multi-level inverse, unused by this model).
- The official code's per-resolution `DualTrendMixer` patch stage always uses
  the configured `patch_len`/`stride` regardless of how short the downsampled
  resolution has become, then zero-pads short inputs; this implementation
  additionally clamps `patch_len`/`stride` to the resolution's length so the
  patch count is never degenerate for very short scales.
- Official defaults use `d_model=64` and no `patch_len`/`stride` CLI
  exposure beyond 16/8; the TSFLab preset keeps `patch_len=16`, `stride=8`,
  and raises `d_model` to 128 to match this catalog's other mixer presets.
- The fusion softmax is computed once per forward call over
  `(down_sampling_layers + 1, enc_in)` learned logits, matching the official
  `Softmax(dim=0)` over the same shape.
