---
name: "WDformer"
summary: "WDformer embeds each variate as a Haar-wavelet-subband token, attends over variate tokens with a differential-attention encoder (subtracting two softmax attention maps to cancel common-mode noise), and maps back to a forecast by projecting to pseudo-subband coefficients that a fixed inverse wavelet transform reassembles."
paper: "https://arxiv.org/abs/2509.25231"
paper_title: "WDformer: A Wavelet-based Differential Transformer Model for Time Series Forecasting"
venue: "arXiv"
year: 2025
code: "https://github.com/xiaowangbc/WDformer"
revision: "c741e4e428f802df41b8b113ad4a4d5241da4805"
license: "unspecified (no LICENSE file at the pinned revision)"
---
# WDformer

WDformer embeds each variate as a Haar-wavelet-subband token, attends over variate tokens with a differential-attention encoder (subtracting two softmax attention maps to cancel common-mode noise), and maps back to a forecast by projecting to pseudo-subband coefficients that a fixed inverse wavelet transform reassembles.

<!-- model-card:canonical:start -->
## Method overview

WDformer embeds each variate as a Haar-wavelet-subband token, attends over variate tokens with a differential-attention encoder (subtracting two softmax attention maps to cancel common-mode noise), and maps back to a forecast by projecting to pseudo-subband coefficients that a fixed inverse wavelet transform reassembles.

## Core architecture

WDformer embeds each variate as a Haar-wavelet-subband token, attends over variate tokens with a differential-attention encoder (subtracting two softmax attention maps to cancel common-mode noise), and maps back to a forecast by projecting to pseudo-subband coefficients that a fixed inverse wavelet transform reassembles.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.25231); title: WDformer: A Wavelet-based Differential Transformer Model for Time Series Forecasting; venue/year: arXiv / 2025
- [codebase](https://github.com/xiaowangbc/WDformer); revision: `c741e4e428f802df41b8b113ad4a4d5241da4805`; license: `unspecified (no LICENSE file at the pinned revision)`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/WDformer.toml`](../../../../configs/models/WDformer.toml).

## Differences

Clean-room implementation: confirmed. The wavelet-subband embedding, the
inverse-wavelet output head, and the differential-attention encoder loop were
re-derived from the pinned official files' module boundaries, default
hyperparameters, and docstrings; no source lines were copied. Differential
attention itself was extracted into the cataloged, paper-neutral
`differential_attention` component, since subtracting two softmax attention
maps with a learned mixing scalar is not specific to this paper's embedding
or head choice.

- The official code decomposes/reconstructs subbands with the third-party
  `ptwt` package's `wavedec`/`waverec`; this implementation uses the
  cataloged `DecimatedWaveletTransform`, a from-scratch strided-convolution
  Haar analysis/synthesis pair that is the same critically sampled orthogonal
  transform (verified round-trip exact on this catalog's own test suite).
- The official `EncoderLayer` (`layers/Transformer_EncDec.py`) applies three
  independently parameterized `RMSNorm` instances to the same input before
  projecting it to queries, keys, and values; because those projections are
  linear, three independent pre-norms and one shared pre-norm have the same
  representational capacity (the projection weights absorb any difference in
  a fixed per-branch rescaling), so this implementation uses one shared
  `RMSNorm` before the query/key/value projections.
- The official `EncoderLayer` also instantiates `conv1`/`conv2` `Conv1d`
  branches sized by `d_ff`, but `forward` never calls them -- the feed-forward
  branch is unconditionally `SwiGLU(d_model)`, whose width is fixed at
  `2 * d_model` and does not read `d_ff`. This implementation keeps `d_ff` as
  an accepted parameter for preset/CLI parity but it has no effect, matching
  the official dead code.
- Haar wavelet decomposition of a sequence whose length is not a multiple of
  `2 ** wave_size` requires one extra padded sample at some level; the
  learned projection to pseudo-subband coefficients is sized to the exact
  padded coefficient lengths (not `pred_len` directly), and
  `DecimatedWaveletTransform.reconstruct` trims the padding back out,
  returning exactly `pred_len` samples.

## Shared components

- [`differential_attention`](../_components/differential_attention/README.md)
- [`wavelet`](../_components/wavelet/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `dropout=0.1`, `wave_size=3`
<!-- model-card:canonical:end -->

## Source and verification

Clean-room implementation: confirmed. The wavelet-subband embedding, the
inverse-wavelet output head, and the differential-attention encoder loop were
re-derived from the pinned official files' module boundaries, default
hyperparameters, and docstrings; no source lines were copied. Differential
attention itself was extracted into the cataloged, paper-neutral
`differential_attention` component, since subtracting two softmax attention
maps with a learned mixing scalar is not specific to this paper's embedding
or head choice.

- The official code decomposes/reconstructs subbands with the third-party
  `ptwt` package's `wavedec`/`waverec`; this implementation uses the
  cataloged `DecimatedWaveletTransform`, a from-scratch strided-convolution
  Haar analysis/synthesis pair that is the same critically sampled orthogonal
  transform (verified round-trip exact on this catalog's own test suite).
- The official `EncoderLayer` (`layers/Transformer_EncDec.py`) applies three
  independently parameterized `RMSNorm` instances to the same input before
  projecting it to queries, keys, and values; because those projections are
  linear, three independent pre-norms and one shared pre-norm have the same
  representational capacity (the projection weights absorb any difference in
  a fixed per-branch rescaling), so this implementation uses one shared
  `RMSNorm` before the query/key/value projections.
- The official `EncoderLayer` also instantiates `conv1`/`conv2` `Conv1d`
  branches sized by `d_ff`, but `forward` never calls them -- the feed-forward
  branch is unconditionally `SwiGLU(d_model)`, whose width is fixed at
  `2 * d_model` and does not read `d_ff`. This implementation keeps `d_ff` as
  an accepted parameter for preset/CLI parity but it has no effect, matching
  the official dead code.
- Haar wavelet decomposition of a sequence whose length is not a multiple of
  `2 ** wave_size` requires one extra padded sample at some level; the
  learned projection to pseudo-subband coefficients is sized to the exact
  padded coefficient lengths (not `pred_len` directly), and
  `DecimatedWaveletTransform.reconstruct` trims the padding back out,
  returning exactly `pred_len` samples.
