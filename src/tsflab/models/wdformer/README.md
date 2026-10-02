---
name: "WDformer"
summary: "WDformer embeds each variate as a Haar-wavelet-subband token, attends over variate tokens with a differential-attention encoder (subtracting two softmax attention maps to cancel common-mode noise), and maps back to a forecast by projecting to pseudo-subband coefficients that a fixed inverse wavelet transform reassembles."
paper: "https://arxiv.org/abs/2509.25231"
paper_title: "WDformer: A Wavelet-based Differential Transformer Model for Time Series Forecasting"
venue: "arXiv preprint"
year: 2025
code: "https://github.com/xiaowangbc/WDformer"
revision: "c741e4e428f802df41b8b113ad4a4d5241da4805"
license: "NOASSERTION"
tagline: "Haar subband tokens per variate, differential-attention encoder over variates, inverse Haar of pseudo-subbands."
tags: ["transformer", "wavelet", "frequency", "attention-variant", "channel-mixing", "normalization"]
composition: ["normalization=local:instance-standardization", "decomposition=component:wavelet", "temporal=local:subband-linear-embedding", "channel=component:differential_attention+local:variate-token-encoder", "head=local:pseudo-subband-projection+component:wavelet", "loss=loss:mse"]
---
# WDformer

## Key ideas

- `DecimatedWaveletTransform` (Haar, `wave_size` levels) splits each variate's history, and each subband gets its own linear embedding concatenated into one token per variate (inverted, variate-token layout).
- `DifferentialAttention` layers (`DifferentialSelfAttentionLayer`) subtract two softmax attention maps to cancel common-mode noise, with RMSNorm and SwiGLU feed-forward and a layer-dependent lambda schedule.
- A linear projector emits pseudo-subband coefficients and the inverse wavelet transform reassembles the forecast; calendar marks, if given, are appended as extra variates.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.25231); title: WDformer: A Wavelet-based Differential Transformer Model for Time Series Forecasting; venue/year: arXiv preprint / 2025
- [codebase](https://github.com/xiaowangbc/WDformer); revision: `c741e4e428f802df41b8b113ad4a4d5241da4805`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/WDformer.toml`](../../../../configs/models/WDformer.toml).

## Differences

The official repository publishes no LICENSE file (recorded as `NOASSERTION`). It was consulted only as a reference for paper details, and no source was copied; the implementation is an independent rewrite from the paper (see THIRD_PARTY_NOTICES.md).

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
- Depth schedule of the differential-attention `lambda_init`: the paper's
  Differential-Transformer schedule is `0.8 - 0.6 * exp(-0.3 * (l - 1))` with
  1-based layer index `l`; the pinned official `model/WDformer.py` instead uses
  `0.7 - 0.5 * exp(-0.3 * l)` with 0-based `l`. This implementation follows the
  official code (checked against the pinned revision), so the paper and code
  differ and TSFLab keeps the code's values.
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

The official repository publishes no LICENSE file (recorded as `NOASSERTION`). It was consulted only as a reference for paper details, and no source was copied; the implementation is an independent rewrite from the paper (see THIRD_PARTY_NOTICES.md).

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
- Depth schedule of the differential-attention `lambda_init`: the paper's
  Differential-Transformer schedule is `0.8 - 0.6 * exp(-0.3 * (l - 1))` with
  1-based layer index `l`; the pinned official `model/WDformer.py` instead uses
  `0.7 - 0.5 * exp(-0.3 * l)` with 0-based `l`. This implementation follows the
  official code (checked against the pinned revision), so the paper and code
  differ and TSFLab keeps the code's values.
- Haar wavelet decomposition of a sequence whose length is not a multiple of
  `2 ** wave_size` requires one extra padded sample at some level; the
  learned projection to pseudo-subband coefficients is sized to the exact
  padded coefficient lengths (not `pred_len` directly), and
  `DecimatedWaveletTransform.reconstruct` trims the padding back out,
  returning exactly `pred_len` samples.
