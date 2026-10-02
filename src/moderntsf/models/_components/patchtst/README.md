---
name: "patchtst"
kind: "component"
module: "moderntsf.models._components.patchtst"
summary: "PatchTSTBackbone: RevIN, per-channel patching with unfold, linear patch embedding, position table, TSTEncoder, flatten head; channel-independent direct multi-horizon forecast."
category: "backbone"
input: "[batch, context_window, channels]"
output: "[batch, target_window, channels]"
origin: "PatchTST, Nie et al., ICLR 2023 (A Time Series is Worth 64 Words: Long-term Forecasting with Transformers)"
origin_models: ["patchtst"]
tags: ["backbone", "channel-independent", "patch", "transformer", "forecasting", "composite"]
---

# patchtst

## Purpose

`PatchTSTBackbone` is the whole PatchTST forward pass as a reusable module.
For input `x [B, L, C]`:

1. `RevIN` normalization (`norm`), then `[B, C, L]`.
2. Optional end padding: replicate-pad `stride` steps when `padding_patch="end"`.
3. `unfold(-1, patch_len, stride)` gives patches `[B, C, N, patch_len]` with
   `N = 1 + (L' - patch_len) // stride`, `L' = L + stride` if padded.
4. Linear `patch_len -> d_model`, add the `[N, d_model]` position table, input dropout.
5. Fold channels into the batch (`[B*C, N, d_model]`), run `TSTEncoder`
   (all channels share weights: channel independence).
6. Back to `[B, C, d_model, N]`, `FlattenForecastHead` (flatten `d_model * N` then
   linear to `target_window`, shared or per-channel via `individual`),
   transpose to `[B, H, C]`, RevIN `denorm`.

## Origin and granularity

Implements PatchTST (Nie et al., ICLR 2023). It was a shared component from the
initial refactors and was rewritten as a clean-room composition in `fba5fa99`
("finish clean-room shared forecasting layer"); the flatten head was split out
in `d38451c3`. The cut keeps the composition (normalize, patch, embed, encode,
head) separate from the pieces that are reusable alone: `revin`,
`positional_encoding`, `tst_transformer`, `flatten_forecast_head`. In the
repository the backbone is consumed only by `quantile_patchtst`; the
`patchtst` model package keeps its own model-local implementation (it imports
only `revin`), so this component is not what the point-forecast `patchtst`
model runs. Quantile heads, decomposition and masked pretraining stay outside.

## Interface

`PatchTSTBackbone(c_in, context_window, target_window, patch_len, stride,
padding_patch, n_layers, d_model, n_heads, d_k, d_v, d_ff, activation, norm,
attn_dropout, res_dropout, ffn_dropout, proj_dropout, head_dropout, pre_norm,
pe, learn_pe, head_type, individual, revin, affine, subtract_last)`: every
argument is required (no defaults). Meaning: `c_in` channels; `context_window`
history length `L` (exact); `target_window` horizon `H`; `patch_len >= 1`,
`stride >= 1`, `patch_len <= context_window`; `padding_patch` is `None` or
`"end"`; encoder arguments are forwarded to `TSTEncoder` (`d_k`/`d_v` must be
`None` or `d_model // n_heads`); `pe` / `learn_pe` go to `positional_encoding`;
`head_type` must be `"flatten"`; `individual` selects per-channel heads;
`revin`, `affine`, `subtract_last` configure `RevIN` (`revin=False` disables it).

`forward(values, *_)`: `values` float `[B, context_window, c_in]`; extra
positional arguments (for example marks) are ignored. Returns `[B, target_window,
c_in]`. Raises `ValueError` for bad shape/length, `patch_len`/`stride`, unsupported
`padding_patch`, or `head_type != "flatten"` (plus errors from the sub-components).
State-dict prefixes: `normalizer.*` (RevIN affine), `patch_projection.*`,
`position` (the table parameter), `encoder.layers.*`, `head.*`. The module is
stateful during a forward pass only through the RevIN statistics cache and the
encoder's BatchNorm running statistics, so do not share one instance across
concurrent forwards.

## Invariants and equivalence evidence

- `tests/test_repository_contracts.py` pins the dependency closure of `patchtst`
  to `flatten_forecast_head`, `patchtst`, `positional_encoding`, `revin`,
  `tst_transformer`.
- `tests/test_probabilistic_attention_forecasters.py` runs it through
  `QuantilePatchTST` (shape, quantile contracts).
- `tests/test_transformer_patch_forecasters_a.py` checks patch overlap and channel
  independence on the model-local `PatchTST` (channel permutation equivariance),
  not on this backbone directly; the backbone shares the same channel-folding
  structure.
- no fixture: no numeric fixture compares this backbone with the original PatchTST
  code.

## Variants and options

`padding_patch="end"` (original PatchTST option) adds one extra patch;
`individual=True` gives one linear head per channel; `revin=False` /
`subtract_last=True` / `affine` for normalization; `pre_norm`, `pe`, `learn_pe`,
`norm` for the encoder. Decomposition (`DLinear`-style) and `head_type` other
than `"flatten"` are not supported.

## When to use and when not to use

Use as the point-forecast backbone for channel-independent patch Transformers
(for example under a quantile head). Do not use for channel-mixing, masked
pretraining or models that need marks or exogenous inputs, and do not expect it
to reproduce the model-local `patchtst` numerics.

## Related components

`revin`, `positional_encoding`, `tst_transformer`, `flatten_forecast_head`,
`quantile_head` (used with it in `quantile_patchtst`).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `PatchTSTBackbone(c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, padding_patch: str | None, n_layers: int, d_model: int, n_heads: int, d_k: int | None, d_v: int | None, d_ff: int, activation: str, norm: str, attn_dropout: float, res_dropout: float, ffn_dropout: float, proj_dropout: float, head_dropout: float, pre_norm: bool, pe: str, learn_pe: bool, head_type: str, individual: bool, revin: bool, affine: bool, subtract_last: bool)`
  Patch each channel independently, encode patches, and forecast directly.

```python
from moderntsf.models._components.patchtst import PatchTSTBackbone
```

## Retrieval terms

`backbone`, `channel-independent`, `patch`, `transformer`

## Current model consumers (1)

`quantile_patchtst`
<!-- component-card:generated:end -->
