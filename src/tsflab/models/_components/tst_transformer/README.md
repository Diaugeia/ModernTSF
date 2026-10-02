---
name: "tst_transformer"
kind: "component"
module: "tsflab.models._components.tst_transformer"
summary: "Batch-first stack of torch nn.TransformerEncoderLayer blocks with a final BatchNorm or LayerNorm, used as the patch-token encoder of channel-independent forecasters."
category: "backbone"
input: "values [batch, tokens, d_model] (batch-first)"
output: "[batch, tokens, d_model]"
origin: "PatchTST time-series Transformer encoder (Nie et al., ICLR 2023, 'A Time Series is Worth 64 Words'); this version is a clean-room rewrite on torch nn.TransformerEncoder"
origin_models: ["patchtst"]
tags: ["attention", "encoder", "time-series", "transformer", "batch-first", "patch-token"]
---

# tst_transformer

## Purpose

`TSTEncoder(d_model, n_heads, ...)` encodes a sequence of tokens (for example
patch embeddings) with `n_layers` copies of `torch.nn.TransformerEncoderLayer`
(`batch_first=True`, `dim_feedforward=d_ff`, activation GELU by default,
`norm_first=pre_norm`) wrapped in `nn.TransformerEncoder`, followed by a final
norm that is either a `BatchNorm1d` over the feature axis (`_BatchFeatureNorm`)
or a `LayerNorm`. No positional information is added; callers add it before.

## Origin and granularity

The original PatchTST code carried a hand-written `TSTEncoder`; it was replaced by this
compact torch-native encoder with the same role. It is cut as its own component because
it is the only transformer part of the PatchTST backbone and is reusable by any
channel-independent patch or token model. Consumers: the `patchtst` component, the
`stdmae` model (final `LayerNorm`), and the `composed` model through its `temporal` slot
adapter (`models/_slots/adapters.py`, which also serves `component:patchtst`). It is not a
numerical copy of the original: the BatchNorm choice only affects the *final* norm (the
inner layers always use torch's LayerNorm), residual attention and custom `d_k`/`d_v` are
dropped. Embeddings, patching, and heads stay in the consumer.

## Interface

The only public symbol is `TSTEncoder` (the module defines no `__all__` and the
catalog entry lists no public symbols, so the generated block shows only the module
import; `_BatchFeatureNorm` and `_activation` are private).

`TSTEncoder(d_model, n_heads, *, n_layers=3, d_k=None, d_v=None, d_ff=256,
activation="gelu", norm="BatchNorm", attn_dropout=0.0, res_dropout=0.0,
ffn_dropout=0.0, proj_dropout=0.0, pre_norm=False, **_)`

- `d_model` divisible by `n_heads`, else `ValueError`; `n_layers` and `d_ff` are not
  validated here (torch errors apply).
- `d_k`/`d_v` must be `None` or `d_model // n_heads`, else `ValueError`.
- `activation`: `"relu"`, `"gelu"` (case-insensitive) or a callable, else `ValueError`.
- `norm`: any string containing `"batch"` or `"layer"` (case-insensitive), else
  `ValueError`; it selects only the final norm after the layer stack.
- The four dropout arguments collapse to one rate, `max(attn, res, ffn, proj)`,
  used for every dropout inside the torch layer.
- Extra keyword arguments are silently swallowed (`**_`), as are extra keywords to `forward`.

`forward(values, **_) -> Tensor` with `values` `[batch, tokens, d_model]`; output
has the same shape. State-dict keys: `layers.layers.<i>.*` (torch layer
parameters) and `layers.norm.*` (`layers.norm.norm.*`, including
`running_mean`/`running_var`/`num_batches_tracked` buffers, for the BatchNorm
variant). The BatchNorm variant is stateful in training (running statistics).
Float dtype/device follow the input.

## Invariants and equivalence evidence

- `test_tst_encoder` (BatchNorm/post-norm and LayerNorm/pre-norm) and
  `test_tst_encoder_validation` in `tests/test_component_contracts_attention.py` check:
  state-dict keys `layers.layers.<i>.*` and `layers.norm.*`, `running_mean` present only for
  BatchNorm, output shape and dtype, extra `forward` keywords ignored, finite input gradient
  and parameter gradients in training mode, and `ValueError` for indivisible `d_model`,
  `d_k=3`, `activation="swish"`, and `norm="rms"` (extra constructor keywords and a
  callable activation accepted). Reference values (eval-mode output):
  `tests/fixtures/components/tst_transformer_batchnorm.pt` and
  `tests/fixtures/components/tst_transformer_layernorm.pt`.
- `tests/test_repository_contracts.py` asserts the dependency closure of
  `patchtst` contains `tst_transformer`.
- `tests/test_probabilistic_attention_forecasters.py` exercises it through
  `QuantilePatchTST` (forward only; shape and quantile contracts).
- No fixture compares this encoder with the original PatchTST `TSTEncoder`, and the
  rewrite is known not to be state-dict compatible with it.

## Variants and options

`pre_norm=True` for pre-norm layers; `norm="LayerNorm"` to avoid batch statistics
(`stdmae` uses it); `activation` as relu/gelu/callable. Per-layer
BatchNorm and residual attention are not available.

## When to use and when not to use

Use as the encoder for `[B*C, patches, d_model]` token sequences in
channel-independent patch forecasters. Do not use when you need per-layer
BatchNorm, distinct key/value widths, attention maps, masks, or swappable
attention cores (use `transformer_encdec` with `self_attention_family`).

## Related components

`patchtst` (consumer: the full backbone that wraps this encoder with RevIN, patching, and a head),
`positional_encoding` (add before encoding), `transformer_encdec` (the other Transformer
encoder stack: post-norm with conv-FFN and an injected attention core, versus this fused torch
layer stack), `self_attention_family` (swappable attention cores this encoder cannot take).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import tsflab.models._components.tst_transformer
```

## Retrieval terms

`attention`, `encoder`, `time-series`, `transformer`

## Current model consumers (2)

`composed`, `stdmae`
<!-- component-card:generated:end -->
