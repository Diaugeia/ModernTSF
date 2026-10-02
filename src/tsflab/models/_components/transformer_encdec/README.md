---
name: "transformer_encdec"
kind: "component"
module: "tsflab.models._components.transformer_encdec"
summary: "Post-norm Transformer encoder/decoder layers and stacks (conv-FFN) with optional Informer distilling ConvLayer, parameterized by an injected attention module."
category: "backbone"
input: "Encoder/EncoderLayer: x [batch, len, d_model]; Decoder/DecoderLayer: x [batch, len_dec, d_model] and cross [batch, len_enc, d_model]; ConvLayer: [batch, len, d_model]"
output: "Encoder: ([batch, len', d_model], list of attention maps); EncoderLayer: ([batch, len, d_model], attn); DecoderLayer: [batch, len_dec, d_model]; Decoder: [batch, len_dec, d_model] or projection output; ConvLayer: [batch, ~len/2, d_model]"
origin: "Transformer (Vaswani et al., NeurIPS 2017) encoder/decoder layout with Informer (Zhou et al., AAAI 2021) distilling ConvLayer, in the Time-Series-Library layer API; upstream copy not recorded in history"
origin_models: ["informer", "transformer"]
tags: ["attention", "decoder", "encoder", "transformer", "post-norm", "distilling", "stateless"]
---

# transformer_encdec

## Purpose

Standard post-norm Transformer blocks where the attention module is injected, so
the same stack serves full attention (`transformer`) and ProbSparse attention
with distilling (`informer`).

- `EncoderLayer`: `x = x + drop(attn(x,x,x)); x = LN1(x); y = conv2(drop(act(conv1(x))));
  out = LN2(x + y)`, where `conv1`/`conv2` are kernel-size-1 `Conv1d` (a
  position-wise FFN with width `d_ff = d_ff or 4*d_model`).
- `Encoder`: runs `attn_layers` in order, optionally interleaved with `conv_layers`
  (distilling), then an optional final `norm_layer`.
- `DecoderLayer`: self-attention, cross-attention to `cross`, FFN, with
  LayerNorms `norm1..norm3` after each residual.
- `Decoder`: layers, optional `norm_layer`, optional `projection`.
- `ConvLayer(c_in)`: Informer distilling: circular `Conv1d(k=3, padding=2)`,
  `BatchNorm1d`, `ELU`, `MaxPool1d(k=3, stride=2, padding=1)` over time.

## Origin and granularity

Present from the initial commit (`0b1fbf2e`) and moved with the other shared
components (`33ea2050`, `02140040`). It was cut separately from the attention
cores (`self_attention_family`) so models choose the attention per role while
sharing the residual/norm/FFN skeleton. `informer`, `transformer` and
`dualformer` (only `EncoderLayer`) consume it. Model-local: embeddings, mask
construction, the generative decoder input, the output projection choice.

## Interface

- `ConvLayer(c_in)`: parameters under `down_conv`, `norm`; length maps
  `L -> floor((L+1)/2) + 1` for the circular padding of 2 plus the stride-2 pool.
  Uses `BatchNorm1d`, so it has running statistics buffers.
- `EncoderLayer(attention, d_model, d_ff=None, dropout=0.1, activation="relu")`:
  `activation == "relu"` selects ReLU, any other string selects GELU (no
  validation). `forward(x, attn_mask=None, tau=None, delta=None) -> (x, attn)`.
  State-dict keys: `attention.*`, `conv1`, `conv2`, `norm1`, `norm2`.
- `Encoder(attn_layers, conv_layers=None, norm_layer=None)`:
  `forward(x, attn_mask=None, tau=None, delta=None) -> (x, attns)`. With
  `conv_layers` the loop zips them with `attn_layers`, so `n-1` conv layers
  pair with the first `n-1` layers; the last attention layer is then called once
  more *without* `attn_mask`. `delta` is only given to the first layer.
  `attns` has one entry per attention layer.
- `DecoderLayer(self_attention, cross_attention, d_model, d_ff=None, dropout=0.1,
  activation="relu")`: `forward(x, cross, x_mask=None, cross_mask=None, tau=None,
  delta=None) -> x`. `tau` reaches both attentions, `delta` only the cross
  attention; attention maps are discarded.
- `Decoder(layers, norm_layer=None, projection=None)`: same forward signature;
  applies the final norm then the projection.

Attention modules must follow `attn(q, k, v, attn_mask=, tau=, delta=) ->
(out, attn)` over `[B, L, d_model]` (use `AttentionLayer`). Axis and dtype
follow the inputs; no module is stateful apart from `ConvLayer`'s batch-norm buffers.
No errors are raised here.

## Invariants and equivalence evidence

- `tests/test_component_contracts_attention.py`: shape, state-dict key, invariant, gradient-flow and seeded numerical-regression tests for every public symbol; reference values in `tests/fixtures/components/transformer_encdec_decoder.pt`, `tests/fixtures/components/transformer_encdec_encoder.pt`.
- `tests/test_local_attention_forecasters.py` asserts the `transformer` and
  `informer` models assemble these layers with the expected attention cores,
  `len(encoder.conv_layers) == e_layers - 1` for Informer distilling, and that
  both models return only the forecast horizon.
- `tests/test_dualformer_forecaster.py` constructs `EncoderLayer` inside `dualformer`.
- no pre-refactor fixture exists; `ConvLayer`, `Decoder`, and `DecoderLayer` are covered by the contract test.

## Variants and options

`conv_layers` turns on Informer distilling; `norm_layer` / `projection` on
`Encoder` / `Decoder` give the final norm and output head; `activation` picks
ReLU or GELU. Pre-norm, RoPE, or parameter-free attention variants are not
supported here.

## When to use and when not to use

Use for classic encoder-decoder Transformers (Vaswani, Informer-style) where the
attention is swappable. Do not use for patch encoders that want batch-first
`nn.TransformerEncoder` semantics (see `tst_transformer`), pre-norm stacks,
or decoders that must return attention maps (they are dropped).

## Related components

`self_attention_family` (attention cores and `AttentionLayer`), `tst_transformer`, `masking`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import tsflab.models._components.transformer_encdec
```

## Retrieval terms

`attention`, `decoder`, `encoder`, `transformer`

## Current model consumers (3)

`dualformer`, `informer`, `transformer`
<!-- component-card:generated:end -->
