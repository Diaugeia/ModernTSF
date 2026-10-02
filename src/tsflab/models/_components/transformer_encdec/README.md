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
tags: ["attention", "decoder", "encoder", "transformer", "post-norm", "distilling", "stateless", "encoder-decoder", "conv-ffn", "informer", "injected-attention", "vanilla-transformer"]
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
sharing the residual/norm/FFN skeleton. `informer` and `transformer` use the full set; `dualformer` uses only
`EncoderLayer`, and `gpht` uses `Encoder` and `EncoderLayer` (no decoder, no distilling). Model-local: embeddings, mask
construction, the generative decoder input, the output projection choice.

## Interface

- `ConvLayer(c_in)`: parameters under `down_conv`, `norm`; length maps
  `L -> floor((L+1)/2) + 1` for the circular padding of 2 plus the stride-2 pool.
  Uses `BatchNorm1d`, so it has running statistics buffers.
- `EncoderLayer(attention, d_model, d_ff=None, dropout=0.1, activation="relu")`:
  `activation` must be `"relu"` or `"gelu"`, else `ValueError`. `forward(x, attn_mask=None, tau=None, delta=None) -> (x, attn)`; `attn` is the
  attention module's second return value. State-dict keys: `attention.*`,
  `conv1.*`, `conv2.*`, `norm1.*`, `norm2.*`. `dropout` in [0, 1) is not validated.
- `Encoder(attn_layers, conv_layers=None, norm_layer=None)`:
  `forward(x, attn_mask=None, tau=None, delta=None) -> (x, attns)`. With
  `conv_layers` the loop zips them with `attn_layers`, so `n-1` conv layers
  pair with the first `n-1` layers; the last attention layer is then called once
  more *without* `attn_mask`. In that
  distilling path `delta` is only given to the first layer; without `conv_layers`
  it is passed to every layer. `attns` has one entry per attention layer and holds
  whatever the attention returns (`None` unless it outputs weights). `conv_layers`
  must have exactly `len(attn_layers) - 1` entries, else `ValueError` (more would skip
  trailing attention layers, fewer would run the last one twice).
  State-dict keys: `attn_layers.{i}.*`, `conv_layers.{i}.*`, `norm.*`.
- `DecoderLayer(self_attention, cross_attention, d_model, d_ff=None, dropout=0.1,
  activation="relu")`: `forward(x, cross, x_mask=None, cross_mask=None, tau=None,
  delta=None) -> x`. State-dict keys: `self_attention.*`, `cross_attention.*`,
  `conv1.*`, `conv2.*`, `norm1.*`, `norm2.*`, `norm3.*`. `tau` reaches both attentions, `delta` only the cross
  attention; attention maps are discarded.
- `Decoder(layers, norm_layer=None, projection=None)`: same forward signature;
  applies the final norm then the projection. State-dict keys: `layers.{i}.*`,
  `norm.*`, `projection.*` (the last two only when supplied).

Attention modules must follow `attn(q, k, v, attn_mask=, tau=, delta=) ->
(out, attn)` over `[B, L, d_model]` (use `AttentionLayer`). Axis and dtype
follow the inputs; no module is stateful apart from `ConvLayer`'s batch-norm buffers.
Errors raised here: `ValueError` for an unsupported `activation` and a wrong `conv_layers` length.

## Invariants and equivalence evidence

- `tests/test_component_contracts_attention.py`: `test_conv_layer` checks the
  state-dict keys (including `norm.running_mean`), the output length
  `(L + 1) // 2 + 1` for `L` in {8, 11}, and finite gradients;
  `test_encoder_layer_and_encoder` checks the `EncoderLayer` key groups, shape, that the
  output is layer-normalized (post-norm), gradients, a 3-layer `Encoder` with a final
  norm, a distilling `Encoder` with two `ConvLayer`s (three attention maps, shortened
  length), the GELU option, and seeded values against
  `tests/fixtures/components/transformer_encdec_encoder.pt`;
  `test_decoder_layer_and_decoder` checks the `DecoderLayer` key groups, shape, that
  later query positions do not change earlier outputs, gradients, `Decoder` with final
  norm and projection (`[2, 5, 3]`), and seeded values against
  `tests/fixtures/components/transformer_encdec_decoder.pt`.
- `tests/test_local_attention_forecasters.py` asserts that `transformer` assembles
  full-attention layers in the encoder, decoder self and cross roles (causal
  self-attention only), that `informer` uses ProbSparse attention with
  `len(encoder.conv_layers) == e_layers - 1` for distilling, and that both return
  only the forecast horizon.
- `tests/test_dualformer_forecaster.py` constructs `dualformer`, which uses `EncoderLayer`.
- No pre-extraction fixture exists, and the `delta` routing is not tested; the
  `activation` and `conv_layers` `ValueError`s are covered by
  `tests/test_component_validation.py`.

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

`self_attention_family` (attention cores and `AttentionLayer`; required to supply the
injected attention), `tst_transformer` (the other Transformer encoder component: a
batch-first stack of torch `nn.TransformerEncoderLayer` for patch tokens with no
decoder, no injected attention and no distilling, whereas this one is the injected-attention
Layer API with conv-FFN and a decoder), `masking` (builds the `attn_mask`/`x_mask` inputs).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `ConvLayer(c_in)`
  No symbol-level description is recorded.
- `EncoderLayer(attention, d_model, d_ff=None, dropout=0.1, activation='relu')`
  No symbol-level description is recorded.
- `Encoder(attn_layers, conv_layers=None, norm_layer=None)`
  No symbol-level description is recorded.
- `DecoderLayer(self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu')`
  No symbol-level description is recorded.
- `Decoder(layers, norm_layer=None, projection=None)`
  No symbol-level description is recorded.

```python
from tsflab.models._components.transformer_encdec import ConvLayer, EncoderLayer, Encoder, DecoderLayer, Decoder
```

## Retrieval terms

`attention`, `decoder`, `encoder`, `transformer`

## Current model consumers (4)

`dualformer`, `gpht`, `informer`, `transformer`
<!-- component-card:generated:end -->
