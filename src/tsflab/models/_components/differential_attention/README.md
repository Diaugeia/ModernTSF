---
name: "differential_attention"
kind: "component"
module: "tsflab.models._components.differential_attention"
summary: "Differential self-attention: softmax(Q1K1) minus a learned lambda times softmax(Q2K2) applied to V, then per-head RMS renormalization scaled by (1 - lambda_init); no projections, no mask."
category: "attention"
input: "queries [batch, seq, heads, 2*head_dim]; keys [batch, key_seq, heads, 2*head_dim]; values [batch, key_seq, heads, 2*head_dim] (head_dim = d_model // num_heads)"
output: "[batch, seq, heads, 2*head_dim]"
origin: "Differential Transformer (Ye et al., ICLR 2025), as used in WDformer (arXiv 2509.25231, 2025)"
origin_models: ["wdformer"]
tags: ["attention", "differential", "noise-cancelling", "rmsnorm", "non-causal", "learned-lambda", "bidirectional"]
---

# differential_attention

## Purpose

For one head, split `Q` and `K` along the last axis into halves `(Q1, Q2)`,
`(K1, K2)` of width `head_dim`:

`A = softmax(s * Q1 K1^T) - lambda * softmax(s * Q2 K2^T)`, `s = head_dim^-0.5`,
`lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init`

(per head, `lq*`, `lk*` learned vectors). The output is `A V` (with dropout on
`A`), then each head's output vector is divided by its RMS over the feature axis
(`sqrt(mean(x^2) + 1e-5)`), multiplied by learned `rms_scale` and by
`(1 - lambda_init)`.

## Origin and granularity

Added with WDformer in the automated model intake (`d73cf9d0`); `wdformer` is
the only consumer. It is cut as the core operator only: `wdformer` keeps the
`DifferentialSelfAttentionLayer` that projects `d_model` to the doubled `Q/K/V`
widths (`2*d_model`), splits heads, and projects back with `out_proj`, plus its
own `RMSNorm`, SwiGLU feed-forward and encoder layer. `lambda_init` is a
constructor constant here; the depth schedule is model-local: `wdformer`
computes one value per encoder layer (`0.7 - 0.5 * exp(-0.3 * layer)`) and passes
it in.

## Interface

`DifferentialAttention(d_model, num_heads, lambda_init=0.8, attention_dropout=0.0)`:
`d_model % num_heads == 0` else `ValueError`; `head_dim = d_model // num_heads`.
Parameters (state-dict keys): `lambda_q1`, `lambda_k1`, `lambda_q2`, `lambda_k2`
(each `[num_heads, head_dim]`, init `N(0, 0.1^2)`), `rms_scale` (`[2*head_dim]`,
init 1, shared by all heads). `head_dim`, `num_heads`, `lambda_init`, `scale`
and `eps` are plain attributes. No other validation (`num_heads >= 1` is not
checked). `lambda` is unconstrained: it is not a convex combination and can be
negative or larger than one.

`forward(queries, keys, values)`: float tensors with `heads == num_heads`;
`queries` `[batch, seq, heads, 2*head_dim]`, `keys` and `values`
`[batch, key_seq, heads, 2*head_dim]` (`queries` and `keys` are each split in
half on the last axis; `values` keeps the full `2*head_dim`, which must match
`rms_scale`). `key_seq` may differ from `seq` (cross-attention works); `wdformer`
only uses self-attention. Returns `[batch, seq, heads, 2*head_dim]`. There are no
input/output projections and no shape validation (a wrong width fails inside
the chunk or einsum). No causal or padding mask: full bidirectional attention.
Dropout (active only in training mode) is applied to the differenced attention
map `A`, not to each softmax separately. Output dtype and device follow the
inputs.

## Invariants and equivalence evidence

- `tests/test_component_contracts_attention.py` (`test_differential_attention`):
  state-dict keys and parameter shapes, output shape and dtype, finite gradients
  for `queries`, `keys`, `values` and all parameters, `ValueError` for
  `d_model` not divisible by `num_heads`; seeded output pinned by
  `tests/fixtures/components/differential_attention.pt`.
  `test_differential_attention_is_bidirectional` shows that changing the last
  value token changes the first output position (no causal mask).
- `tests/test_wdformer_structure.py`: `wdformer` forward/backward over all
  parameters finite (including `lambda_*` and `rms_scale`), strict state-dict
  round trip, and `attention.attention._lambda()` finite.
- no fixture against the official Differential Transformer or WDformer code;
  the equation itself is not covered by an isolated closed-form unit test, and
  the cross-attention (`key_seq != seq`) path is untested.

## Variants and options

`lambda_init` (re-centres lambda and sets the output scale `1 - lambda_init`) and
`attention_dropout`. The causal variant and the in-component depth schedule of
`lambda_init` are not provided.

## When to use and when not to use

Use for bidirectional token mixing (for example variate or patch tokens) when a
sharper, noise-cancelled attention map is wanted, with an external projection
layer providing doubled `Q/K/V` widths. Do not use for causal decoding, when
the caller cannot supply `2*head_dim` per head, when a padding mask is needed,
or when attention maps must be non-negative (the difference can be negative).

## Related components

`self_attention_family` (plain single-softmax attention cores plus the
projecting `AttentionLayer`; this component is the differential alternative
without projections), `topk_expert_attention` (sparse routed alternative),
`wavelet` (the other building block of `wdformer`).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `DifferentialAttention(d_model: int, num_heads: int, lambda_init: float=0.8, attention_dropout: float=0.0)`
  Full (non-causal) differential self-attention.

```python
from tsflab.models._components.differential_attention import DifferentialAttention
```

## Retrieval terms

`attention`, `differential`, `noise-cancelling`, `rmsnorm`

## Current model consumers (1)

`wdformer`
<!-- component-card:generated:end -->
