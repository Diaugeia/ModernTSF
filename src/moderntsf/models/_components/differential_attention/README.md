---
name: "differential_attention"
kind: "component"
module: "moderntsf.models._components.differential_attention"
summary: "Differential self-attention: softmax(Q1K1) minus a learned lambda times softmax(Q2K2) applied to V, then per-head RMS renormalization scaled by (1 - lambda_init); no projections."
category: "attention"
input: "queries, keys: [batch, seq, heads, 2*head_dim]; values: [batch, seq, heads, 2*head_dim] (head_dim = d_model // num_heads)"
output: "[batch, seq, heads, 2*head_dim]"
origin: "Differential Transformer (Ye et al., ICLR 2025, 'Differential Transformer'), as used in WDformer (arXiv 2509.25231, 2025)"
origin_models: ["wdformer"]
tags: ["attention", "differential", "noise-cancelling", "rmsnorm", "non-causal", "learned-lambda"]
---

# differential_attention

## Purpose

For one head, split `Q` and `K` along the last axis into halves `(Q1, Q2)`,
`(K1, K2)` of width `head_dim`:

`A = softmax(s * Q1 K1^T) - lambda * softmax(s * Q2 K2^T)`, `s = head_dim^-0.5`,
`lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init`

(per head, `lq*`, `lk*` learned vectors). The output is `A V`, then each head's
output is divided by its RMS over the feature axis (`sqrt(mean(x^2) + 1e-5)`),
multiplied by learned `rms_scale` and by `(1 - lambda_init)`.

## Origin and granularity

Added with WDformer in the automated model intake (`d73cf9d0`); `wdformer` is
the only consumer. It is cut as the core operator only: `wdformer` keeps the
`DifferentialSelfAttentionLayer` that projects `d_model` to the doubled `Q/K/V`
widths (`2*d_model`), splits heads, and projects back with `out_proj`, plus its
own `RMSNorm`, SwiGLU feed-forward and encoder layer. `lambda_init` is a
constructor constant (0.8), not the depth-dependent schedule; I did not find
a depth schedule recorded in history.

## Interface

`DifferentialAttention(d_model, num_heads, lambda_init=0.8, attention_dropout=0.0)`:
`d_model % num_heads == 0` else `ValueError`; `head_dim = d_model // num_heads`.
Parameters (state-dict keys): `lambda_q1`, `lambda_k1`, `lambda_q2`, `lambda_k2`
(each `[num_heads, head_dim]`, init `N(0, 0.1^2)`), `rms_scale` (`[2*head_dim]`,
init 1, shared by all heads). `lambda_init`, `scale`, `eps` are plain attributes.

`forward(queries, keys, values)`: all four-axis float tensors
`[batch, seq, heads, 2*head_dim]` with `heads == num_heads`; `queries` and `keys`
last axis `2*head_dim` (split in half), values last axis `2*head_dim`.
Returns `[batch, seq, heads, 2*head_dim]`. There are no input/output
projections and no shape validation (a wrong width fails inside the chunk or
einsum). No causal mask or padding mask: full bidirectional attention.
Dropout is applied to the differenced attention map `A` (not to each softmax
separately). Key/value length may equal query length only in practice (the
output reshape uses the query `seq`). Stateless.

## Invariants and equivalence evidence

- `tests/test_wdformer_structure.py`: `wdformer` forward/backward over all
  parameters finite (including `lambda_*` and `rms_scale`), strict state-dict
  round trip, and `attention.attention._lambda()` finite.
- no fixture: no numeric fixture against the official Differential Transformer
  or WDformer code; the equation is not covered by an isolated unit test.

## Variants and options

`lambda_init` (re-centres lambda and sets the output scale `1 - lambda_init`) and
`attention_dropout`. The causal and multi-layer-lambda-init variants of the
paper are not provided.

## When to use and when not to use

Use for bidirectional token mixing (for example variate or patch tokens) when a
sharper, noise-cancelled attention map is wanted, with an external projection
layer providing doubled `Q/K/V` widths. Do not use for causal decoding, when
the caller cannot supply `2*head_dim` per head, or when attention maps must be
non-negative (the difference can be negative).

## Related components

`self_attention_family` (plain softmax attention), `topk_expert_attention`
(sparse routed alternative), `wavelet` (the other building block of `wdformer`).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `DifferentialAttention(d_model: int, num_heads: int, lambda_init: float=0.8, attention_dropout: float=0.0)`
  Full (non-causal) differential self-attention.

```python
from moderntsf.models._components.differential_attention import DifferentialAttention
```

## Retrieval terms

`attention`, `differential`, `noise-cancelling`, `rmsnorm`

## Current model consumers (1)

`wdformer`
<!-- component-card:generated:end -->
