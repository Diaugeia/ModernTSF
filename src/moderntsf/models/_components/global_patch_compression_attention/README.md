---
name: "global_patch_compression_attention"
kind: "component"
module: "moderntsf.models._components.global_patch_compression_attention"
summary: "Two-stage cross-patch attention: each group's last patch queries all patches to form a summary token, then every patch attends to the summaries; post-norm residual blocks with GELU MLPs."
category: "attention"
input: "patches [batch, groups, patches_per_group, d_model]"
output: "[batch, groups, patches_per_group, d_model]"
origin: "Sensor Attention Block of Sensorformer (arXiv 2501.03284, 2025): cross-patch attention with global-patch compression"
origin_models: ["sensorformer"]
tags: ["attention", "compression", "cross-patch", "global", "patch", "sensor", "transformer", "linear-in-patches"]
---

# global_patch_compression_attention

## Purpose

For tokens `P [B, G, N, d]` (G groups such as variables, N patches each):

1. Stage 1 (compress): query `q_g = P[:, g, -1]` (the last patch of each group),
   keys/values all `G*N` patches flattened. `s = LN(q + MHA(q, P, P))`, then
   `s = LN(s + MLP(s))`: one summary ("sensor") token per group, `[B, G, d]`.
2. Stage 2 (broadcast): every patch attends to the `G` summaries,
   `r = LN(P + MHA(P, s, s))`, then `r = LN(r + MLP(r))`.

Cost `O(G^2 N d)` instead of `O(G^2 N^2 d)` for full cross-patch attention.

## Origin and granularity

Added with Sensorformer in the automated model intake (`6b491e13`), whose paper
describes the Sensor Attention Block; the only consumer is `sensorformer`,
which stacks `layers` copies. The cut is the block alone: the module docstring
frames it as a paper-neutral "compress tokens, then attend through the
compressed set" primitive. Patching, embedding and the flatten forecast head
stay in the model. The choice of the last patch as the compression query is
inherited from the model description and is hard-coded here.

## Interface

`GlobalPatchCompressionAttention(d_model, n_heads, d_ff, dropout=0.0)`:
`d_model`, `n_heads`, `d_ff` positive ints, `d_model % n_heads == 0` (else
`ValueError`); `dropout` in `[0, 1)` applied in both `nn.MultiheadAttention`
and both MLPs (`Linear-GELU-Dropout-Linear`).

`forward(patches)`: float `[B, G, N, d_model]`; raises `ValueError` if
`ndim != 4`. Returns the same shape. Parameters/state-dict keys: `compress_attn.*`,
`broadcast_attn.*` (`nn.MultiheadAttention`, batch-first), `compress_mlp.*`,
`broadcast_mlp.*`, `norm_compress_attn`, `norm_compress_mlp`,
`norm_broadcast_attn`, `norm_broadcast_mlp`. Stateless, no masking, no
positional information, attention maps are discarded. Because the query
is the last patch, patch order within a group matters; `N >= 1` is required.

## Invariants and equivalence evidence

- `tests/test_frequency_wavelet_attention_forecasters.py`
  (`test_global_patch_compression_attention_shapes_and_last_patch_query`):
  output shape equals input shape and perturbing non-last patches changes the
  output (information flows through the shared summaries).
- no fixture: no numeric fixture against the official Sensorformer code; the
  model `sensorformer` is also exercised by the runtime tests in the same file.

## Variants and options

Only `d_model`, `n_heads`, `d_ff`, `dropout`. A different compression query
(learned or mean-pooled summary) or multiple summaries per group are not options.

## When to use and when not to use

Use for patch-token forecasters over many variables where full cross-patch
attention is too costly and every patch should see cross-variable context
through a bottleneck. Do not use when ordering of groups must be preserved
causally (no mask), when `G` is tiny (the bottleneck gains nothing), or if the
last patch is not a sensible summary query.

## Related components

`patchtst` and `tst_transformer` (channel-independent encoders without
cross-variable mixing), `topk_expert_attention` (routed alternative),
`flatten_forecast_head` (the head `sensorformer` pairs with this block).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `GlobalPatchCompressionAttention(d_model: int, n_heads: int, d_ff: int, dropout: float=0.0)`
  Compress each group's patches into one summary, then attend through it.

```python
from moderntsf.models._components.global_patch_compression_attention import GlobalPatchCompressionAttention
```

## Retrieval terms

`attention`, `compression`, `cross-patch`, `global`, `patch`, `sensor`, `transformer`

## Current model consumers (1)

`sensorformer`
<!-- component-card:generated:end -->
