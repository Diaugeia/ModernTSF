---
name: "topk_expert_attention"
kind: "component"
module: "tsflab.models._components.topk_expert_attention"
summary: "Temporal Mix-of-Experts attention: each query keeps its top-k scoring key/value positions (softmax-reweighted), optional shared mean key/value expert, depthwise-conv positional residual."
category: "attention"
input: "x [batch, tokens, dim]"
output: "[batch, tokens, dim]"
origin: "Temporal Mix of Experts (TMOE) of TimeExpert, arXiv 2509.23145 (2025); local implementation in TSFLab"
origin_models: ["timeexpert"]
tags: ["attention", "expert", "mixture-of-experts", "routing", "top-k", "sparse", "stateful-debug"]
---

# topk_expert_attention

## Purpose

Self-attention in which each key/value position is a candidate "expert". Per
head, for query `q_i`: router logits `l_ij = scale * q_i . k_j`; keep the `topk`
largest, `w = softmax(top-k logits)`; gather `k_j, v_j` of the selected
positions and multiply both by `w_ij` (`gather_experts`); then ordinary softmax
attention over the `topk` gathered pairs, `out_i = sum_j softmax(scale q_i . (w k_j))_j (w v_j)`.
With `shared=True` one extra expert, the mean key and mean value over all
tokens, is appended to every query's set. `topk = 0` falls back to full softmax
attention (plus the shared expert if enabled). Before projection a depthwise
`Conv1d(k=3, padding=1)` over time is added residually to `x`.

## Origin and granularity

Added with TimeExpert in the automated model intake (`d73cf9d0`); the only
consumer is `timeexpert`, which wraps it in `TMOEBlock` (attention plus
pre-norm feed-forward) over patch tokens. The cut contains the router, the
gather helper and the attention operator; patching, residual/FFN blocks, and
the head stay in the model. The model paper and `timeexpert` README are the
source for the method; I could not verify the official-code details beyond
what `timeexpert` states.

## Interface

`LocalExpertRouter(qk_dim, topk, scale=None)`: `forward(query, key)` with
`[n, m, c]` tensors returns `(weight, index)` each `[n, m, topk]`;
`scale` defaults to `qk_dim ** -0.5`. `torch.topk` raises `RuntimeError` if
`topk > m`.

`gather_experts(index, weight, kv)`: `index, weight: [n, m, topk]`, `kv: [n, m, c]`
-> `[n, m, topk, c]`, selected rows already multiplied by `weight`.

`TopKExpertAttention(dim, num_heads=8, topk=4, shared=False, qk_dim=None, dropout=0.0)`:
`dim % num_heads == 0` else `ValueError`; `qk_dim` defaults to `dim` (the
per-head qk width `qk_dim // num_heads`; not validated for divisibility);
`topk >= 0` (and `<= tokens` at call time, else `RuntimeError`); `dropout` is
applied to the attention weights. `forward(x)`: float `[batch, tokens, dim]`
-> same shape. Parameters/state-dict keys: `positional.weight/bias`
(depthwise conv), `qkv.weight/bias` (`dim -> 2*qk_dim + dim`), `out_proj.weight/bias`;
no `router` parameters (none exist), and no router at all when `topk == 0`.

Statefulness: each forward with `topk > 0` stores the router weights in the
plain attribute `last_route_weight` `[batch*heads, tokens, topk]` (not in the
state dict); it keeps the autograd graph alive until overwritten. No masks,
non-causal. Router weights sum to 1 over `topk` (tested), while the shared
expert is added after routing so attention weights over the `topk+1` set are
re-softmaxed.

## Invariants and equivalence evidence

- `tests/test_timeexpert_structure.py`: routing weights sum to one with `topk=2,
  shared=True`; `topk=0` falls back to full attention with finite output;
  forward/backward gives finite gradients for every parameter (including
  `qkv`, `positional`, `out_proj`) and a strict state-dict round trip.
- no fixture: no numeric fixture against the official TimeExpert code.

## Variants and options

`topk` (sparsity), `shared` (global mean expert), `qk_dim` (router/query-key
width), `dropout`. `topk=0` is the dense baseline.

## When to use and when not to use

Use for patch-token encoders that want sparse, data-dependent attention with
a global fallback token. Do not use for causal attention, long sequences where
the dense `[tokens, tokens]` router logits are too large (the router still
forms the full score matrix, so there is no memory saving), or when `tokens <
topk`.

## Related components

`self_attention_family` (dense and ProbSparse cores), `differential_attention`,
`global_patch_compression_attention`, `soft_tree` (other differentiable
routing).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `TopKExpertAttention(dim: int, num_heads: int=8, topk: int=4, shared: bool=False, qk_dim: int | None=None, dropout: float=0.0)`
  Adaptive local self-attention over the top-``topk`` scoring positions.

```python
from tsflab.models._components.topk_expert_attention import TopKExpertAttention
```

## Retrieval terms

`attention`, `expert`, `mixture-of-experts`, `routing`, `top-k`

## Current model consumers (1)

`timeexpert`
<!-- component-card:generated:end -->
