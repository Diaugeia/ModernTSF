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
tags: ["attention", "expert", "mixture-of-experts", "routing", "top-k", "sparse", "sparse-attention", "last-route-weight", "temporal-moe"]
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
the head stay in the model. The `timeexpert` README (paper and pinned official code) is the
source for the method; this card makes no claim about official-code details beyond it.
`LocalExpertRouter` is not the same thing as `topk_expert_router` (see Related components).

## Interface

`LocalExpertRouter(qk_dim, topk, scale=None)` (no parameters; `qk_dim` only sets the default scale): `forward(query, key)` with
`[n, m, c]` tensors returns `(weight, index)` each `[n, m, topk]`;
`scale` defaults to `qk_dim ** -0.5`. `torch.topk` raises `RuntimeError` if
`topk > m`.

`gather_experts(index, weight, kv)`: `index, weight: [n, m, topk]`, `kv: [n, m, c]`
-> `[n, m, topk, c]`, selected rows already multiplied by `weight`.

`TopKExpertAttention(dim, num_heads=8, topk=4, shared=False, qk_dim=None, dropout=0.0)`:
`dim % num_heads == 0` else `ValueError`; `qk_dim` defaults to `dim` (per-head qk width is
`qk_dim // num_heads`; divisibility is not validated, so a non-divisible `qk_dim` fails later in
`forward` with a view `RuntimeError`);
`topk >= 0` (a negative value is not rejected and silently behaves like `topk=0`, dense attention; `topk <= tokens` is required at call time, else `RuntimeError` from `torch.topk`); `dropout` in `[0, 1]` is
applied to the attention weights after the softmax over the selected set. `forward(x)`: float `[batch, tokens, dim]`
-> same shape (dtype and device follow the parameters; no padding or mask). Parameters/state-dict keys: `positional.weight/bias`
(depthwise conv), `qkv.weight/bias` (`dim -> 2*qk_dim + dim`), `out_proj.weight/bias`;
no `router` parameters (none exist), and no router at all when `topk == 0`.

Statefulness: each forward with `topk > 0` stores the router weights in the
plain attribute `last_route_weight` `[batch*heads, tokens, topk]` (not in the
state dict); it keeps the autograd graph alive until overwritten. No masks,
non-causal. Router weights sum to 1 over `topk` (tested), while the shared
expert is added after routing so attention weights over the `topk+1` set are
re-softmaxed.

## Invariants and equivalence evidence

- `tests/test_component_contracts_attention.py` (`test_local_expert_router_and_gather`,
  `test_topk_expert_attention` for `(topk, shared)` in `{(2, T), (2, F), (0, F), (0, T)}`,
  `test_topk_expert_attention_errors`) checks: router state dict empty, `[n, m, topk]` shapes,
  weights sum to 1, int64 indices in range, default scale, `gather_experts` rows equal to
  `w * kv[idx]`, `RuntimeError` for `topk > tokens`; attention state-dict keys, output
  shape and dtype, `last_route_weight` shape `[batch*heads, tokens, topk]` with unit sums
  (`None` and no `router` attribute when `topk == 0`), finite input and parameter gradients,
  and the `dim % num_heads` `ValueError`. Seeded regressions use
  `tests/fixtures/components/topk_expert_attention_k0_plain.pt`,
  `tests/fixtures/components/topk_expert_attention_k0_shared.pt`,
  `tests/fixtures/components/topk_expert_attention_k2_plain.pt`,
  `tests/fixtures/components/topk_expert_attention_k2_shared.pt` and
  `tests/fixtures/components/topk_expert_attention_router.pt`.
- `tests/test_timeexpert_structure.py` (`test_timeexpert_routes_queries_to_topk_local_experts`,
  `test_topk_zero_falls_back_to_full_attention`,
  `test_forward_backward_active_parameters_and_round_trip`): routing weights sum to one with `topk=2,
  shared=True`; `topk=0` falls back to full attention with finite output;
  forward/backward gives finite gradients for every parameter (including
  `qkv`, `positional`, `out_proj`) and a strict state-dict round trip.
- No fixture compares against the official TimeExpert code; the fixtures above are
  regressions of this implementation only.

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

`self_attention_family` (dense and ProbSparse cores: ProbSparse keeps a subset of queries,
whereas here every query keeps its own top-k keys and re-weights them), `differential_attention`
and `global_patch_compression_attention` (dense, no routing), `topk_expert_router` (a gating
MLP plus top-k mixing of independent expert outputs, unlike this in-attention key/value
selection), `sparse_connection_router` (input-independent learned top-k connectivity, not
per-query), `soft_tree` (other differentiable routing).
- `graph_masked_attention`: sparsifies attention with a fixed graph rather than learned routing.

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
