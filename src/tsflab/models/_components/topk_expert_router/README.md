---
name: "topk_expert_router"
kind: "component"
module: "tsflab.models._components.topk_expert_router"
summary: "Two-layer GELU gating MLP with optional trainable noise returning softmax expert weights, and top-k sparsification with a dense-weight floor and renormalization."
category: "routing"
input: "GatingMLP: [*, in_features]; topk_dense_mix: dense weights [*, experts] (distribution on the last axis)"
output: "GatingMLP: [*, experts] softmax weights; topk_dense_mix: same shape, rows sum to 1"
origin: "gate of DUET (KDD 2025, arXiv 2412.10859) plus the shared top-k concentration step of DUET and Dynamic TMoE (ICML 2026, arXiv 2605.20678); noisy gating follows the sparsely-gated MoE idea"
origin_models: ["duet", "dynamic_tmoe"]
tags: ["expert", "gate", "gating", "mixture", "moe", "routing", "sparse", "top-k"]
---

# topk_expert_router

## Purpose

`GatingMLP(in_features, experts, hidden, noisy)` computes
`logits = Linear(GELU(Linear(x)))`; in training mode and when `noisy`, it adds
`randn * softplus(noise_scale)` (per expert) to the logits; it returns
`softmax(logits, -1)`. `topk_dense_mix(w, k, floor)` keeps the `k` largest entries of
`w` (zeroing the rest), adds `floor * w` everywhere and renormalizes:
`(sparse + floor*w) / sum(sparse + floor*w)`. The floor keeps non-selected experts
gradient-alive.

## Origin and granularity

Extracted in commit `cd1b398d` ("topk_expert_router for DUET and DynamicTMoE"),
which says both routers were reproduced bit-for-bit. `GatingMLP` is DUET's
`DistributionalRouter` gate; `topk_dense_mix` is the same arithmetic DUET applied
after its router and DynamicTMoE applied in `routing_weights` (it passes a
configurable `routing_floor`; DUET passes 1e-3). Model-local: the gate input
features (DUET: per-channel mean and std), DynamicTMoE's drift, MMD and memory
logits (it uses only `topk_dense_mix`), expert networks, and MAGE's gate, DUET's
even-kernel moving average and STWave's Haar step, which the commit kept local
with documented reasons.

## Interface

`GatingMLP(in_features, experts, hidden, noisy=True)`. Keys: `network.0.weight`,
`network.0.bias` (`[hidden, in_features]`, `[hidden]`), `network.2.weight`,
`network.2.bias` (`[experts, hidden]`, `[experts]`) and, only when `noisy`,
`noise_scale` `[experts]` (zeros, so initial noise std is softplus(0)=0.693). No
buffers, no constructor validation. `forward(features)` returns softmax over the
last axis, shape `[*, experts]`; noise uses the global torch RNG and only applies
when `self.training` is true (eval is a plain softmax).

`topk_dense_mix(weights, k, floor) -> Tensor`: `weights` is a non-negative dense
distribution on the last axis (typically softmax output); `1 <= k <= experts` (torch
`topk` error otherwise); `floor >= 0` (a `floor` of 0 gives exact hard top-k
renormalization). Differentiable through the kept values and the floor term; the
selection is not. Ties follow `torch.topk`. Stateless, same dtype/device as input.

## Invariants and equivalence evidence

- `tests/fixtures/duet_pre_refactor.pt` and
  `tests/fixtures/dynamic_tmoe_pre_refactor.pt`, driven by
  `tests/test_component_extraction_moe.py`, compare state-dict keys, shapes and
  values, forward outputs and input gradients of both consumers before and after
  extraction.
- The same file proves `topk_dense_mix` equals both original inline formulas term
  for term, rows sum to one, the noisy gate in eval mode equals a plain softmax, and
  state-dict attribute names are preserved.

## Variants and options

`noisy=False` removes `noise_scale` and the noise. `k` and `floor` set sparsity
and gradient leakage. Not covered: load-balancing or auxiliary losses, capacity
limits, token dispatch or exact sparse expert execution (DUET and DynamicTMoE run
every expert densely and mix with the weights), learned `k`.

## When to use and when not to use

Use for expert mixing weights that concentrate on `k` experts but keep a small
dense gradient path. Do not use if non-selected experts must be skipped to save
compute, if an exactly sparse weight vector (zeros) is required with `floor > 0`,
or when the gate needs a different architecture (it is fixed at two layers with GELU).

## Related components

`sparse_connection_router`, `topk_expert_attention`, `freq_band_moe`, `soft_tree`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `GatingMLP(in_features: int, experts: int, hidden: int, noisy: bool=True)`
  Two-layer gate: ``Linear -> GELU -> Linear`` with optional trainable noise.
- `topk_dense_mix(weights: torch.Tensor, k: int, floor: float)`
  Zero non-top-k experts, blend back a ``floor`` of the dense weights, renormalize.

```python
from tsflab.models._components.topk_expert_router import GatingMLP, topk_dense_mix
```

## Retrieval terms

`expert`, `gate`, `gating`, `mixture`, `moe`, `routing`, `sparse`, `top-k`

## Current model consumers (2)

`duet`, `dynamic_tmoe`
<!-- component-card:generated:end -->
