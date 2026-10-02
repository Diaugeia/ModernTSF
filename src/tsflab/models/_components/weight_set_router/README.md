---
name: "weight_set_router"
kind: "component"
module: "tsflab.models._components.weight_set_router"
summary: "Low-rank weight sharing: softmax-with-temperature routing matrix over M weight sets and the per-channel linear mix of those sets."
category: "routing"
input: "WeightSetRouter: temperature float; mix_weight_sets: weights [num_sets, *shape], routing [num_sets, channels]"
output: "WeightSetRouter: [num_sets, channels] convex columns; mix_weight_sets: [channels, *shape]"
origin: "Low-rank Weight Sharing of DiPE-Linear (Zhao et al., arXiv 2411.17257, Eqs. 7-8), inspired by mixture-of-experts and dynamic convolution"
origin_models: ["dipelinear"]
tags: ["low-rank", "routing", "softmax", "temperature", "weight-sharing", "channel-wise", "stateless"]
---

# weight_set_router

## Purpose

`WeightSetRouter(num_sets, channels)` owns a learnable matrix `R` of shape
`[num_sets, channels]` (standard normal initialization) and returns
`R' = softmax(R / tau, dim=0)` for a caller-supplied temperature `tau`, so every
channel receives a convex combination of `num_sets` weight sets.
`mix_weight_sets(weights, routing)` forms `G_c = sum_m R'[m, c] * G_m`, turning
`[num_sets, *shape]` weights into `[channels, *shape]`. With `num_sets` much
smaller than `channels` this interpolates between a fully shared and a fully
per-channel parameterisation.

## Origin and granularity

Extracted from the DiPE-Linear implementation, where the same routing mixes the
frequency attention, temporal attention and the frequential mapping. The
temperature schedule (annealing) is deliberately not part of the component; the
caller decides `tau`.

## Interface

`WeightSetRouter(num_sets, channels)`: sizes must be positive. Parameter
`logits` `[num_sets, channels]`. `forward(temperature=1.0)` requires
`temperature > 0` and returns `[num_sets, channels]` whose columns sum to 1.

`mix_weight_sets(weights, routing)`: `weights` is `[num_sets, ...]`, `routing`
is `[num_sets, channels]`; mismatched sets raise `ValueError`. Returns
`[channels, ...]` and is differentiable in both arguments.

## Invariants and equivalence evidence

- `tests/test_dipelinear.py` (WeightSetRouterTests) checks convexity, the
  high-temperature uniform limit, equality of `mix_weight_sets` with the explicit
  weighted sum and its gradient; reference values in
  `tests/fixtures/components/weight_set_router.pt`.
- `tests/test_dipelinear.py` (test_matches_official_reference_values)
  matches the official router (`softmax(route / tau)` over the expert axis).

## Variants and options

None. A hard assignment is the `tau -> 0` limit; a uniform average the `tau -> inf`
limit. Input-dependent routers belong in `topk_expert_router`.

## When to use and when not to use

Use to share a small number of parameter sets across many channels with a
learned, input-independent assignment. Do not use for input-dependent expert
selection or sparse top-k routing (see `topk_expert_router`), or when the
channel count varies.

## Related components

`fft_extrapolation_conv` (consumes the routing as `mixing`), `topk_expert_router`,
`sparse_connection_router`, `channel_wise_linear`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `WeightSetRouter(num_sets: int, channels: int)`
  Learn a ``(num_sets, channels)`` routing matrix normalized over the sets.
- `mix_weight_sets(weights: torch.Tensor, routing: torch.Tensor)`
  Combine ``(num_sets, *shape)`` weights into ``(channels, *shape)``.

```python
from tsflab.models._components.weight_set_router import WeightSetRouter, mix_weight_sets
```

## Retrieval terms

`low-rank`, `routing`, `softmax`, `temperature`, `weight-sharing`

## Current model consumers (1)

`dipelinear`
<!-- component-card:generated:end -->
