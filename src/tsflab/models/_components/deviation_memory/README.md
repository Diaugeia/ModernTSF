---
name: "deviation_memory"
kind: "component"
module: "tsflab.models._components.deviation_memory"
summary: "Learnable prototype bank with softmax-attention retrieval and nearest/second-nearest lookup, plus an L1 deviation score between two representations."
category: "memory"
input: "PrototypeMemory: h [..., query_dim]; deviation_score: current [..., D]; reference [..., D]"
output: "PrototypeRetrieval fields value/query/nearest/second_nearest [..., prototype_dim] and indices [..., 2]; deviation_score: [...]"
origin: "prototype memory and deviation learning of ST-SSDL, 'How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning' (NeurIPS 2025, arXiv 2510.04908)"
origin_models: ["st_ssdl"]
tags: ["contrastive", "deviation", "memory", "prototype", "retrieval", "self-supervised"]
---

# deviation_memory

## Purpose

`PrototypeMemory` holds `M` learnable prototypes `P: [M, d]` and a query projection
`Wq: [query_dim, d]`. For `h: [..., query_dim]`:
`q = h @ Wq`; `s = softmax(q @ P^T, -1)`; `value = s @ P`; the indices of the
top-2 scores give `nearest` and `second_nearest` (rows of `P`).
`deviation_score(current, reference) = sum_D |current - reference|`, one
non-negative scalar per leading position.

## Origin and granularity

Added with `st_ssdl` (commit `6663e2e0`, "add VisiFold, Extralonger, ST-SSDL,
RAGC", automated intake) as the paper-neutral part of its self-supervised deviation
mechanism; `st_ssdl` is the only consumer. Model-local: the Chebyshev graph-GRU
encoder, the triplet margin loss on (query, nearest, second_nearest), the L1
`deviation_loss` between latent and prototype deviations, and the pairing of a
current and a historical window. The component names no spatiotemporal axis:
`h` may carry node, batch or any leading shape.

## Interface

`PrototypeMemory(query_dim, prototype_dim, num_prototypes)`: all >= 1 and
`num_prototypes >= 2` (needed for top-2), else `ValueError`. Parameters (the only
state-dict keys): `prototypes` `[num_prototypes, prototype_dim]` and `query_proj`
`[query_dim, prototype_dim]`, both Xavier-normal. No buffers. `forward(h)` requires
`h.shape[-1] == query_dim` (`ValueError` otherwise) and returns a
`PrototypeRetrieval` dataclass (`value`, `query`, `nearest`, `second_nearest`,
`indices`; trailing axis `prototype_dim`, or 2 for `indices`). `query` and `value`
carry gradients; `nearest`/`second_nearest` are gathered by index, so gradients reach
the selected prototype rows but not the selection itself. Stateless, no memory
update rule: prototypes are trained only by whatever loss the caller builds.

`deviation_score(current, reference) -> Tensor`: last axes must match (`ValueError`),
leading axes broadcast; returns `[...]`. Quirk: the docstring says "mean absolute
deviation" but the code sums over the last axis (`.sum(dim=-1)`), so the value
scales with `D`; the card's formula follows the code.

## Invariants and equivalence evidence

- no fixture: no tensor fixture exists.
- `tests/test_local_graph_forecasters.py`
  (`test_st_ssdl_prototype_memory_retrieval_shapes`) pins all five output shapes for
  `h: [B, N, query_dim]`, `deviation_score(x, x) == 0` exactly and
  `deviation_score >= 0`. The same file checks `st_ssdl`'s contrastive and deviation
  losses are finite and that the deviation loss is about 0 for an identical history.

## Variants and options

Only the three sizes. No temperature on the softmax, no top-k other than 2, no
hard-assignment or EMA prototype update, no cosine scoring (the score is a raw dot
product of the projected query with prototypes).

## When to use and when not to use

Use for an auxiliary "how far from the learned normal patterns" signal on any
encoder state, with a triplet-style loss over nearest/second-nearest prototypes.
Do not use when a mean-normalized or other distance is needed (`deviation_score` is
an unnormalized L1 sum), when prototype updates should be non-gradient, or as a
general retrieval store over real training samples (see `periodic_query_bank`).

## Related components

`periodic_query_bank` (learned bank queried by phase), `regularized_adaptive_graph_conv`
(extracted in the same intake batch).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `PrototypeMemory(query_dim: int, prototype_dim: int, num_prototypes: int)`
  Learnable prototype bank with soft attention retrieval and top-2 lookup.
- `PrototypeRetrieval()`
  Outputs of one :class:`PrototypeMemory` query.
- `deviation_score(current: torch.Tensor, reference: torch.Tensor)`
  Mean absolute deviation between two same-shaped representations.

```python
from tsflab.models._components.deviation_memory import PrototypeMemory, PrototypeRetrieval, deviation_score
```

## Retrieval terms

`contrastive`, `deviation`, `memory`, `prototype`, `retrieval`, `self-supervised`

## Current model consumers (1)

`st_ssdl`
<!-- component-card:generated:end -->
