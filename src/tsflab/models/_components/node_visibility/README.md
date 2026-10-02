---
name: "node_visibility"
kind: "component"
module: "tsflab.models._components.node_visibility"
summary: "Token-axis helpers: random keep/drop mask, per-sample shuffle and its inverse, and zero-padded grouping into fixed-size subgraphs and its inverse."
category: "utility"
input: "tokens [batch, length, dim] (ungroup takes [batch * num_groups, subgraph_size, dim_out])"
output: "kept [batch, keep_length, dim] + keep_indices [keep_length]; shuffled [batch, length, dim] + perm [batch, length]; grouped [batch * num_groups, subgraph_size, dim] + num_groups + orig_length"
origin: "Node-visibility mechanism of VisiFold (ICDE 2026, 'Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility')"
origin_models: ["visifold"]
tags: ["graph", "grouping", "masking", "node", "sampling", "subgraph", "visibility", "token"]
---

# node_visibility

## Purpose

Reduces the cost of all-pairs attention over `L` nodes (or any tokens) during
training. The five functions compose into a pipeline on `[B, L, D]` tokens:

1. `random_mask_tokens`: keep `max(1, int(L * (1 - mask_ratio)))` random positions
   (sorted, shared across the batch).
2. `shuffle_tokens`: independent random permutation per sample.
3. `group_into_subgraphs`: right zero-pad to a multiple of `subgraph_size` and fold
   groups into the batch axis, so attention only spans `subgraph_size` tokens.
4. (model-local attention/head on the grouped tokens.)
5. `ungroup_subgraphs` then `unshuffle_tokens` invert 3 and 2.

## Origin and granularity

Added in commit `6663e2e0` (automated intake of VisiFold, Extralonger, ST-SSDL,
RAGC) as a new shared component extracted from the `visifold` port; there was no
earlier duplicated code to consolidate. The functions are generic over what `L` indexes
(the docstring says nodes, patches, or any tokens), but `visifold` is the only consumer.
Kept model-local in `visifold`: the token construction, the `self.training` gating (the
pipeline runs only in training), the attention layers and output head, scattering
predictions for dropped nodes back to zero in the full `[B, N, pred_len]` output, and
restricting the loss to `keep_indices`.

## Interface

All five are module-level functions in `__all__`; no parameters, buffers or state. Random
sampling uses the global torch RNG unless a `generator` is passed.

- `random_mask_tokens(x, mask_ratio, generator=None) -> (kept, keep_indices)`:
  `x [B, L, D]`; `mask_ratio` in `[0, 1)` else `ValueError`. `keep_indices` is a 1-D
  long tensor `[keep_length]` of sorted positions shared by all batch rows. With
  `mask_ratio <= 0` returns `x` unchanged and `arange(L)`.
- `shuffle_tokens(x, generator=None) -> (shuffled, perm)`: `perm [B, L]` gives, per row,
  the source index placed at each output position.
- `unshuffle_tokens(x, perm) -> Tensor`: inverse via `argsort(perm)`.
- `group_into_subgraphs(x, subgraph_size) -> (grouped, num_groups, orig_length)`:
  `subgraph_size > 0` else `ValueError`. If `L <= subgraph_size`, returns `(x, 1, L)`
  unchanged; otherwise `num_groups = ceil(L / subgraph_size)` and
  `grouped [B * num_groups, subgraph_size, D]`, zero padding at the end of the token axis
  (padding tokens are attended to unless the caller masks them).
- `ungroup_subgraphs(x, num_groups, orig_length, subgraph_size) -> Tensor`: returns `x`
  unchanged when `orig_length <= subgraph_size`, else regroups and drops padding
  to `[B, orig_length, D']`; `D'` may differ from the grouping width.

## Invariants and equivalence evidence

- `tests/test_component_contracts_graph.py` checks the Interface shapes, dtype, errors, invariants, gradient flow, and seeded numerical regression against `tests/fixtures/components/node_visibility.pt`.
- `tests/test_local_graph_forecasters.py`
  (`test_visifold_node_visibility_masks_and_regroups_during_training`) runs `visifold` in
  training mode through the full pipeline and checks output shape `(2, 3, 4)` and
  finiteness; the same file asserts `subgraph_size` wiring.
- no fixture: there is no `.pt` fixture and no dedicated round-trip test of
  `shuffle/unshuffle` or `group/ungroup`; round-trip behaviour follows from the
  code (`argsort` inverse; reshape/slice) and is only exercised through `visifold`.

## Variants and options

- `mask_ratio=0` disables masking but keeps the index return value.
- `subgraph_size >= L` disables grouping.
- Pass a seeded `torch.Generator` for reproducible masks/shuffles (it must live on the
  same device as `x`).
- Masking is shared across the batch whereas shuffling is per sample; this asymmetry is
  part of the current behaviour.

## When to use and when not to use

Use for training-time subsampling and subgraph partitioning of node tokens before
quadratic attention. Do not use when the pipeline must also run at inference (nothing
here is gated on training; the caller must gate it), when node order or graph structure
matters inside a group (groups are random subsets and no adjacency is carried), or when
padding tokens must not take part in attention.

## Related components

`graph_masked_attention` (dense/local node attention that benefits from smaller sets),
`self_attention_family`, `sparse_connection_router` (alternative learned sparsification).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `random_mask_tokens(x: torch.Tensor, mask_ratio: float, generator: torch.Generator | None=None)`
  Randomly keep a ``(1 - mask_ratio)`` fraction of tokens along dim 1.
- `shuffle_tokens(x: torch.Tensor, generator: torch.Generator | None=None)`
  Apply an independent random per-sample permutation along dim 1.
- `unshuffle_tokens(x: torch.Tensor, perm: torch.Tensor)`
  Invert :func:`shuffle_tokens` given the permutation it returned.
- `group_into_subgraphs(x: torch.Tensor, subgraph_size: int)`
  Partition tokens into zero-padded fixed-size subgraph groups.
- `ungroup_subgraphs(x: torch.Tensor, num_groups: int, orig_length: int, subgraph_size: int)`
  Invert :func:`group_into_subgraphs`, dropping any padding tokens.

```python
from tsflab.models._components.node_visibility import random_mask_tokens, shuffle_tokens, unshuffle_tokens, group_into_subgraphs, ungroup_subgraphs
```

## Retrieval terms

`graph`, `grouping`, `masking`, `node`, `sampling`, `subgraph`, `visibility`

## Current model consumers (1)

`visifold`
<!-- component-card:generated:end -->
