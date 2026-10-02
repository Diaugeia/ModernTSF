---
name: "masking"
kind: "component"
module: "tsflab.models._components.masking"
summary: "Boolean attention-mask holders: TriangularCausalMask, ProbMask (ProbSparse query rows), and LocalMask (causal band); True marks a position to be filled with -inf."
category: "attention"
input: "construction args: batch_size, length (and num_heads, index, scores for ProbMask; series_length for LocalMask), device"
output: "TriangularCausalMask.mask [batch, 1, length, length] bool; ProbMask.mask same shape as scores [batch, heads, n_selected_queries, key_length] bool; LocalMask.mask [batch, 1, length, series_length] bool"
origin: "Mask helpers of the Informer/Time-Series-Library lineage (Informer, Zhou et al., AAAI 2021, for ProbMask); original authorship of LocalMask is not recorded in history"
origin_models: ["informer", "transformer"]
tags: ["attention", "causal", "mask", "probsparse", "boolean", "stateless", "local-band", "triangular"]
---

# masking

## Purpose

Three small classes that build boolean masks of the form used with
`scores.masked_fill_(mask, -inf)`: `True` means "may not attend".

- `TriangularCausalMask(batch_size, length, device)`: strict upper triangle
  (`j > i` masked), so position `i` sees `0..i`.
- `ProbMask(batch_size, num_heads, length, index, scores, device)`: the causal
  mask rows only for the selected (top-sparsity) queries `index`, as needed by
  ProbSparse attention.
- `LocalMask(batch_size, length, series_length, device)`: causal and limited to
  the previous `ceil(log2(length))` positions (band mask).

Each exposes the tensor through the `.mask` property.

## Origin and granularity

The classes came with the initial repository commit (`0b1fbf2e`, `models/module/masking.py`)
and match the long-standing Informer / Time-Series-Library mask helpers.
They were kept as a shared component because `self_attention_family` needs them
for full attention (`TriangularCausalMask`) and ProbSparse attention
(`ProbMask`), which serve `transformer`, `informer` and `dualformer` through that
component. `LocalMask` has no consumer in `src` (only the contract test builds
it) and its origin is not recorded. No model imports `masking` directly, which is
why the generated block lists no model consumers. Mask use stays model-local:
which attention flags it, where `-inf` is applied, and padding/validity masks.

## Interface

`TriangularCausalMask(batch_size, length, device="cpu")`: `.mask` is bool
`[batch_size, 1, length, length]`.

`ProbMask(batch_size, num_heads, length, index, scores, device="cpu")`:
`length` is the number of query rows, `index` an integer tensor
`[batch_size, num_heads, n_selected]` of selected query rows (values below
`length`), `scores` `[batch_size, num_heads, n_selected, key_length]` (only its
shape is used). The mask for selected row `index` masks keys with position
greater than the row id. `.mask` is bool with the shape of `scores`.

`LocalMask(batch_size, length, series_length, device="cpu")`: `.mask` is bool
`[batch_size, 1, length, series_length]`; position `i` may attend to keys in
`[i - ceil(log2(length)), i]` (strict future and far past masked). Assumes
`series_length >= length` for a square-diagonal meaning.

General notes: plain Python classes (no `__call__`), not `nn.Module`s: no
parameters, buffers, state-dict keys, or training state. They do not validate
arguments; the mask is rebuilt on every construction (allocated on CPU, then
moved to `device`) and exposed read-only through `.mask`. The result is a bool
tensor and never requires grad. The generated block shows a module-level import
because the catalog spec lists no public symbols; import the three classes by name.

## Invariants and equivalence evidence

- `tests/test_component_contracts_basic.py` (`test_triangular_causal_mask`,
  `test_prob_mask`, `test_local_mask_and_reference`): bool dtype and shapes; the
  causal mask equals `triu(ones, 1)` and does not require grad; every `ProbMask`
  row equals `arange(key_length) > row id` for the selected rows; `LocalMask` is
  masked exactly where `j > i or j < i - ceil(log2(length))` with an unmasked
  diagonal. The fixture `tests/fixtures/components/masking.pt` pins a
  5-position causal mask and the 8-position local mask (not `ProbMask`).
- `tests/test_component_contracts_attention.py` applies `TriangularCausalMask`
  inside its `self_attention_family` checks; `transformer`, `informer` and
  `dualformer` exercise `TriangularCausalMask` and `ProbMask` through the model
  contract tests (indirect coverage).

## Variants and options

Three classes for three masking patterns. Missing: padding masks, non-causal
block masks, and graph-distance masks (see `graph_masked_attention`).

## When to use and when not to use

Use for decoder-style causal self-attention or ProbSparse attention in the
Informer-lineage layers of `self_attention_family`. Do not use for padding or
validity masks, for additive float masks (these are boolean fill masks), or in
code that expects `nn.Module` behaviour (device moves, state dict).

## Related components

`self_attention_family` (the consumer; masks are passed as its `attn_mask`
objects), `transformer_encdec` (encoder/decoder blocks that carry those masks),
`graph_masked_attention` (graph-structure masks), `node_visibility` (random
token masking and subgraph grouping, a different mechanism). The differential
attention in `differential_attention` and the routed attention in
`topk_expert_attention` take no mask object.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import tsflab.models._components.masking
```

## Retrieval terms

`attention`, `causal`, `mask`

## Current model consumers (0)

No model currently declares this component directly.
<!-- component-card:generated:end -->
