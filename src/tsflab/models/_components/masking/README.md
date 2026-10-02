---
name: "masking"
kind: "component"
module: "tsflab.models._components.masking"
summary: "Boolean attention-mask holders: TriangularCausalMask, ProbMask (ProbSparse query rows), and LocalMask (causal band); True marks a position to be filled with -inf."
category: "attention"
input: "construction args: batch_size, length (and num_heads, index, scores for ProbMask; series_length for LocalMask)"
output: "TriangularCausalMask.mask [batch, 1, length, length] bool; ProbMask.mask same shape as scores [batch, heads, n_selected_queries, key_length] bool; LocalMask.mask [batch, 1, length, series_length] bool"
origin: "Mask helpers of the Informer/Autoformer Time-Series-Library lineage (Informer, Zhou et al., AAAI 2021, for ProbMask); original authorship of LocalMask is not recorded in history"
origin_models: ["informer", "transformer"]
tags: ["attention", "causal", "mask", "probsparse", "boolean", "stateless"]
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
(`ProbMask`), which serve `transformer`, `informer` and `dualformer`. `LocalMask`
has no consumer in `src` or `tests`; it is unused and its origin is not recorded.
Mask use stays model-local: which attention flags it, where `-inf` is applied,
and padding/validity masks.

## Interface

`TriangularCausalMask(batch_size, length, device="cpu")`: `.mask` is bool
`[batch_size, 1, length, length]`.

`ProbMask(batch_size, num_heads, length, index, scores, device="cpu")`:
`length` is the number of query rows, `index` an integer tensor
`[batch_size, num_heads, n_selected]` of selected query rows, `scores`
`[batch_size, num_heads, n_selected, key_length]` (only its shape is used). The
mask for selected row `index` masks keys with position greater than the row id.
`.mask` is bool with the shape of `scores`. `index` must be a tensor on the
same device indexing into the first `length` rows.

`LocalMask(batch_size, length, series_length, device="cpu")`: `.mask` is bool
`[batch_size, 1, length, series_length]`; position `i` may attend to keys in
`[i - ceil(log2(length)), i]` (strict future and far past masked). Assumes
`series_length >= length` for a square-diagonal meaning.

General notes: plain Python classes, not `nn.Module`s: no parameters, buffers,
state-dict keys, or training state. They do not validate arguments; shapes are
rebuilt on every call (allocation on CPU then moved to `device`). The mask is
created under `no_grad` (Triangular and Local) and is a bool tensor.

## Invariants and equivalence evidence

- Contract, invariant, gradient, and seeded numerical-regression tests: `tests/test_component_contracts_basic.py`, reference values in `tests/fixtures/components/masking.pt`.
- no fixture and no dedicated test: nothing in `tests` imports this module
  (`tests/test_local_cats.py` only mentions an unrelated "masking probability").
  Behaviour is covered indirectly through `transformer`, `informer` and
  `dualformer` forward passes in the model contract tests, and was confirmed
  here with a CPU snippet: `TriangularCausalMask(1, 4)` yields the strict upper
  triangle; `ProbMask` with `index = [0, 2]` masks keys `> 0` and `> 2`
  respectively; `LocalMask(1, 8, 8)` masks `j > i` and `j < i - 3`.

## Variants and options

Three classes for three masking patterns. Missing: padding masks, non-causal
block masks, and graph-distance masks (see `graph_masked_attention`).

## When to use and when not to use

Use for decoder-style causal self-attention or ProbSparse attention in the
Informer-lineage layers of `self_attention_family`. Do not use for padding or
validity masks, for additive float masks (these are boolean fill masks), or in
code that expects `nn.Module` behaviour (device moves, state dict).

## Related components

`self_attention_family` (the consumer), `graph_masked_attention`,
`node_visibility` (random token masking), `transformer_encdec`.

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
