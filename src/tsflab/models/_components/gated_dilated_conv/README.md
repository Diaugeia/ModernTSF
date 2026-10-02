---
name: "gated_dilated_conv"
kind: "component"
module: "tsflab.models._components.gated_dilated_conv"
summary: "Left-pad for a causal dilated convolution and the WaveNet gated activation unit tanh(filter(x)) * sigmoid(gate(x)) over caller-owned conv modules."
category: "convolution"
input: "causal_pad: [batch, channels, time] or [batch, channels, nodes, time]; gated_dilated_conv: same layouts, with caller-owned filter and gate Conv1d/Conv2d"
output: "causal_pad: same rank, time axis longer by dilation*(kernel_size-1); gated_dilated_conv: filter conv output, time length unchanged"
origin: "WaveNet gated activation unit, van den Oord et al., 2016 (arXiv 1609.03499), as used in Graph WaveNet (IJCAI 2019) and MTGNN"
origin_models: ["wavenet", "gwnet", "dfdgcn", "mtgnn"]
tags: ["causal", "dilated", "gate", "gated-activation", "wavenet", "convolution"]
---

# gated_dilated_conv

## Purpose

`causal_pad(x, dilation, kernel_size)` zero-pads `dilation * (kernel_size - 1)`
steps on the left of the last (time) axis only. `gated_dilated_conv(x, filter_conv,
gate_conv)` pads `x` this way, using `filter_conv`'s last-axis `kernel_size` and
`dilation`, and returns `tanh(filter_conv(p)) * sigmoid(gate_conv(p))` with
`p = causal_pad(x)`. If the convs have no padding, the output has the same time
length as `x` and position `t` depends only on inputs up to `t`.

## Origin and granularity

Extracted in commit `dd8d8196` ("gated_dilated_conv and
adaptive_node_embedding_adjacency") from near-identical copies in `wavenet`,
`gwnet`, `dfdgcn` and `mtgnn`, with equivalence tests for all four. The cut is
only pad plus gated product. Model-local: the conv modules themselves (their
parameters and state-dict names stay on each calling layer), residual and skip
convolutions, graph mixing, post-gate normalization, and the adjacency (MTGNN's
adjacency is a different operator and stays local).

## Interface

`causal_pad(x, dilation, kernel_size) -> Tensor`: `x` rank 3
`[batch, channels, time]` or rank 4 `[batch, channels, nodes, time]`; any other
rank raises `ValueError`. `dilation` and `kernel_size` are ints >= 1. Pads the last
axis on the left only.

`gated_dilated_conv(x, filter_conv, gate_conv) -> Tensor`: `filter_conv` and
`gate_conv` are caller-owned `nn.Conv1d` (rank-3 `x`) or `nn.Conv2d` (rank-4 `x`,
kernel such as `(1, k)`, dilation `(1, d)`) modules. Only `filter_conv.kernel_size[-1]`
and `filter_conv.dilation[-1]` are read for padding; `gate_conv` must use the same
last-axis kernel and dilation and the same output channels (not checked). Their own
`padding` should be 0 along time, otherwise the result is not causal. Output
channels are those of the convs; the time length equals the input's. No parameters,
buffers or state of its own; no dtype handling beyond the convs'.

## Invariants and equivalence evidence

- `tests/test_component_contracts_graph.py` (`test_causal_pad_and_gated_conv`) checks
  `causal_pad` shapes for rank 3 and 4, that the left pad is zero and the rest is the
  input, kernel 1 adds no padding, rank 2 raises `ValueError`; `gated_dilated_conv`
  length preservation with `Conv1d` and `Conv2d` `(1, k)`, dtype, `|y| < 1`,
  gradients to input and both convs, causality (perturbing future steps leaves earlier
  outputs unchanged), and a seeded regression against
  `tests/fixtures/components/gated_dilated_conv.pt`.
- `tests/test_component_extraction_graph.py`:
  `test_wavenet_gated_dilated_conv_equivalence`,
  `test_gwnet_gated_dilated_conv_and_adaptive_adjacency_equivalence`,
  `test_dfdgcn_gated_dilated_conv_and_adaptive_adjacency_equivalence` and
  `test_mtgnn_gated_dilated_conv_equivalence` compare each refactored model against
  an in-test copy of the pre-extraction reference (same state-dict names, outputs,
  gradients, atol 1e-6). These references live in the test file; there is no `.pt`
  fixture for them.

## Variants and options

The caller picks `Conv1d` or `Conv2d`, kernel width, dilation (e.g. exponential
across layers) and channel widths. A kernel of 1 yields zero padding. No
non-causal or symmetric padding option, no residual or skip output.

## When to use and when not to use

Use for WaveNet-style temporal gating in a stack of dilated causal convs, on
`[B, C, T]` or `[B, C, N, T]` layouts. Do not use when the module should own its
convolutions, when the filter and gate differ in kernel or dilation, when
non-causal context is wanted, or for other gating forms (GLU, see `gated_fusion`).

## Related components

`diffusion_conv` (the graph-mixing step that follows the gated unit in Graph WaveNet;
it mixes over nodes, this one over time), `adaptive_node_embedding_adjacency` (extracted
in the same commit; supplies the adjacency, not temporal gating), `gated_fusion`
(a learned sigmoid blend of two tensors, not a tanh x sigmoid temporal conv).
`composed` also reaches this component through its slot registry.
- `softmax_gate`: softmax feature gate on one tensor, not a tanh x sigmoid temporal convolution.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `causal_pad(x: torch.Tensor, dilation: int, kernel_size: int)`
  Left-pad the trailing (temporal) axis for a causal dilated convolution.
- `gated_dilated_conv(x: torch.Tensor, filter_conv: nn.Module, gate_conv: nn.Module)`
  Apply the WaveNet gated activation unit to ``x``.

```python
from tsflab.models._components.gated_dilated_conv import causal_pad, gated_dilated_conv
```

## Retrieval terms

`causal`, `dilated`, `gate`, `gated-activation`, `wavenet`

## Current model consumers (5)

`composed`, `dfdgcn`, `gwnet`, `mtgnn`, `wavenet`
<!-- component-card:generated:end -->
