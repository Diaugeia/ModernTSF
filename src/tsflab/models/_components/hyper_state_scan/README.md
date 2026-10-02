---
name: "hyper_state_scan"
kind: "component"
module: "tsflab.models._components.hyper_state_scan"
summary: "Scalar-state (d_state=1) selective scan returning the raw state trajectory, plus a depthwise 2-D conv that mixes that state over a (patch, variate) grid."
category: "state-space"
input: "diagonal_selective_scan: u, delta, b [batch, channels, length]; a [channels]; GridStateMixer: [batch, channels, height, width]"
output: "diagonal_selective_scan: [batch, channels, length] state trajectory; GridStateMixer: [batch, channels, height, width]"
origin: "TimePro hyper-state, 'TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state' (arXiv 2505.20774, 2025), built on the Mamba selective scan; GridStateMixer is a portable stand-in for DCNv4-style deformable convolution"
origin_models: ["timepro"]
tags: ["grid", "hyper-state", "mamba", "scan", "ssm", "state-space", "depthwise", "scalar-state"]
---

# hyper_state_scan

## Purpose

`diagonal_selective_scan(u, delta, a, b)` evaluates, per batch and channel, the
scalar recurrence `h_t = exp(delta_t * a) * h_{t-1} + delta_t * b_t * u_t` with
`h_0 = 0` and returns every `h_t`. Unlike the `mamba` scan it does not apply the
`C` read-out or the `D` skip, so the caller can edit the raw state first.
`GridStateMixer` applies a depthwise `Conv2d` over a channel-first grid to mix each
channel's state with its spatial neighbours.

## Origin and granularity

Added with `timepro` (automated intake), which is the only consumer. TimePro scans over the
variate axis with a scalar state, reshapes the state to a (time-patch, variate)
grid (patch-position height, variate width, patch features as channels), mixes
it locally, then reads it out. The module docstring states that
`GridStateMixer` replaces offset-based deformable convolution (DCNv4, needs a
custom CUDA kernel) by a fixed local receptive field, losing the learned sampling
offsets. Origin of the scan: Mamba (see `mamba`); the cut isolates only the two
paper-neutral pieces. Model-local in `timepro`: projections, `dt_rank`, `A` and `D`
parameters, the `C` read-out, the grid reshape, normalization and bidirectionality.

## Interface

`diagonal_selective_scan(u, delta, a, b) -> Tensor`: `u`, `delta`, `b` all
`[batch, channels, length]` with identical shapes and `length >= 1` (an empty length axis fails in the initial-state indexing); `a` is `[channels]` (already
negative, i.e. `-exp(a_log)`). Returns `[batch, channels, length]`; `h[..., t]` is
the state after position `t`. Raises `ValueError` if `u`, `delta`, `b` shapes differ or
`a` is not 1-D with length `channels`. Pure function; `u`, `delta`, `a`, `b` must share dtype and device; sequential Python loop over
`length`, differentiable, no parameters.

`GridStateMixer(channels, kernel_size=3)`: `channels >= 1`, `kernel_size` positive
odd (else `ValueError`). Holds `self.conv`, a depthwise `Conv2d` with `groups=channels`,
`padding=kernel_size//2`, so the spatial size is preserved; state-dict keys
`conv.weight` `[channels, 1, k, k]` and `conv.bias` `[channels]`. `forward(grid)`
requires `ndim == 4` `[batch, channels, height, width]` (else `ValueError`);
height and width are caller-defined. Zero padding at the grid border; channels
never mix with one another.

## Invariants and equivalence evidence

- `tests/test_2025_query_gate_hyperstate_forecasters.py`:
  `test_diagonal_selective_scan_matches_manual_recurrence` checks the scan against
  an explicit loop; `test_grid_state_mixer_preserves_grid_shape` checks the shape;
  `test_timepro_forward_and_gradient` runs both inside `timepro` (with `d_state=1`).
- `tests/test_component_contracts_signal.py` (`test_diagonal_scan_recurrence_and_shape`,
  `test_diagonal_scan_errors_and_grad`, `test_grid_state_mixer`) checks shape and
  dtype, the recurrence step by step, zero input giving zero state, `ValueError` for
  mismatched `u`/`b` shapes and a wrong-length `a`, gradients to `u` and `a`, the
  mixer's state-dict keys and `groups == channels`, channel independence
  (depthwise), and invalid `(channels, kernel_size)` pairs. Reference values:
  `tests/fixtures/components/hyper_state_scan.pt` (scan) and
  `tests/fixtures/components/hyper_state_scan_mixer.pt` (mixer).
- Equivalence with the `mamba` scan at `d_state=1` is not tested.

## Variants and options

Only `kernel_size` for the mixer. Not provided: multi-state (`d_state > 1`) scan
(use `mamba`), parallel or fused scan, learned sampling offsets or deformable
sampling, non-depthwise mixing.

## When to use and when not to use

Use when a Mamba-style model keeps one decay per channel and needs the raw state
trajectory to modify (for example spatially mix) before reading it out. Do not use
for `d_state > 1`, when a read-out and skip inside the scan are wanted (use
`mamba`), for very long lengths (Python loop), or when true deformable sampling
is required.

## Related components

`mamba` (multi-state kernel-free selective scan with `C` read-out and `D` skip; this
module is the `d_state=1`, raw-state variant, not a replacement), `diffusion_conv`
and `graph_masked_attention` (other ways to mix over a variate axis, but with
supports or masks rather than a local conv grid).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `diagonal_selective_scan(u: torch.Tensor, delta: torch.Tensor, a: torch.Tensor, b: torch.Tensor)`
  Evaluate a scalar-state selective scan over the trailing length axis.
- `GridStateMixer(channels: int, kernel_size: int=3)`
  Depthwise 2-D convolution mixing a channel-first state grid locally.

```python
from tsflab.models._components.hyper_state_scan import diagonal_selective_scan, GridStateMixer
```

## Retrieval terms

`grid`, `hyper-state`, `mamba`, `scan`, `ssm`, `state-space`

## Current model consumers (1)

`timepro`
<!-- component-card:generated:end -->
