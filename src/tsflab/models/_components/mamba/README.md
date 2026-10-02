---
name: "mamba"
kind: "component"
module: "tsflab.models._components.mamba"
summary: "Pure-PyTorch selective state-space (Mamba) mixer with causal depthwise conv and sequential scan, plus RMSNorm and a pre-norm residual wrapper."
category: "state-space"
input: "[batch, length, d_model]"
output: "MambaBlock, MambaResidualBlock and RMSNorm: same shape as input [batch, length, d_model]"
origin: "Mamba selective state space model, Gu and Dao, 2023 (arXiv 2312.00752); the portable scan follows the MambaSimple port in the Time-Series-Library (thuml)"
origin_models: ["mambasimple", "s_mamba", "bimamba"]
tags: ["mamba", "mixer", "rmsnorm", "ssm", "state-space", "selective-scan", "causal", "kernel-free"]
---

# mamba

## Purpose

`MambaBlock` is a Mamba mixer that needs no `mamba_ssm` or CUDA kernel. For input
`x: [B, L, d_model]` it computes:

1. `in_proj` (no bias) to `d_inner` signal `x'` and `d_inner` gate `res`.
2. A causal depthwise `Conv1d` (kernel `d_conv`, `padding=d_conv-1`, output cropped to `L`)
   over time on `x'`, then SiLU.
3. `ssm`: `A = -exp(A_log)` (`[d_inner, d_state]`); `x_proj` yields `delta_raw`
   (`dt_rank`), `B`, `C` (each `d_state`) per step; `delta = softplus(dt_proj(delta_raw))`;
   `selective_scan` runs `h_t = exp(delta_t A) * h_{t-1} + delta_t B_t u_t`,
   `y_t = C_t . h_t + D * u_t` with a Python loop over `t`.
4. `out_proj(y * SiLU(res))`.

`MambaResidualBlock` is `x + MambaBlock(RMSNorm(x))`. `RMSNorm` is
`x * rsqrt(mean(x^2, -1) + eps) * weight`.

## Origin and granularity

The block equations are those of Mamba (Gu and Dao, 2023). The repository's first
copy was model-local in `mambasimple` (commit `43d54934`, "dependency-free Mamba,
manual selective scan, no mamba_ssm kernels", from TSLib's MambaSimple); it was
moved to this component and shared by `s_mamba` and `bimamba` (commit `3ac9e264`
then `33ea2050` colocated it under the components package). The boundary is the
mixer plus its normalization and residual wrapper. Model-local: tokenization or
inverted embedding (`s_mamba`), forward/backward fusion, the forget/new-feature
gate and FFN (`bimamba`'s `MambaPlus`), the horizon projection, and choosing
`d_inner`, `dt_rank`, `d_conv`, `d_state`. The original extraction rationale is only
recorded in the commit messages above.

## Interface

`MambaBlock(d_model, d_inner, dt_rank, d_conv, d_state, *, use_conv=True,
x_dropout=0.0, reference_dt_init=False)`, the five widths are positive ints with no
defaults; the keyword-only options are described under Variants. `d_inner` is the expanded width (callers use `expand * d_model`),
`dt_rank` the low-rank width of the step-size path, `d_conv` the conv kernel,
`d_state` the state size per channel. Methods: `forward(x)` for `[B, L, d_model]` to
`[B, L, d_model]`; `ssm(x)` for `[B, L, d_inner]` to `[B, L, d_inner]`;
static `selective_scan(u, delta, a, b, c, d)` with `u, delta: [B, L, d_inner]`,
`a: [d_inner, d_state]`, `b, c: [B, L, d_state]`, `d: [d_inner]` returning
`[B, L, d_inner]`.

- State-dict keys of `MambaBlock`: `A_log` `[d_inner, d_state]` (init log(1..d_state)),
  `D` `[d_inner]` (ones), `in_proj.weight`, `conv1d.weight`/`conv1d.bias`,
  `x_proj.weight`, `dt_proj.weight`/`dt_proj.bias`, `out_proj.weight`. No buffers.
  `use_conv=False` removes the two `conv1d.*` keys; the other options add no keys.
  `dt_proj` uses default `nn.Linear` init unless `reference_dt_init=True`.
- `RMSNorm(d_model, eps=1e-5)`: parameter `weight` `[d_model]`; no shape check.
- `MambaResidualBlock(d_model, d_inner, dt_rank, d_conv, d_state)`: keys `mixer.*`
  and `norm.weight`; output shape equals input, so `d_model` is preserved.
- Stateless (no cache across calls); no errors are raised by the component, shape
  mismatches surface as torch errors. The scan keeps its hidden state in a local
  zero tensor allocated on `delta`'s device with default float dtype, and computes
  `A` and `D` in float32, so half-precision inputs are not a supported contract.
  Cost is O(L) Python-loop steps; memory holds `[B, L, d_inner, d_state]` tensors.

## Invariants and equivalence evidence

- `tests/test_ssm_sequence_forecasters.py` asserts `s_mamba` layers are instances
  of the shared `MambaBlock` and exercises `bimamba`'s `MambaPlus` wrapper around it.
- Causality (outputs at positions before a perturbation are unchanged) and the
  `[B, L, d_model]` shape were confirmed with a tiny CPU snippet while writing
  this card; they are now pinned by the contract test below.
- `tests/test_component_contracts_signal.py` also pins the keyword options (same keys as the default block apart from `use_conv=False`, the dt-init ranges, and train-only dropout).
- `tests/test_component_contracts_signal.py` pins the interface (shapes, dtype, state-dict keys, invariants, gradient flow, error cases) and a seeded numerical regression against `tests/fixtures/components/mamba.pt`.

## Variants and options

The constructor widths plus three keyword-only options, all defaulting to the
original block: `use_conv=False` skips the causal convolution and its SiLU (the
scan then sees the in-projection output directly, as in MambaTS);
`x_dropout=p` applies dropout to the joint step-size/B/C projection output, active
only in training mode (the "selective parameter dropout" of MambaTS); and
`reference_dt_init=True` draws `dt_proj.weight` uniformly in `+-dt_rank**-0.5` and sets
the bias to the inverse softplus of a log-uniform step in `[1e-3, 1e-1]` (floor
`1e-4`), as the reference Mamba does. Not covered here: bidirectional use (instantiate two
blocks and flip the sequence, as `bimamba` and `s_mamba` do), a gated "Mamba+"
variant (model-local in `bimamba`), a scalar-state scan (see `hyper_state_scan`),
parallel-scan or fused CUDA kernels, and step/cached inference.

## When to use and when not to use

Use for a portable CPU/GPU selective SSM over a token axis (time tokens or
variate tokens) with modest length. Do not use for very long sequences where the
Python-loop scan is too slow, when exact numerical match to `mamba_ssm` kernels or
fp16 training is required, or when a scalar-state variant is wanted.

## Related components

`hyper_state_scan` (scalar-state scan plus grid mixing), `revin` (typical input
normalization in front of SSM forecasters), `mixer_block`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `RMSNorm(d_model: int, eps: float=1e-05)`
  Root-mean-square normalization over the final feature dimension.
- `MambaBlock(d_model: int, d_inner: int, dt_rank: int, d_conv: int, d_state: int, *, use_conv: bool=True, x_dropout: float=0.0, reference_dt_init: bool=False)`
  Pure-PyTorch selective state-space mixer with an optional causal depthwise convolution.
- `MambaResidualBlock(d_model: int, d_inner: int, dt_rank: int, d_conv: int, d_state: int)`
  Pre-normalized residual wrapper around :class:`MambaBlock`.

```python
from tsflab.models._components.mamba import RMSNorm, MambaBlock, MambaResidualBlock
```

## Retrieval terms

`mamba`, `mixer`, `rmsnorm`, `ssm`, `state-space`

## Current model consumers (4)

`bimamba`, `composed`, `mambasimple`, `s_mamba`
<!-- component-card:generated:end -->
