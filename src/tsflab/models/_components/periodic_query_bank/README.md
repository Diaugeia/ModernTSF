---
name: "periodic_query_bank"
kind: "component"
module: "tsflab.models._components.periodic_query_bank"
summary: "Zero-initialized learnable [period, channels] table read as a phase-aligned window of consecutive rows (wrapping modulo period) from a per-sample start phase."
category: "memory"
input: "start_phase int tensor [batch]; length (int)"
output: "[batch, length, channels]"
origin: "Learnable cycle table of TQNet, Temporal Query Network for Efficient Multivariate Time Series Forecasting (ICML 2025, arXiv 2505.12917); the module docstring also names CycleNet's residual cycle as the same indexing pattern"
origin_models: ["tqnet"]
tags: ["cycle", "gather", "period", "phase", "query", "learnable-table"]
---

# periodic_query_bank

## Purpose

`PeriodicQueryBank(period, channels)` stores one learnable `channels`-vector per
phase of a fixed period. For start phases `s[b]` and a window `length`, it
returns `table[(s[b] + j) mod period]` for `j = 0..length-1`, shape
`[batch, length, channels]`. It is gather-only indexing: no value is computed
from the data.

## Origin and granularity

Extracted from `tqnet` (commit `b94ed873`, automated intake of TQNet,
TimePro, Gateformer, CANet), where the retrieved window is transposed to
`[B, C, L]` and used as the attention query over the normalized input
(channel aggregation). The docstring says CycleNet's residual cycle uses the
same table-and-offset idea; `cyclenet` does not import this component, so that
is a documented similarity, not an equivalence. Model-local in `tqnet`: how the
start phase is derived from the last input stamp (`_start_phase`: weekday, or
`weekday*24 + hour`, or hour depending on `cycle`, plus one step, modulo `cycle`), and what the window is used for.

## Interface

`PeriodicQueryBank(period: int, channels: int)`

- `period` (int >= 1) and `channels` (int >= 1), else `ValueError`.
- Parameter / state-dict key: `table`, shape `[period, channels]`, initialized to
  zeros (so an untrained bank returns zeros).
- `forward(start_phase, length) -> Tensor`: `start_phase` integer tensor of shape
  `[batch]` (`ndim != 1` raises `ValueError`; must be an integer dtype for
  indexing), `length` int >= 1 (else `ValueError`; may exceed `period`, then the
  window wraps repeatedly). Phases may be any integers; `remainder` makes them
  periodic (negative values wrap into range). Output is on the table's device;
  `start_phase` should be on the same device.
- Gradients reach only the gathered rows of `table`. Stateless otherwise.

## Invariants and equivalence evidence

- `tests/test_component_contracts_basic.py` (`test_periodic_query_bank_contract_and_reference`)
  checks the single state-dict key `table` `[period, channels]` and zero init, output shape and
  dtype, every output row against `table[(phase + t) % period]` including a negative phase and
  `length > period`, gradients reaching `table`, the four `ValueError` cases (2-D phases,
  `length < 1`, `period < 1`, `channels < 1`), and a seeded regression of output and gradient in
  `tests/fixtures/components/periodic_query_bank.pt`.
- `test_periodic_query_bank_gathers_wrapped_phase_windows` in
  `tests/test_2025_query_gate_hyperstate_forecasters.py` checks shape and
  wrap-around rows for `period=4, channels=2`, phases `[3, 0]`, length 3.
- `test_tqnet_forward_uses_temporal_query_as_attention_query` in the same file covers
  it as a consumer. No pre-extraction `tqnet` fixture exists, so state-dict and output
  equivalence with the pre-refactor model is not tested.

## Variants and options

None. There is no random or non-zero initialization, no per-channel period, and
no interpolation for fractional phases.

## When to use and when not to use

Use when a model has a known fixed period (daily/weekly cycle) and wants a
learnable per-phase vector aligned to each sample's calendar position. Do not
use for unknown or varying periods (see `dominant_periods`), or when the phase
index cannot be derived at the call site. Initialization at zero means the
bank carries no information until trained.

## Related components

`dominant_periods` (discovers periods instead of assuming one), `marks`
(calendar features from which a start phase can be computed),
`positional_encoding` (position tables indexed by token position, not by calendar phase),
`periodic_alibi_bias` (periodic structure applied as an additive attention bias, no learned
vectors), `deviation_memory` (a learned bank queried by content with attention, not by index).
`cyclenet` keeps its own local cycle table and does not use this component (same indexing
idea, not shared code).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `PeriodicQueryBank(period: int, channels: int)`
  One learnable vector per phase of a fixed period, indexed by offset.

```python
from tsflab.models._components.periodic_query_bank import PeriodicQueryBank
```

## Retrieval terms

`cycle`, `gather`, `period`, `phase`, `query`

## Current model consumers (1)

`tqnet`
<!-- component-card:generated:end -->
