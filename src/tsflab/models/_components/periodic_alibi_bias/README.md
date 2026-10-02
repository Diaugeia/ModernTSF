---
name: "periodic_alibi_bias"
kind: "component"
module: "tsflab.models._components.periodic_alibi_bias"
summary: "Additive attention bias [heads, N, N]: ALiBi slopes times |i - j|, or its triangle-wave periodic form with one period per head group."
category: "attention"
input: "seq_len, num_heads, periods (one integer or None per head group)"
output: "[num_heads, seq_len, seq_len] bias to add to attention scores"
origin: "ALiBi (Press et al., 2022) extended with periodic distances by PENGUIN, Sun et al., AISTATS 2026 (arXiv 2508.13773), Eq. 5-6"
origin_models: ["penguin"]
tags: ["alibi", "bias", "periodic", "relative-position", "attention", "stateless"]
---

# periodic_alibi_bias

## Purpose

`periodic_alibi_bias(seq_len, num_heads, periods)` builds the relative-position
bias added to pre-softmax attention scores. For a head with slope `m_k` the
bias is `-m_k * d(i, j)`. Plain ALiBi uses `d = |i - j|`; the periodic form
replaces it with the triangle wave of `|i - j| mod P`, so tokens a whole period
apart are as close as neighbours. Heads are grouped, each group owning one period
(or `None` for the plain distance).

## Origin and granularity

Extracted with PENGUIN. Only the bias table is shared; grouped-query projections,
masking and softmax stay in the model because they differ across attention
designs. It is a pure function with no parameters.

## Interface

`periodic_alibi_bias(seq_len, num_heads, periods=None, *, device=None, dtype=torch.float32) -> Tensor`

- `seq_len`, `num_heads` (int >= 1). `periods`: sequence of `int >= 1` or `None`, one entry per head group, in head order; `None` or empty means one non-periodic group. `num_heads` must be divisible by the number of groups, else `ValueError`.
- Slopes inside a group of `n` heads are `2 ** (-8 k / n)`, `k = 1..n`, identical across groups.
- Returns `[num_heads, seq_len, seq_len]`, non-positive, symmetric in `(i, j)`, zero on the diagonal. Period distances never exceed `P / 2`.
- No state, no parameters; deterministic.

## Invariants and equivalence evidence

- Contract, symmetry, period-wrap and hand-computed value tests: `tests/test_component_periodic_alibi_bias.py` (no numerical fixture file; expected tensors are written out in the test).
- Equals the standard ALiBi bias for a single non-periodic group and a power-of-two head count.

## Variants and options

Not covered: the extra slope interpolation ALiBi uses for head counts that are not powers of two, per-head (not per-group) periods, learned slopes, or the cross-attention form with different query and key lengths.

## When to use and when not to use

Use as an additive score bias for self-attention over equally spaced tokens with a known period (or none). Do not use for cross-attention with unequal lengths, for irregular timestamps, or when slopes should be learned.

## Related components

`self_attention_family` (attention cores without relative bias), `positional_encoding` (absolute tables).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `periodic_alibi_bias(seq_len: int, num_heads: int, periods: Sequence[int | None] | None=None, *, device: torch.device | str | None=None, dtype: torch.dtype=torch.float32)`
  Return an additive attention bias ``[num_heads, seq_len, seq_len]``.

```python
from tsflab.models._components.periodic_alibi_bias import periodic_alibi_bias
```

## Retrieval terms

`alibi`, `bias`, `periodic`, `relative-position`, `attention`

## Current model consumers (1)

`penguin`
<!-- component-card:generated:end -->
