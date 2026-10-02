---
name: "positional_encoding"
kind: "component"
module: "tsflab.models._components.positional_encoding"
summary: "Factory positional_encoding(kind, learnable, length, width) returning an nn.Parameter position table (sincos, 1-D/2-D coordinate, or random init) added to patch tokens."
category: "embedding"
input: "kind: str | None, learnable: bool, length: int >= 1, width: int >= 1 (no tensor input)"
output: "nn.Parameter [length, width] (kinds sincos, zeros, small_uniform_2d, lin2d, exp2d, None); [length, 1] for kinds zero, small_uniform, normal, gauss, uniform, lin1d, exp1d"
origin: "Position-table modes of the PatchTST code base (Nie et al., ICLR 2023); implementation is a clean-room rewrite"
origin_models: ["patchtst"]
tags: ["encoding", "patch", "position", "transformer", "sinusoidal", "learnable", "additive", "table", "sincos", "factory"]
---

# positional_encoding

## Purpose

`positional_encoding(kind, learnable, length, width)` builds the additive
positional table for `length` token positions and `width` features and wraps it
in an `nn.Parameter` (`requires_grad=learnable`). The caller adds it to the token
tensor (`tokens + table`), broadcasting over batch.

- `sincos`: sinusoidal table with sine on even columns and cosine on odd columns,
  `sin/cos(pos * 10000^(-2i/width))`, then standardized (below).
- `lin1d`/`exp1d`/`lin2d`/`exp2d`: coordinate tables, standardized. 1-D kinds
  have width 1 and equal `2 * t^p - 1`; 2-D kinds equal `2 * t^p * f^p - 1`.
  Here `t` is a linspace over positions in [0, 1], `f` a linspace over features
  in [0, 1], and `p = 1` (lin) or `0.5` (exp). A 2-D kind with `width == 1`
  reduces to the 1-D formula.
- `zeros` (alias `small_uniform_2d`): uniform(-0.02, 0.02) of shape `[length, width]`; not all zeros.
- `zero` (alias `small_uniform`; not all zeros)/`normal`/`gauss`/`uniform`: width-1 random columns (uniform(-0.02, 0.02),
  normal(0, 0.1), normal(0, 0.1), uniform(0, 0.1)) that broadcast over features.
- `None`: uniform(-0.02, 0.02) `[length, width]` and forced non-learnable.

Standardization (`sincos` and the four coordinate kinds): subtract the table
mean and divide by `10 *` the unbiased standard deviation of the whole table, so
a non-degenerate table has mean 0 and std 0.1; if that std is exactly 0 the table
is only mean-centred.

## Origin and granularity

The mode names match the position-table choices of the PatchTST code base; the
original module was rewritten as part of `fba5fa99` ("finish clean-room shared
forecasting layer"), and the history does not record numerical equivalence with
the original. It is a pure factory for tables only: where the table is added,
dropout after addition, and 3-D (per-channel) positions stay in the models.
Consumers: the `patchtst` component backbone (configurable `pe`/`learn_pe`),
`gateformer` (`"sincos"`, frozen), `semixer`, `lsinet` and `mou` (`"zeros"`,
learnable), `canet` (`"exp2d"`, learnable, width = patch size), and the composed
`temporal` slot adapter in `_slots` (`"sincos"`, frozen), which is why `composed`
declares it.

## Interface

`positional_encoding(kind: str | None, learnable: bool, length: int, width: int)
-> nn.Parameter`. The only public symbol (the generated block shows a
module-level import because the catalog spec lists no public symbols). Raises
`ValueError` if `length < 1` or `width < 1`, or if `kind` is not one of `None`,
`zero`, `small_uniform`, `zeros`, `small_uniform_2d`, `normal`, `gauss`,
`uniform`, `sincos`, `lin1d`, `exp1d`, `lin2d`, `exp2d`. Returns a
default-dtype (float32) CPU parameter; the caller registers it on a module
(`self.position = ...`), so the state-dict key is whatever attribute name the
model chooses. Random kinds draw from the global torch RNG at call time. Note the
naming trap: `zero` is a random width-1 column, while `zeros` is the full
`[length, width]` random table (neither is all zeros). `learnable` is coerced
with `bool(...)`. A standardized table with a single element (`sincos` with
`length == width == 1`, or `lin1d`/`exp1d` with `length == 1`) has an undefined
unbiased std and returns `NaN`; `length == 1` with `width > 1` is fine.
Odd `width` is supported for `sincos`.

## Invariants and equivalence evidence

- `tests/test_component_contracts_basic.py` (`test_positional_encoding_shapes`)
  runs every kind except the aliases at `length=6, width=4`: result is an
  `nn.Parameter`, float32 on CPU, shape `[6, 1]` for `zero`, `normal`, `gauss`,
  `uniform`, `lin1d`, `exp1d` and `[6, 4]` otherwise, finite, `requires_grad`
  true exactly when `learnable` and `kind is not None`, and `learnable=False`
  always frozen.
- `test_positional_encoding_errors_and_reference` in the same file checks the
  `ValueError` cases (unknown kind, `length=0`, `width=0`), that the same seed
  reproduces a random table, that a `sincos` table has mean 0 and std 0.1, and
  pins `sincos` (6x5), `lin2d` (5x3) and `exp1d` (5x3) in
  `tests/fixtures/components/positional_encoding.pt`.
- `tests/test_repository_contracts.py` includes `positional_encoding` in the
  dependency closure of the `patchtst` component.
- no fixture for the random kinds, the aliases, `exp2d`/`lin1d` values, or the
  single-element NaN case (the latter is by inspection of the standardization),
  and none against the original PatchTST tables.

## Variants and options

Choose by `kind` as above; `learnable=False` freezes the table (still registered
as a parameter, so it appears in `state_dict` and `parameters()`, with
`requires_grad=False`). `kind=None` is always frozen.

## When to use and when not to use

Use for a per-position (patch index) additive table of fixed length. Do not use
when the sequence length varies at run time (the table is fixed-length), for
rotary or relative encodings, or when per-channel positions are needed.

## Related components

`patchtst` (configurable consumer), `tst_transformer` (encodes tokens after the
table is added), `embed` (its `PositionalEmbedding` is also a fixed sin/cos table,
but as a non-trainable `[1, max_len, d_model]` buffer inside `DataEmbedding`, with
no standardization and an even-`d_model` requirement), `periodic_alibi_bias`
(an attention bias instead of an additive token table).
- `periodic_query_bank`: phase-indexed learned table, versus a fixed position table.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import tsflab.models._components.positional_encoding
```

## Retrieval terms

`encoding`, `patch`, `position`, `transformer`

## Current model consumers (6)

`canet`, `composed`, `gateformer`, `lsinet`, `mou`, `semixer`
<!-- component-card:generated:end -->
