---
name: "positional_encoding"
kind: "component"
module: "moderntsf.models._components.positional_encoding"
summary: "Factory positional_encoding(kind, learnable, length, width) returning an nn.Parameter position table (sincos, 1-D/2-D coordinate, or random init) for patch-token transformers."
category: "embedding"
input: "kind: str | None, learnable: bool, length: int >= 1, width: int >= 1 (no tensor input)"
output: "nn.Parameter of shape [length, width] (or [length, 1] for the 'zero'/'normal'/'gauss'/'uniform'/'lin1d'/'exp1d' kinds)"
origin: "Position-table modes of the PatchTST code base (Nie et al., ICLR 2023); implementation is a clean-room rewrite"
origin_models: ["patchtst"]
tags: ["encoding", "patch", "position", "transformer", "sinusoidal", "learnable", "additive"]
---

# positional_encoding

## Purpose

`positional_encoding(kind, learnable, length, width)` builds the additive
positional table for `length` token positions and `width` features and wraps it
in an `nn.Parameter` (`requires_grad=learnable`). The caller adds it to the token
tensor (`tokens + table`), broadcasting over batch.

- `sincos`: sinusoidal table `sin/cos(pos * 10000^(-2i/width))`, then centred and
  divided by `10 * std` (the standardisation `_standardize`).
- `lin1d`/`exp1d`/`lin2d`/`exp2d`: coordinate tables `2 * t^p * f^p - 1` with
  `t` linspace over positions, `f` linspace over features (2-D only), `p = 1`
  (lin) or `0.5` (exp), again standardised. 1-D kinds have width 1.
- `zeros`: uniform(-0.02, 0.02) of shape `[length, width]`.
- `zero`/`normal`/`gauss`/`uniform`: width-1 random columns (uniform(-0.02, 0.02),
  normal(0, 0.1), normal(0, 0.1), uniform(0, 0.1)) that broadcast over features.
- `None`: uniform(-0.02, 0.02) `[length, width]` and forced non-learnable.

## Origin and granularity

The mode names match the position-table choices of the PatchTST code base; the
original module was rewritten as part of `fba5fa99` ("finish clean-room shared
forecasting layer"), and the history does not record numerical equivalence with
the original. It is a pure factory for tables only: where the table is added,
dropout after addition, and 3-D (per-channel) positions stay in the models.
Consumers: `patchtst` (configurable), `gateformer` (`"sincos"`, frozen), `semixer`
and `lsinet` (`"zeros"`, learnable), `canet` (`"exp2d"`, learnable, width =
patch size).

## Interface

`positional_encoding(kind: str | None, learnable: bool, length: int, width: int)
-> nn.Parameter`. The only public symbol. Raises `ValueError` if `length < 1`
or `width < 1`, or if `kind` is not one of `None`, `zero`, `zeros`, `normal`,
`gauss`, `uniform`, `sincos`, `lin1d`, `exp1d`, `lin2d`, `exp2d`. Returns a
float32 CPU parameter; the caller registers it on a module (`self.position =
...`), so the state-dict key is whatever attribute name the model chooses. Random
kinds draw from the global torch RNG at call time. Note the naming trap: `zero`
is a random width-1 column, while `zeros` is the full `[length, width]` random
table (neither is all zeros). Standardisation uses the unbiased std of the whole table and falls back to
mean-centring only when that std is exactly 0. A table with a single element
(`lin1d`/`exp1d` with `length == 1`) has an undefined std and returns `NaN`
(confirmed on CPU); `length == 1` with width > 1 is fine.

## Invariants and equivalence evidence

- `tests/test_repository_contracts.py` includes `positional_encoding` in the
  dependency closure of `patchtst`.
- no fixture: no numeric fixture pins the tables; shapes and learnability were
  confirmed by running the factory on a tiny CPU snippet, and consumer model
  tests (for example `tests/test_transformer_patch_forecasters_a.py` for the
  model-local PatchTST) do not call this factory directly.

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
table is added), `embed` (token and temporal embeddings).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import moderntsf.models._components.positional_encoding
```

## Retrieval terms

`encoding`, `patch`, `position`, `transformer`

## Current model consumers (4)

`canet`, `gateformer`, `lsinet`, `semixer`
<!-- component-card:generated:end -->
