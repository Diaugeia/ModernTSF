---
name: "self_attention_family"
kind: "component"
module: "tsflab.models._components.self_attention_family"
summary: "Time-Series-Library-style attention cores (full softmax, ProbSparse, flow, tiled flash-style, Reformer LSH) plus the AttentionLayer that wraps any core with Q/K/V/output projections."
category: "attention"
input: "AttentionLayer: queries [batch, len_q, d_model], keys/values [batch, len_k, d_model]; cores: queries [batch, len_q, heads, d_head], keys/values [batch, len_k, heads, d_head]"
output: "AttentionLayer: ([batch, len_q, d_model], attention map or None); FullAttention/FlowAttention/FlashAttention cores: ([batch, len_q, heads, d_head], map or None); ProbAttention core: ([batch, heads, len_q, d_head], map or None)"
origin: "Attention classes follow the Informer / Time-Series-Library layer API (Zhou et al., AAAI 2021 for ProbSparse; Vaswani et al., 2017 for full attention); FlowAttention, FlashAttention and ReformerLayer are named after Flowformer, tiled flash attention and Reformer LSH attention but their upstream source is not recorded in history"
origin_models: ["informer", "transformer"]
tags: ["attention", "full", "probabilistic", "probsparse", "causal-mask", "multi-head", "stateless"]
---

# self_attention_family

## Purpose

A bundle of interchangeable attention "cores" sharing one call signature
`core(queries, keys, values, attn_mask, tau=None, delta=None) -> (output, attn_or_None)`
and one wrapper, `AttentionLayer`, that projects a `d_model` sequence into
heads, calls a core, and projects back.

- `FullAttention`: `softmax(scale * Q K^T) V` per head with `scale = scale or
  1/sqrt(d_head)`, optional causal masking, attention dropout.
- `ProbAttention`: Informer ProbSparse attention. Samples `u_part = factor *
  ceil(ln len_k)` keys per query, scores queries by `max - mean` of the sampled
  logits, keeps the top `u = factor * ceil(ln len_q)` queries, runs exact
  attention only for those, and fills the remaining query rows with the mean of
  `V` (`mask_flag=False`) or the running cumulative sum of `V` (`mask_flag=True`).
- `FlowAttention`: sigmoid-kernel linear attention with row/column flow
  normalizers; no softmax map is produced and `attn_mask` is ignored.
- `FlashAttention`: pure-PyTorch block-wise online-softmax attention (an
  algorithmic tiling reference, not a fused kernel).
- `ReformerLayer`: thin wrapper over `reformer_pytorch.LSHSelfAttention`.
- `AttentionLayer(attention, d_model, n_heads, d_keys=None, d_values=None)`.

## Origin and granularity

Present since the initial commit and consolidated in the component move
(`33ea2050`, "colocate shared components under models"). The cut keeps the
head-split core separate from the projection wrapper so encoder and decoder
blocks (`transformer_encdec`) can swap cores (full vs ProbSparse) without
changing anything else; `informer` and `transformer` build exactly this way, and
`dualformer` reuses `AttentionLayer` with its own frequency-domain core.
Per-model choices stay model-local: which core to use per role, mask objects,
`factor`, and distilling. Which paper each non-Informer core was copied from is
not recorded in history; the identification above is by class behaviour only.

## Interface

Cores `FullAttention`, `ProbAttention`, `FlashAttention`: `(mask_flag=True,
factor=5, scale=None, attention_dropout=0.1, output_attention=False)`;
`factor` is used only by `ProbAttention`. `FlowAttention(attention_dropout=0.1)`.
`ReformerLayer(attention, d_model, n_heads, d_keys=None, d_values=None,
causal=False, bucket_size=4, n_hashes=4)`; raises `ImportError` at construction
if `reformer_pytorch` is missing (`LSHSelfAttention` is then `None`).

`AttentionLayer(attention, d_model, n_heads, d_keys=None, d_values=None)`:
`d_keys`/`d_values` default to `d_model // n_heads`; `d_model` should be divisible
by `n_heads` when defaults are used. Parameters/state-dict keys:
`query_projection`, `key_projection`, `value_projection` (`d_model -> d_keys *
n_heads`, `d_values * n_heads`) and `out_projection`, plus whatever the inner
core holds under `inner_attention.*` (the cores have no parameters). `forward(queries,
keys, values, attn_mask, tau=None, delta=None)` takes batch-first
`[B, L, d_model]`, returns `(out [B, L_q, d_model], attn)`.

Mask handling: `attn_mask` must be an object with a `.mask` bool tensor (see
`masking`: `TriangularCausalMask`, `ProbMask`). `FullAttention` with
`mask_flag=True` and `attn_mask=None` builds a causal mask itself; with
`mask_flag=False` the mask is ignored. `FlashAttention` instead treats
`attn_mask` as a `[batch, len_k]` 0/1 key-padding tensor and never applies a
causal mask. `tau`/`delta` are accepted and ignored by every core here
(they exist for de-stationary attention consumers). `output_attention=True`
returns the `[B, H, L_q, L_k]` map (`ProbAttention`: uniform `1/L` rows for
unselected queries). Sampling in `ProbAttention` uses unseeded `torch.randint`
on CPU, so it is stochastic even in eval mode; `FullAttention` dropout is the
only other randomness. All cores are stateless.

Quirk to know: `ProbAttention.forward` returns the context as `[B, heads,
len_q, d_head]` (heads before time), unlike the other cores, and
`AttentionLayer` then reshapes with `view(B, L, -1)`; any consumer that mixes
`ProbAttention` with other cores must not assume the same output layout. Its
`_get_initial_context` also asserts `len_q == len_k` when `mask_flag=True`.

## Invariants and equivalence evidence

- `tests/test_component_contracts_attention.py`: shape, state-dict key, invariant, gradient-flow and seeded numerical-regression tests for every public symbol; reference values in `tests/fixtures/components/self_attention_family_flash.pt`, `tests/fixtures/components/self_attention_family_flow.pt`, `tests/fixtures/components/self_attention_family_full.pt`, `tests/fixtures/components/self_attention_family_layer.pt`, `tests/fixtures/components/self_attention_family_prob.pt`.
- `tests/test_local_attention_forecasters.py`: `FullAttention` matches the
  scaled dot-product equation; `transformer` uses `FullAttention` in the encoder
  and decoder roles (decoder self-attention causal, cross-attention not);
  `informer` uses `ProbAttention` with distilling.
- `tests/test_dualformer_forecaster.py` checks `AttentionLayer` wrapping
  `FullAttention` alongside a model-local core.
- no pre-refactor fixture exists; the contract test also covers `FlowAttention`, `FlashAttention`
  (matches `FullAttention` unmasked), and `ProbAttention`; `ReformerLayer` is only checked for its
  optional-dependency behavior.

## Variants and options

Choose the core by role: `FullAttention` (exact, `O(L^2)`), `ProbAttention`
(sub-quadratic, stochastic, needs `factor`), `FlowAttention` (linear in length,
no map), `FlashAttention` (exact, key-padding mask only, slow Python loops),
`ReformerLayer` (optional dependency; ignores keys/values and `attn_mask`,
pads length to a multiple of `2 * bucket_size`).

## When to use and when not to use

Use for encoder/decoder Transformers in the Informer/Transformer line where the
attention core must be swappable behind `AttentionLayer`. Do not use
`ProbAttention` for deterministic evaluation or short sequences (it falls back to
all queries when `u >= len_q` but still samples); do not use `FlashAttention` as
a performance primitive (it is a reference). For patch-token encoders with
a fused PyTorch attention use `tst_transformer`; for noise-cancelling or routed
variants see `differential_attention` and `topk_expert_attention`.

## Related components

`transformer_encdec` (blocks that consume `AttentionLayer`), `tst_transformer`,
`differential_attention`, `topk_expert_attention`,
`global_patch_compression_attention`, `masking`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import tsflab.models._components.self_attention_family
```

## Retrieval terms

`attention`, `full`, `probabilistic`

## Current model consumers (4)

`dualformer`, `gpht`, `informer`, `transformer`
<!-- component-card:generated:end -->
