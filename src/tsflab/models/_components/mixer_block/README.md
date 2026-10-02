---
name: "mixer_block"
kind: "component"
module: "tsflab.models._components.mixer_block"
summary: "TSMixer basic block: pre-LayerNorm residual time-mixing Linear then pre-LayerNorm residual two-layer feature MLP, both with GELU and dropout."
category: "mixer"
input: "[batch, seq_len, channels]"
output: "[batch, seq_len, channels]"
origin: "TSMixer, Chen et al., TMLR 2023 (An All-MLP Architecture for Time Series Forecasting), Appendix B.3.2 basic block"
origin_models: ["tsmixer"]
tags: ["feature", "gelu", "layernorm", "mixer", "residual", "time", "tsmixer", "mlp", "pre-norm", "time-mixing", "feature-mixing"]
---

# mixer_block

## Purpose

`MixerBlock(seq_len, channels, hidden, dropout)` applies two residual updates to
`x [B, L, C]`:

1. Time mixing: `x = x + drop(gelu(Linear_L(LN_time(x)^T)))^T`, where `LN_time` is a LayerNorm over the joint `(L, C)` shape, and the linear acts on the time axis (shared across channels).
2. Feature mixing: `x = x + drop(Linear_out(drop(gelu(Linear_in(LN_feat(x))))))` with `Linear_in: C -> hidden`, `Linear_out: hidden -> C`.

Shape is preserved, so blocks stack.

## Origin and granularity

Extracted in commit `ea49cee5` ("mixer_block from TSMixer; audit mixer family")
verbatim from the model-local block in `tsmixer`, with attribute names unchanged;
the `tsmixer` model card records a clean-room implementation from paper
Appendix B.3.1-B.3.2 of the basic historical-target variant (arXiv 2303.06053).
The module docstring says to reuse the block only when mixing order, normalization
shape, and residual placement match exactly; paper-specific factorizations (other
mixers, low-rank channel bottlenecks, post-norm, extra hidden layers) stay local.
The final temporal projection stays in `tsmixer` (via `channel_wise_linear`).

## Interface

`MixerBlock(seq_len: int, channels: int, hidden: int, dropout: float)`

- `seq_len` (int >= 1): fixed; must equal the input time length (LayerNorm shape and time Linear are tied to it).
- `channels` (int >= 1): must equal the input channel width.
- `hidden` (int >= 1): feature-MLP width.
- `dropout` (float in [0, 1)): one `nn.Dropout` applied after the time activation, after the feature activation, and on the feature delta.
- `forward(x [B, seq_len, channels]) -> same shape`; float tensors, no explicit validation (shape errors come from the layers).
- State-dict keys: `time_norm.weight/bias` and `feature_norm.weight/bias` (each `[seq_len, channels]`), `time_projection.weight/bias` (`[seq_len, seq_len]`), `feature_in.weight/bias`, `feature_out.weight/bias`. GELU and dropout have no parameters. Stateless apart from dropout.

## Invariants and equivalence evidence

- `tests/test_component_contracts_basic.py` (`test_mixer_block_contract_and_reference`)
  checks the exact state-dict key set, the `[6, 3]` norm and `[6, 6]` time-projection
  shapes, shape and dtype preservation, gradients to the input and every parameter,
  that zeroing `time_projection` and `feature_out` makes the block the identity, and
  seeded values against `tests/fixtures/components/mixer_block.pt`.
- `tests/test_component_extraction_mixer.py` freezes a copy of the pre-extraction
  `MixerBlock` and TSMixer `Model` and checks identical `state_dict()` keys and
  shapes, forward output, and gradients from a fixed seed
  (`test_state_dict_keys_and_shapes_match`, `test_forward_outputs_match`,
  `test_gradients_match`).
- There is no test of non-default dropout behaviour or of shape-mismatch errors.

## Variants and options

None beyond the constructor arguments; activation is fixed to GELU, normalization
to joint `(L, C)` LayerNorm, and pre-norm order is fixed.

## When to use and when not to use

Use to stack TSMixer-style blocks over fixed-length `[B, L, C]` inputs where the
time-mixing linear shares weights across channels and features are mixed per step.
Do not use for variable-length inputs, for the auxiliary/static-feature TSMixer
extension, for BatchNorm or post-norm variants, or when the time-mixing linear must
be channel-specific.

## Related components

`channel_wise_linear` (the temporal projection that follows the stack in
`tsmixer`), `gated_fusion` (learned blending of two branches rather than residual
mixing within one stream). `mtsmixer` (`FactorizedMixerBlock`) and `rpmixer`
(`RPMixerBlock`) keep their own mixer blocks, model-local, because their
factorization differs. `composed` only imports this block as a registration marker
(`# noqa: F401`), not as a used layer.
- `softmax_gate`: the gated-attention filter PatchTSMixer adds after its MLP mixers; this block has no gate.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `MixerBlock(seq_len: int, channels: int, hidden: int, dropout: float)`
  Paper time mixing followed by feature mixing, both residual.

```python
from tsflab.models._components.mixer_block import MixerBlock
```

## Retrieval terms

`feature`, `gelu`, `layernorm`, `mixer`, `residual`, `time`

## Current model consumers (2)

`composed`, `tsmixer`
<!-- component-card:generated:end -->
