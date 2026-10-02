---
name: "energy_frequency_pooling"
kind: "component"
module: "moderntsf.models._components.energy_frequency_pooling"
summary: "Per-frequency softmax-of-energy selection of one token's complex spectral value across tokens, broadcast to all tokens (sampled in training, argmax in eval)."
category: "frequency"
input: "spectrum complex [batch, tokens, freq]"
output: "complex [batch, tokens, freq]; the same value for every token at each frequency"
origin: "Energy-based key-frequency pooling of ReFocus, Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting (arXiv 2502.16890, 2025)"
origin_models: ["refocus"]
tags: ["energy", "frequency", "key-frequency", "pooling", "softmax", "stochastic", "complex", "train-eval-difference"]
---

# energy_frequency_pooling

## Purpose

`EnergyBasedFrequencyPooling()` takes a complex spectrum `S[b, n, f]` with a
token axis `n` (channels or variables) and for every `(b, f)` selects one token
`n*` and returns `S[b, n*, f]` copied to all token positions:

- `energy = |S|^2`, `p = softmax(energy, dim=tokens)`.
- training: `n* ~ Categorical(p[b, :, f])` (`torch.multinomial`).
- eval: `n* = argmax_n p[b, n, f]`.

The output is a single shared "key-frequency" spectrum that tokens can be fused
with. The softmax is over raw (unnormalized) energies, so it is peaked whenever
energies differ by more than a few units.

## Origin and granularity

Extracted from `refocus` (commit `6b491e13`, automated intake of ReFocus,
FreqMoE, SWIFT, Sensorformer), which uses it inside its key-frequency block:
rfft of hidden features, pooling, irfft, then fusion with the original input.
The module docstring says the deterministic eval argmax is a documented
train/eval difference, as the paper only describes the stochastic training
behaviour. Cut at the pooling operator; the FFT/IFFT pair, the fusion layers,
and the choice of what the token axis is stay in `refocus`.

## Interface

`EnergyBasedFrequencyPooling()` takes no arguments.

- `forward(spectrum)`: `spectrum` must be a complex tensor of shape
  `[batch, tokens, freq]`; otherwise `ValueError` (non-complex, or `ndim != 3`).
  Returns a complex tensor of the same shape and device (an expanded view of a
  gathered `[batch, 1, freq]` tensor, so it is not contiguous in memory).
- No parameters, buffers, or state-dict keys. Behaviour depends on
  `self.training`; the training path consumes the global RNG
  (`torch.multinomial`), so seeds affect it.
- Gradient flows to the selected complex values through `gather`; the selection
  itself is not differentiated (the energy softmax receives no gradient).

## Invariants and equivalence evidence

- `test_energy_based_frequency_pooling_picks_the_higher_energy_token` in
  `tests/test_frequency_wavelet_attention_forecasters.py` checks, in eval, that
  both tokens receive the higher-energy token's value.
- no fixture: no stored pre-refactor tensors exist for this component; the
  training-time stochastic path is not covered by a dedicated test (only by
  the `refocus` model tests through the consumer).

## Variants and options

None. There is no temperature, top-k, or straight-through variant; making
eval stochastic or the softmax temperature configurable would be a new
component.

## When to use and when not to use

Use to build a cross-token shared spectrum where each frequency is taken from
its strongest token, on `[B, N, F]` complex input. Do not use when a
differentiable weighted average over tokens is wanted (this selects, it does not
average), when eval must equal the training expectation (eval is argmax, train
is a sample), or when the input is real-valued.

## Related components

`freq_band_moe` (frequency-domain gating by band), `frequency_band_sampler`,
`revin` (ReFocus normalizes with it before this block).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `EnergyBasedFrequencyPooling()`
  Softmax-energy-weighted pooling of a complex spectrum across tokens.

```python
from moderntsf.models._components.energy_frequency_pooling import EnergyBasedFrequencyPooling
```

## Retrieval terms

`energy`, `frequency`, `key-frequency`, `pooling`, `softmax`, `stochastic`

## Current model consumers (1)

`refocus`
<!-- component-card:generated:end -->
