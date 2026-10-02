---
name: "freq_band_moe"
kind: "component"
module: "tsflab.models._components.freq_band_moe"
summary: "Instance-normalizes a series, splits its rfft into expert_num contiguous bands at sorted sigmoid boundaries, recombines bands with softmax gating, and rescales."
category: "decomposition"
input: "x [batch, channels, seq_len]"
output: "combined [batch, channels, seq_len]; boundaries [expert_num + 1]; gating_scores [batch, expert_num]"
origin: "Frequency-band decomposition mixture of experts of FreqMoE, Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts (arXiv 2501.15125, AISTATS 2025)"
origin_models: ["freqmoe"]
tags: ["band", "decomposition", "experts", "frequency", "gating", "mixture", "rfft", "instance-normalization", "stateless-output"]
---

# freq_band_moe

## Purpose

`FrequencyBandMixtureOfExperts(expert_num, seq_len)` denoises a series by a
gated recombination of frequency bands. For `x [B, C, T]`:

1. Instance-normalize over time: `mu = mean_t x`, `v = var_t(x - mu) + 1e-5`
   (unbiased variance), `z = (x - mu) / sqrt(v)`.
2. `F = rfft(z)` (`T//2 + 1` bins).
3. Boundaries: `sort(sigmoid(band_boundaries))`, padded with 0 and 1, scaled by
   the bin count and cast to integers; band `e` is the half-open bin range
   `[idx[e], idx[e+1])` (the last index is forced to the bin count).
4. Gate: `g = softmax(MLP(mean_c |F|))` with an MLP
   `Linear(F, F) -> ReLU -> Linear(F, expert_num)`, so `g` is `[B, expert_num]`.
5. `out = irfft(sum_e g[b, e] * (F masked to band e), n=T) * sqrt(v) + mu`.

## Origin and granularity

Extracted from `freqmoe` (commit `6b491e13`, automated intake). The module
docstring calls it a paper-neutral block for frequency-domain MoE forecasters.
Kept here: normalization, band split, gate, recombination, and restoration of
the instance scale. Model-local in `freqmoe`: the downstream frequency-extension
blocks (complex linear layers upsampling the spectrum from `seq_len` to
`seq_len + pred_len`), the complex ReLU/dropout, and the use of the returned
boundaries and gate scores (stored as `last_band_boundaries`, `last_gating_scores`).

## Interface

`FrequencyBandMixtureOfExperts(expert_num: int, seq_len: int, learnable_boundaries: bool = False, boundary_temperature: float = 1.0)`

- `expert_num` (int >= 1): number of bands/experts. `seq_len` (int >= 1): exact
  time length accepted; `freq_len = seq_len // 2 + 1`. Violations raise
  `ValueError`.
- `learnable_boundaries` (default `False`): fixed boundaries, as in the official
  code. `True`: boundaries train (see Variants). `boundary_temperature` (> 0): soft
  edge width in bins for the straight-through gradient, used only when learnable.
- State-dict keys: `band_boundaries` (shape `[max(expert_num - 1, 0)]`, init
  `torch.rand`; a registered buffer by default, an `nn.Parameter` when
  `learnable_boundaries=True`; the key is the same in both, so checkpoints saved
  from the former Parameter, or from either mode, load into either mode),
  `gating_network.0.weight/bias` (`[F, F]`, `[F]`), `gating_network.2.weight/bias`
  (`[expert_num, F]`, `[expert_num]`).
- `forward(x)`: `x` floating `[batch, channels, seq_len]`; wrong rank or last
  length raises `ValueError`. Returns `(combined, boundaries, gating_scores)`:
  `combined` same shape as `x`; `boundaries` shape `[expert_num + 1]` in
  `[0, 1]` (0 first, 1 last); `gating_scores` `[batch, expert_num]`, rows sum to 1.
- Stateless between calls (no cached tensors in the module).

## Invariants and equivalence evidence

- `test_frequency_band_moe_partitions_are_contiguous_and_gates_sum_to_one` in
  `tests/test_frequency_wavelet_attention_forecasters.py` checks output shape,
  boundary vector length and monotonicity, and gate rows summing to 1.
- no fixture: no pre-refactor tensor fixture; consumer-level behaviour is
  covered by the `freqmoe` model tests in the same file.
- `tests/test_component_numeric_fixes.py`: the default module is bit-identical
  to the former implementation (checked once against the pre-change code, with
  the old state dict loaded); `band_boundaries` is a buffer with no gradient and
  round-trips through `state_dict`; with `learnable_boundaries=True` the forward
  output equals the fixed module's and `band_boundaries.grad` is finite and nonzero.
- Paper vs code: the paper (Sec. on the frequency-decomposition MoE) says the
  boundaries are learned end-to-end, but the official code casts them to integers,
  which blocks every gradient. The default follows the official code (fixed,
  random-initialization boundaries; a buffer, so the module no longer advertises a
  parameter that cannot train). Learning is opt-in.
- Bands that round to zero width are empty (their mask is all zero). With
  `expert_num == 1` the single band covers all bins and `boundaries == [0, 1]`.

## Variants and options

`learnable_boundaries=True` turns `band_boundaries` into a Parameter trained with
a straight-through estimator: the forward uses the exact hard 0/1 band masks, the
backward uses soft masks `sigmoid((bin + 0.5 - edge) / T)` differences, with
`edge = boundary * freq_len`. `boundary_temperature` is `T` (bins). This is not in
the paper or official code. Boundaries are shared over channels and batch;
only the gate depends on the input, and it sees the channel-averaged amplitude
spectrum.

## When to use and when not to use

Use as a front-end that gates frequency bands of a `[B, C, T]` window with a
fixed window length. Pass `learnable_boundaries=True` when the boundaries should be learned. Do not use when the sequence length varies, or when the
normalization should be handled by a shared `revin` (this module normalizes
internally and restores scale itself).

## Related components

`frequency_band_sampler` (deterministic depth-indexed bands), `revin`
(external reversible normalization), `series_decomposition` (time-domain
decomposition), `topk_expert_router` (routing over experts without a spectrum).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `FrequencyBandMixtureOfExperts(expert_num: int, seq_len: int, learnable_boundaries: bool=False, boundary_temperature: float=1.0)`
  Decompose a series into frequency bands and gate their mixture.

```python
from tsflab.models._components.freq_band_moe import FrequencyBandMixtureOfExperts
```

## Retrieval terms

`band`, `decomposition`, `experts`, `frequency`, `gating`, `mixture`, `rfft`

## Current model consumers (1)

`freqmoe`
<!-- component-card:generated:end -->
