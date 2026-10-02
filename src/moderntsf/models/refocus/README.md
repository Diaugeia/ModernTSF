---
name: "ReFocus"
summary: "ReFocus reinforces mid-frequency and key-frequency spectral content for multivariate forecasting: it reversibly normalizes each instance, subtracts a beta-scaled moving-average trend to attenuate the dominant low-frequency band (AMEO), embeds the residual into the frequency domain with dense complex-linear projections, and refines it through a stack of Energy-based Key-Frequency Picking Blocks (EKPB) that stochastically pool one channel's spectrum per frequency bin (weighted by spectral energy) and fuse that shared key-frequency representation back into every channel before projecting to the forecast horizon."
paper: "https://arxiv.org/abs/2502.16890"
paper_title: "ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting"
venue: "arXiv"
year: 2025
code: "https://github.com/Levi-Ackman/ReFocus"
revision: "5b883b29f364b52a73835f1465e993433f94a1ed"
license: "unlicensed (no LICENSE file present in the repository; inspected only for read-only paper-structure clarification, no source copied)"

---
# ReFocus

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2502.16890); title: ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting; venue/year: arXiv / 2025
- [codebase](https://github.com/Levi-Ackman/ReFocus); revision: `5b883b29f364b52a73835f1465e993433f94a1ed`; license: `unlicensed (no LICENSE file present in the repository; inspected only for read-only paper-structure clarification, no source copied)`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ReFocus.toml`](../../../../configs/models/ReFocus.toml).

## Differences

Inspected official files: `models/LiNo.py` (the paper's `Model` class),
`layers/FLinear.py` (`FLinear`, `Filter`), `layers/Encoder.py` (the EKPB
`Encoder`), and `layers/RevIN.py`, at revision
`5b883b29f364b52a73835f1465e993433f94a1ed`.

**Paper-driven local implementation.** `FLinear` performs a dense complex
matrix mapping between every input and output frequency bin (no low-pass
truncation), so it is not equivalent to any cataloged frequency-interpolation
component and stays model-local. The "Key-Frequency Enhanced Training" (KET)
strategy described in Section 3.4 is a training-loop-level channel mix-up
augmentation applied to inputs and targets across alternating epochs; it is
outside this model's single `forward()` contract (which only defines the
architecture, not a custom training loop) and is not implemented — only the
architectural AMEO and EKPB blocks are. `EnergyBasedFrequencyPooling` samples
stochastically (`multinomial`) during training exactly as the official
`Encoder.forward` does, but uses a deterministic arg-max pick during evaluation
so verification and inference are reproducible; the paper only specifies the
training-time stochastic behavior. `series_decomposition.EdgePaddedMovingAverage`
is reused for the AMEO moving-average filter — for an odd `kernel_size` it is
the exact edge-replicate, symmetric moving average the official `Filter`
computes via a fixed-weight depthwise `Conv1d` with replicate padding. The
external repository is reference-only; no source file was copied or adapted.

## Shared components

- [`energy_frequency_pooling`](../_components/energy_frequency_pooling/README.md)
- [`revin`](../_components/revin/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_pick=32`, `layers=2`, `dropout=0.1`, `beta=0.5`, `kernel_size=25`, `initial=True`
<!-- model-card:canonical:end -->

## Source and verification

Inspected official files: `models/LiNo.py` (the paper's `Model` class),
`layers/FLinear.py` (`FLinear`, `Filter`), `layers/Encoder.py` (the EKPB
`Encoder`), and `layers/RevIN.py`, at revision
`5b883b29f364b52a73835f1465e993433f94a1ed`.

**Paper-driven local implementation.** `FLinear` performs a dense complex
matrix mapping between every input and output frequency bin (no low-pass
truncation), so it is not equivalent to any cataloged frequency-interpolation
component and stays model-local. The "Key-Frequency Enhanced Training" (KET)
strategy described in Section 3.4 is a training-loop-level channel mix-up
augmentation applied to inputs and targets across alternating epochs; it is
outside this model's single `forward()` contract (which only defines the
architecture, not a custom training loop) and is not implemented — only the
architectural AMEO and EKPB blocks are. `EnergyBasedFrequencyPooling` samples
stochastically (`multinomial`) during training exactly as the official
`Encoder.forward` does, but uses a deterministic arg-max pick during evaluation
so verification and inference are reproducible; the paper only specifies the
training-time stochastic behavior. `series_decomposition.EdgePaddedMovingAverage`
is reused for the AMEO moving-average filter — for an odd `kernel_size` it is
the exact edge-replicate, symmetric moving average the official `Filter`
computes via a fixed-weight depthwise `Conv1d` with replicate padding. The
external repository is reference-only; no source file was copied or adapted.
