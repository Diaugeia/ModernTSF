---
name: "ReFocus"
summary: "ReFocus reinforces mid-frequency and key-frequency spectral content for multivariate forecasting: it reversibly normalizes each instance, subtracts a beta-scaled moving-average trend to attenuate the dominant low-frequency band (AMEO), embeds the residual into the frequency domain with dense complex-linear projections, and refines it through a stack of Energy-based Key-Frequency Picking Blocks (EKPB) that stochastically pool one channel's spectrum per frequency bin (weighted by spectral energy) and fuse that shared key-frequency representation back into every channel before projecting to the forecast horizon."
paper: "https://arxiv.org/abs/2502.16890"
paper_title: "ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting"
venue: "arXiv preprint"
year: 2025
code: "https://github.com/Levi-Ackman/ReFocus"
revision: "5b883b29f364b52a73835f1465e993433f94a1ed"
license: "NOASSERTION"

tagline: "Beta-scaled moving-average residual favors mid frequencies; complex frequency-linear encoder picks key frequencies."
tags: ["mlp", "frequency", "decomposition", "channel-mixing", "normalization"]
composition: ["normalization=component:revin", "decomposition=component:series_decomposition", "temporal=local:frequency-domain-complex-linear-encoder", "channel=component:energy_frequency_pooling", "head=local:complex-frequency-linear-projection", "loss=loss:mse"]
---
# ReFocus

## Key ideas

- AMEO: `x - beta * moving_average(x)` (`EdgePaddedMovingAverage` from `series_decomposition`) attenuates the low-frequency trend band before encoding.
- `FLinear` is a dense complex linear map from every input frequency bin to every output bin, used for the embedding, every encoder sub-layer, and the output `projection`.
- `EncoderBlock` (EKPB) pools one channel's spectrum per frequency bin with a softmax energy distribution across channels via `EnergyBasedFrequencyPooling`, then fuses it with each channel's own features.
- `revin` wraps the model; the output projection starts near identity when `initial=True`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2502.16890); title: ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting; venue/year: arXiv preprint / 2025
- [codebase](https://github.com/Levi-Ackman/ReFocus); revision: `5b883b29f364b52a73835f1465e993433f94a1ed`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ReFocus.toml`](../../../../configs/models/ReFocus.toml).

## Differences

The official repository publishes no LICENSE file (recorded as `NOASSERTION`). It was consulted only as a reference for paper details, and no source was copied; the implementation is an independent rewrite from the paper (see THIRD_PARTY_NOTICES.md).

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

The official repository publishes no LICENSE file (recorded as `NOASSERTION`). It was consulted only as a reference for paper details, and no source was copied; the implementation is an independent rewrite from the paper (see THIRD_PARTY_NOTICES.md).

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
