---
name: "FreqMoE"
summary: "FreqMoE first reconstructs a denoised series from a learned, input-gated Mixture of Experts over contiguous rFFT frequency bands, then forecasts the horizon with a small stack of residual frequency-extension blocks that each upsample the spectrum from seq_len to seq_len+pred_len bins with a complex linear layer, a complex ReLU/dropout nonlinearity, and a second complex linear refinement before inverting with irfft; each block's backcast residual feeds the next block and every block's forecast segment is summed into the final prediction."
paper: "https://arxiv.org/abs/2501.15125"
paper_title: "FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts"
venue: "AISTATS 2025"
year: 2025
code: "https://github.com/sunbus100/FreqMoE-main"
revision: "b34e93703159a22fdc9f97f0be4fa32b5600a3bf"
license: "unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)"

---
# FreqMoE

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2501.15125); title: FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts; venue/year: AISTATS 2025 / 2025
- [codebase](https://github.com/sunbus100/FreqMoE-main); revision: `b34e93703159a22fdc9f97f0be4fa32b5600a3bf`; license: `unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/FreqMoE.toml`](../../../../configs/models/FreqMoE.toml).

## Differences

Inspected official file: `models/FreqMoE.py` (`FreqDecompMoE`, `Block`,
`ComplexReLU`, `ComplexDropout`, `Model`), at revision
`b34e93703159a22fdc9f97f0be4fa32b5600a3bf`.

**Paper-driven local implementation.** The official `band_boundaries`
parameter is cast to `long()` indices before being used to slice the
frequency axis; this integer cast blocks autograd, so — exactly matching the
official behavior — `band_boundaries` never receives a gradient from the
training objective even though it is a registered parameter. This is a
property of the official design (confirmed by reading `models/FreqMoE.py`
line-by-line), not an omission introduced locally, and is recorded here so
gradient-flow checks intentionally exclude that one parameter. `ComplexReLU`
and `ComplexDropout` remain tiny, paper-specific glue and stay model-local
rather than becoming shared components. The external repository is
reference-only; no source file was copied or adapted.

## Shared components

- [`freq_band_moe`](../_components/freq_band_moe/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `expert_num=4`, `freq_num_blocks=1`, `dropout_freq=0.1`
<!-- model-card:canonical:end -->

## Source and verification

Inspected official file: `models/FreqMoE.py` (`FreqDecompMoE`, `Block`,
`ComplexReLU`, `ComplexDropout`, `Model`), at revision
`b34e93703159a22fdc9f97f0be4fa32b5600a3bf`.

**Paper-driven local implementation.** The official `band_boundaries`
parameter is cast to `long()` indices before being used to slice the
frequency axis; this integer cast blocks autograd, so — exactly matching the
official behavior — `band_boundaries` never receives a gradient from the
training objective even though it is a registered parameter. This is a
property of the official design (confirmed by reading `models/FreqMoE.py`
line-by-line), not an omission introduced locally, and is recorded here so
gradient-flow checks intentionally exclude that one parameter. `ComplexReLU`
and `ComplexDropout` remain tiny, paper-specific glue and stay model-local
rather than becoming shared components. The external repository is
reference-only; no source file was copied or adapted.
