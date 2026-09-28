---
name: "COSA"
summary: "COSA is a clean-room output-space correction method implementing a context-conditioned linear residual and bounded scalar gate around a frozen base forecast."
paper: "https://openreview.net/forum?id=L7Z5wBMPrW"
paper_title: "COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/bigbases/COSA_ICLR2026"
revision: "527c0feb9e997dd85af485ee027616b446e4ae77"
license: "CC-BY-NC-4.0"
---
# COSA

COSA corrects a frozen forecast directly in output space using recently revealed context statistics; it is an adaptation method rather than a standalone backbone.

<!-- model-card:canonical:start -->
## Method overview

COSA is a clean-room output-space correction method implementing a context-conditioned linear residual and bounded scalar gate around a frozen base forecast.

## Core architecture

COSA is a clean-room output-space correction method implementing a context-conditioned linear residual and bounded scalar gate around a frozen base forecast.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=L7Z5wBMPrW); title: COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/bigbases/COSA_ICLR2026); revision: `527c0feb9e997dd85af485ee027616b446e4ae77`; license: `CC-BY-NC-4.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/COSA.toml`](../../../../configs/models/COSA.toml).

## Differences

Pinned source inspection: `tta/cosa.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. `correct` implements the paper equation exactly:
`H=W[Y0||C]+b`, `Y=Y0+tanh(g)H`; only the residual and gate are trainable.

The repository does not run the paper's streaming buffer, delayed-label update
loop, PAAS, CALR, or repeated optimizer steps. A frozen last-value base and
latest-input context are explicit self-contained fallbacks; callers can pass an
external frozen forecast and revealed-label context. Evidence is in
`../../../verification/evidence/COSA.json`.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `context_len=10`, `gate_init=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026
- **OpenReview**: https://openreview.net/forum?id=L7Z5wBMPrW

## Abstract
Deployed time-series forecasters suffer performance degradation under non-stationarity and distribution shifts. Test-time adaptation (TTA) for time-series forecasting differs from vision TTA because ground truth becomes observable shortly after prediction. Existing time-series TTA methods typically employ dual input/output adapters that indirectly modify data distributions, making their effect on the frozen model difficult to analyze. We introduce the Context-aware Output-Space Adapter (COSA), a minimal, plug-and-play adapter that directly corrects predictions of a frozen base model. COSA performs residual correction modulated by gating, utilizing the original prediction and a lightweight context vector that summarizes statistics from recently observed ground truth. At test time, only the adapter parameters (linear layer and gating) are updated under a leakage-free protocol, using observed ground truth with an adaptive learning rate schedule for faster adaptation. Across diverse scenarios, COSA demonstrates substantial performance gains versus baselines without TTA (13.91∼17.03%) and SOTA TTA methods (10.48∼13.05%), with particularly large improvements at long horizons, while adding a reasonable level of parameters and negligible computational overhead. The simplicity of COSA makes it architecture-agnostic and deployment-friendly.

## Source and verification

Pinned source inspection: `tta/cosa.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. `correct` implements the paper equation exactly:
`H=W[Y0||C]+b`, `Y=Y0+tanh(g)H`; only the residual and gate are trainable.

The repository does not run the paper's streaming buffer, delayed-label update
loop, PAAS, CALR, or repeated optimizer steps. A frozen last-value base and
latest-input context are explicit self-contained fallbacks; callers can pass an
external frozen forecast and revealed-label context. Evidence is in
`../../../verification/evidence/COSA.json`.

## In ModernTSF
Default config: `configs/models/COSA.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@inproceedings{im2026cosa,
  author    = {Jeonghwan Im and Hyuk-Yoon Kwon},
  title     = {{COSA}: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=L7Z5wBMPrW}
}
```
