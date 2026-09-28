---
name: "PWS"
summary: "PWS (Patch Weighted Sum) is a deliberately minimal in-repo baseline for univariate and multivariate time-series forecasting. It splits the look-back window period-wise into fixed-size patches, refines each patch with a small analysis MLP, and produces the forecast with a learned map from historical periods to future periods. It has optional RevIN normalization but no attention or convolution."
paper: ""
paper_title: "Patch Weighted Sum (ModernTSF baseline)"
venue: "ModernTSF"
year: 2026
---
# PWS

PWS (Patch Weighted Sum) is a deliberately minimal in-repo baseline for univariate and multivariate time-series forecasting. It splits the look-back window period-wise into fixed-size patches, refines each patch with a small analysis MLP, and produces the forecast with a learned map from historical periods to future periods. It has optional RevIN normalization but no attention or convolution.

<!-- model-card:canonical:start -->
## Method overview

PWS (Patch Weighted Sum) is a deliberately minimal in-repo baseline for univariate and multivariate time-series forecasting.

## Core architecture

It splits the look-back window period-wise into fixed-size patches, refines each patch with a small analysis MLP, and produces the forecast with a learned map from historical periods to future periods. It has optional RevIN normalization but no attention or convolution.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Patch Weighted Sum (ModernTSF baseline); venue/year: ModernTSF / 2026
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PWS.toml`](../../../../configs/models/PWS.toml).

## Differences

- Local implementation. PWS is an intentional ModernTSF baseline with no external paper or official reference repository. For patch `k`, it evaluates `A_k(X_k) + X_k` followed by `Y_k = W_k(A_k(X_k)+X_k)+b_k` along the historical-period axis.
- `analysis_hidden` is a typed list and `analysis_act` accepts only implemented activations, preventing silent no-op configurations.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `period=24`, `patch_size=6`, `revin=True`, `affine=False`, `subtract_last=False`, `analysis_act='relu'`, `analysis_hidden=[512, 256]`
<!-- model-card:canonical:end -->

## Paper
PWS (Patch Weighted Sum) has no associated publication. It is a deliberately simple baseline implemented directly in ModernTSF, with no official reference codebase or external paper.
- **Venue**: N/A (simple in-repo baseline)
- **arXiv**: N/A

## Abstract
PWS partitions the look-back window into complete periods of a user-specified length and divides each period into non-overlapping patches. A patch-specific residual MLP analyzes the historical-period axis, followed by a learned linear map to future periods. The final prediction concatenates patch outputs across the period and trims to the horizon. Optional RevIN handles distribution shift.

## In ModernTSF
Default config: `configs/models/PWS.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

- Local implementation. PWS is an intentional ModernTSF baseline with no external paper or official reference repository. For patch `k`, it evaluates `A_k(X_k) + X_k` followed by `Y_k = W_k(A_k(X_k)+X_k)+b_k` along the historical-period axis.
- `analysis_hidden` is a typed list and `analysis_act` accepts only implemented activations, preventing silent no-op configurations.

## Citation

PWS is an in-repository baseline and has no associated paper or canonical
BibTeX entry.
