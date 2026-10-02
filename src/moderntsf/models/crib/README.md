---
name: "CRIB"
summary: "CRIB forecasts directly from partially observed multivariate series. It embeds non-overlapping value/missingness patches with temporal convolutions, applies unified-variate attention across every channel-patch token, learns a Gaussian information-bottleneck latent, and predicts with an MLP. Random-mask and Gaussian-noise views provide the consistency objective."
paper: "https://arxiv.org/abs/2509.23494"
paper_title: "Revisiting Multivariate Time Series Forecasting with Missing Values"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/Muyiiiii/CRIB"
revision: "a457672c7b0152f74c929858dba2a9c886405519"
license: "NOASSERTION"
---
# CRIB

CRIB forecasts directly from partially observed multivariate series. Missing
entries use NaNs through the common four-input interface. Model-specific callers
that already hold an explicit observation mask may use `forecast_masked`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.23494); title: Revisiting Multivariate Time Series Forecasting with Missing Values; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/Muyiiiii/CRIB); revision: `a457672c7b0152f74c929858dba2a9c886405519`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CRIB.toml`](../../../../configs/models/CRIB.toml).

## Differences

Pinned source inspection: `TSL_models/CRIB.py`, `TSL_models/CRIB_module.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

**Local implementation: confirmed.** The linked repository has no explicit
license and is `reference-only`; its source was inspected at the pinned revision; no external source code was copied. The
local implementation maps paper Eqs. 3--5 to temporal patch encoding,
all-channel/all-patch attention, and the predictor; Eqs. 7--9 to a diagonal
Gaussian bottleneck; and Eqs. 11--12 to augmented-view consistency plus KL
`aux_loss`. Dataset-specific missingness generation is outside the model; the
runtime instead accepts NaNs or a same-shaped observation mask. Published
training schedules, checkpoints, and metric reference comparison are not claimed.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=8`, `model_dim=32`, `heads_num=4`, `enc_num=2`, `dropout=0.1`, `activation='relu'`, `consis_weight=1.0`, `kl_weight=1e-06`, `augmentation_rate=0.1`
<!-- model-card:canonical:end -->

## Training objective
`L = IB_weight · MAE(Ŷ, Y) + Consis_weight · MSE(enc_clean, enc_noisy) + KL_weight · KL(q(z|x)‖N(0,I))`
(defaults `IB_weight=1`, `Consis_weight=1`, `KL_weight=1e-6`).

## In ModernTSF
Default config: `configs/models/CRIB.toml`; specification: `spec.py`; implementation:
`model.py`.

The consistency and KL terms are computed in `forward` and exposed through the
trainer's `aux_loss` convention; the prediction term remains the configured
training loss (use MAE for the paper objective). `patch_len` must divide
`seq_len`, and `model_dim` must be divisible by `heads_num`.

Official reference reference: https://github.com/Muyiiiii/CRIB

## Source and verification

Pinned source inspection: `TSL_models/CRIB.py`, `TSL_models/CRIB_module.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

**Local implementation: confirmed.** The linked repository has no explicit
license and is `reference-only`; its source was inspected at the pinned revision; no external source code was copied. The
local implementation maps paper Eqs. 3--5 to temporal patch encoding,
all-channel/all-patch attention, and the predictor; Eqs. 7--9 to a diagonal
Gaussian bottleneck; and Eqs. 11--12 to augmented-view consistency plus KL
`aux_loss`. Dataset-specific missingness generation is outside the model; the
runtime instead accepts NaNs or a same-shaped observation mask. Published
training schedules, checkpoints, and metric reference comparison are not claimed.
