---
name: "LIFT"
summary: "LIFT is a plug-in that refines a backbone's multivariate forecast with leading indicators. A non-parametric lead estimator finds, for every variate, the variates that lead it by some number of steps (largest lagged cross-correlation), the leaders are shifted into alignment with the horizon, and an adaptive frequency mixer with state-dependent filters merges them into the normalized forecast. This entry pairs the refiner with the DLinear backbone used in the official examples."
paper: "https://arxiv.org/abs/2401.17548"
paper_title: "Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators"
venue: "ICLR 2024"
year: 2024
code: "https://github.com/SJTU-Quant/LIFT"
revision: "79cf8f157bb9a616c9733daf1e9c2cfefdc98c94"
license: "NOASSERTION"
tagline: "DLinear forecast refined by FFT-estimated, lag-shifted leading variates through state-gated frequency filters."
tags: ["linear", "plug-in", "channel-dependence", "lead-lag", "frequency", "lightweight"]
composition: ["normalization=local:instance-normalization-without-affine", "decomposition=none", "temporal=component:dlinear", "channel=local:lead-lag-shifted-leaders", "head=local:adaptive-frequency-mixer", "loss=loss:mse"]
---
# LIFT

## Key ideas

- `Model.estimate_leaders` computes circular cross-correlations of all variate pairs by FFT, keeps lag-local maxima, and picks the top-K leaders with their leading steps and signs (O(L log L) per pair).
- `Model.shift_leaders` shifts each leader by its leading step over the observed window concatenated with the preliminary forecast, and flips negatively correlated leaders.
- `Model.lead_filters` and `Model.refine` generate 2K+1 frequency filters from the leader correlations and a per-variate state distribution, filter the spectra of the forecast, leaders, and leader differences, and mix them with a complex linear map as a residual on the normalized forecast.
- The backbone is the `dlinear` component; the refiner is trained jointly with it.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2401.17548); title: Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/SJTU-Quant/LIFT); revision: `79cf8f157bb9a616c9733daf1e9c2cfefdc98c94`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/LIFT.toml`](../../../../configs/models/LIFT.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`dlinear`](../_components/dlinear/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `leader_num=4`, `state_num=8`, `temperature=1.0`, `kernel_size=25`
<!-- model-card:canonical:end -->
