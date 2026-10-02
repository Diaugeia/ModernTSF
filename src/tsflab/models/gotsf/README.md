---
name: "GOTSF"
summary: "GOTSF is a time series forecasting model built around a goal-oriented training methodology that enables forecasting models to bias their predictions toward application-specified regions of interest at inference time without retraining. The model partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated at inference time to emphasize whichever target range an application specifies."
paper: "https://arxiv.org/abs/2504.17493"
paper_title: "Goal-Oriented Time-Series Forecasting: Foundation Framework Design"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/netop-team/gotsf"
revision: "31b17e55a0cb6f41bfe25230db3f81567efd58f3"
license: "MIT"
tagline: "Interval-conditioned experts with membership confidences, reweighted at inference to emphasize a chosen target range."
tags: ["mlp", "goal-oriented", "channel-independent", "mixture-of-experts"]
composition: ["normalization=none", "decomposition=none", "temporal=local:interval-conditioned-mlp-encoder", "channel=local:channel-independent-shared-weights", "head=local:interval-regression-and-confidence-heads", "loss=local:interval-soft-boundary-regression-and-membership"]
---
# GOTSF

## Key ideas

- Splits the value range into `num_intervals` disjoint bins; each bin's midpoint and half-width condition a shared channel-independent MLP (`history_encoder` plus `interval_encoder`).
- A regression head and a membership-confidence head produce a forecast and a confidence per interval (`interval_outputs`).
- `forecast_interval` averages the forecasts of bins intersecting a target interval, weighted by confidence, so the emphasized range can change without retraining; plain `forward` uses all bins.
- `goal_oriented_loss` trains one bin with an exponential soft-boundary weight (`decay`) plus a membership BCE term; the model's `training_objective` averages it over all bins (replacing the configured criterion while training); validation and test score `forward` with the standard loss.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2504.17493); title: Goal-Oriented Time-Series Forecasting: Foundation Framework Design; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/netop-team/gotsf); revision: `31b17e55a0cb6f41bfe25230db3f81567efd58f3`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GOTSF.toml`](../../../../configs/models/GOTSF.toml).

## Differences

Pinned source inspection: `experiments/exp.py`, `experiments/exp_long_term_forecasting_discrete.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This is an independent implementation from Eqs. (8)--(14) of the paper. It
uses disjoint interval conditions, the exponential soft-boundary weight,
membership confidence, and confidence-weighted averaging over intersecting
bins. The common `forward` call covers the full configured forecasting range;
applications can call `forecast_interval(..., target_interval=...)` or train a selected interval with
`goal_oriented_loss`. The runner objective averages the per-bin loss over all bins each step (the official trainer samples one random bin per example), using element-wise soft-boundary weights on an L1 term and an unweighted BCE membership term as in the official trainer. A latent interval embedding replaces the paper experiment's
repeated interval-bound channels. The paper permits multiple host forecasters,
while this entry intentionally uses a compact channel-independent MLP and does not claim
the paper's dataset recipes or reported accuracy. The reference-only project
was inspected at the pinned revision; no external source code was copied. Strict evidence is recorded in
`../../../verification/evidence/GOTSF.json`.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_intervals=4`, `interval_min=-2.0`, `interval_max=2.0`, `decay_rate=50.0`, `classification_weight=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Goal-Oriented Time-Series Forecasting: Foundation Framework Design
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-04)
- **arXiv**: https://arxiv.org/abs/2504.17493

## Abstract
Conventional time-series forecasting methods typically aim to minimize overall prediction error, without accounting for the varying importance of different forecast ranges in downstream applications. We propose a training methodology that enables forecasting models to adapt their focus to application-specific regions of interest at inference time, without retraining. The approach partitions the prediction space into fine-grained segments during training, which are dynamically reweighted and aggregated to emphasize the target range specified by the application. Unlike prior methods that predefine these ranges, our framework supports flexible, on-demand adjustments. Experiments on standard benchmarks and a newly collected wireless communication dataset demonstrate that our method not only improves forecast accuracy within regions of interest but also yields measurable gains in downstream task performance. These results highlight the potential for closer integration between predictive modeling and decision-making in real-world systems.

## Source and verification

Pinned source inspection: `experiments/exp.py`, `experiments/exp_long_term_forecasting_discrete.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This is an independent implementation from Eqs. (8)--(14) of the paper. It
uses disjoint interval conditions, the exponential soft-boundary weight,
membership confidence, and confidence-weighted averaging over intersecting
bins. The common `forward` call covers the full configured forecasting range;
applications can call `forecast_interval(..., target_interval=...)` or train a selected interval with
`goal_oriented_loss`. The runner objective averages the per-bin loss over all bins each step (the official trainer samples one random bin per example), using element-wise soft-boundary weights on an L1 term and an unweighted BCE membership term as in the official trainer. A latent interval embedding replaces the paper experiment's
repeated interval-bound channels. The paper permits multiple host forecasters,
while this entry intentionally uses a compact channel-independent MLP and does not claim
the paper's dataset recipes or reported accuracy. The reference-only project
was inspected at the pinned revision; no external source code was copied. Strict evidence is recorded in
`../../../verification/evidence/GOTSF.json`.

## In TSFLab
Default config: `configs/models/GOTSF.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/FecheteSAPLDS26,
  author       = {Luca{-}Andrei Fechete and
                  Mohamed Sana and
                  Fadhel Ayed and
                  Nicola Piovesan and
                  Wenjie Li and
                  Antonio De Domenico and
                  Tareq Si Salem},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Goal-Oriented Time-Series Forecasting: Foundation Framework Design},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {21065--21073},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i25.39249},
  doi          = {10.1609/AAAI.V40I25.39249},
  timestamp    = {Fri, 27 Mar 2026 17:13:39 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/FecheteSAPLDS26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
