---
name: "PCDCNet"
summary: "PCDCNet is a covariate-prediction model for air quality forecasting in a node-structured spatiotemporal setting, where each node is a monitoring station. It integrates numerical modeling principles (emissions, meteorological influences, and physical-chemical domain constraints) with deep learning components — specifically graph-based spatial transport, recurrent temporal accumulation, and local interaction representation enhancement — to forecast 72-hour PM2.5 and O3 concentrations at the station level."
paper: "https://arxiv.org/abs/2505.19842"
paper_title: "PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints"
venue: "arXiv preprint"
year: 2025
code: "https://github.com/PoorOtterBob/CauAir"
revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
license: "NOASSERTION"
---
# PCDCNet

PCDCNet is a covariate-prediction model for air quality forecasting in a node-structured spatiotemporal setting, where each node is a monitoring station. It integrates numerical modeling principles (emissions, meteorological influences, and physical-chemical domain constraints) with deep learning components — specifically graph-based spatial transport, recurrent temporal accumulation, and local interaction representation enhancement — to forecast 72-hour PM2.5 and O3 concentrations at the station level.

<!-- model-card:canonical:start -->
## Method overview

PCDCNet is a covariate-prediction model for air quality forecasting in a node-structured spatiotemporal setting, where each node is a monitoring station.

## Core architecture

It integrates numerical modeling principles (emissions, meteorological influences, and physical-chemical domain constraints) with deep learning components — specifically graph-based spatial transport, recurrent temporal accumulation, and local interaction representation enhancement — to forecast 72-hour PM2.5 and O3 concentrations at the station level.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://arxiv.org/abs/2505.19842); title: PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints; venue/year: arXiv preprint / 2025
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PCDCNet.toml`](../../../../configs/models/PCDCNet.toml).

## Differences

Clean-room implementation: confirmed. The reference-only source code was not copied.

The implementation is independently derived from the paper equations; no
author implementation was identified and no CauAir source was copied. It keeps
the LID/STD/TAD update order and exposes `domain_informed_constraint()` after a
forward pass. Standard calendar covariates are a reduced substitute for the
paper's meteorology and emissions inputs.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `d_model=64`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.19842

## Abstract
Air quality forecasting (AQF) is critical for public health and environmental management, yet remains challenging due to the complex interplay of emissions, meteorology, and chemical transformations. Traditional numerical models, such as CMAQ and WRF-Chem, provide physically grounded simulations but are computationally expensive and rely on uncertain emission inventories. Deep learning models, while computationally efficient, often struggle with generalization due to their lack of physical constraints. To bridge this gap, we propose PCDCNet, a surrogate model that integrates numerical modeling principles with deep learning. PCDCNet explicitly incorporates emissions, meteorological influences, and domain-informed constraints to model pollutant formation, transport, and dissipation. By combining graph-based spatial transport modeling, recurrent structures for temporal accumulation, and representation enhancement for local interactions, PCDCNet achieves state-of-the-art (SOTA) performance in 72-hour station-level PM2.5 and O3 forecasting while significantly reducing computational costs. Furthermore, our model is deployed in an online platform, providing free, real-time air quality forecasts, demonstrating its scalability and societal impact. By aligning deep learning with physical consistency, PCDCNet offers a practical and interpretable solution for AQF, enabling informed decision-making for both personal and regulatory applications.

## In ModernTSF
Default config: `configs/models/PCDCNet.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. The reference-only source code was not copied.

The implementation is independently derived from the paper equations; no
author implementation was identified and no CauAir source was copied. It keeps
the LID/STD/TAD update order and exposes `domain_informed_constraint()` after a
forward pass. Standard calendar covariates are a reduced substitute for the
paper's meteorology and emissions inputs.

## Citation

```bibtex
@misc{wang2025pcdcnet,
  author        = {Shuo Wang and
                  Yun Cheng and
                  Qingye Meng and
                  Olga Saukh and
                  Jiang Zhang and
                  Jingfang Fan and
                  Yuanting Zhang and
                  Xingyuan Yuan and
                  Lothar Thiele},
  title         = {PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints},
  year          = {2025},
  eprint        = {2505.19842},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.19842}
}
```
