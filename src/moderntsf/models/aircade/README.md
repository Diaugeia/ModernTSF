---
name: "AirCade"
summary: "AirCade separates synchronous AQI-weather causality from propagation through uncertain future weather. This local implementation maps paper equations (1)-(13) to domain prompts, four-path DK-MSA, historical Cade, future Cadi, intervention masks, and a point predictor."
paper: "https://doi.org/10.1109/ICASSP49660.2025.11099015"
paper_title: "Spatiotemporal Causal Decoupling Model for Air Quality Forecasting"
venue: "ICASSP 2025"
year: 2025
code: "https://github.com/PoorOtterBob/AirCade"
revision: "179067f5b9fbc05f894022809e0b1c83e9f61fd8"
license: "NOASSERTION"
---
# AirCade

AirCade explicitly separates synchronous AQI--weather causality from its propagation through uncertain future weather. This entry is a clean-room paper implementation; the unlicensed reference repository was inspected at the pinned revision; no external source code was copied while producing it.

<!-- model-card:canonical:start -->
## Method overview

AirCade separates synchronous AQI-weather causality from propagation through uncertain future weather.

## Core architecture

This local implementation maps paper equations (1)-(13) to domain prompts, four-path DK-MSA, historical Cade, future Cadi, intervention masks, and a point predictor.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1109/ICASSP49660.2025.11099015); title: Spatiotemporal Causal Decoupling Model for Air Quality Forecasting; venue/year: ICASSP 2025 / 2025
- [codebase](https://github.com/PoorOtterBob/AirCade); revision: `179067f5b9fbc05f894022809e0b1c83e9f61fd8`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/AirCade.toml`](../../../../configs/models/AirCade.toml).

## Differences

Pinned source inspection: `src/moderntsf/models/AirCade.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

- Local implementation: confirmed from the paper; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Structure evidence maps equations (1)--(13) to prompts, DK-MSA, Cade/Cadi, interventions, and the forecast head. Runtime evidence covers covariates, gradients, serialization, CPU, batch/sequence/node boundaries, and the graph-free adjacency contract.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=6`, `d_model=32`, `prompt_dim=8`, `adaptive_dim=8`, `num_heads=4`, `temporal_layers=2`, `spatial_layers=2`, `environments=3`, `cov_dim=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatiotemporal Causal Decoupling Model for Air Quality Forecasting
- **Venue**: ICASSP 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.20119

## Abstract
Due to the profound impact of air pollution on human health, livelihoods, and economic development, air quality forecasting is of paramount significance. Initially, we employ the causal graph method to scrutinize the constraints of existing research in comprehensively modeling the causal relationships between the air quality index (AQI) and meteorological features. In order to enhance prediction accuracy, we introduce a novel air quality forecasting model, AirCade, which incorporates a causal decoupling approach. AirCade leverages a spatiotemporal module in conjunction with knowledge embedding techniques to capture the internal dynamics of AQI. Subsequently, a causal decoupling module is proposed to disentangle synchronous causality from past AQI and meteorological features, followed by the dissemination of acquired knowledge to future time steps to enhance performance. Additionally, we introduce a causal intervention mechanism to explicitly represent the uncertainty of future meteorological features, thereby bolstering the model's robustness. Our evaluation of AirCade on an open-source air quality dataset demonstrates over 20% relative improvement over state-of-the-art models. Our source code is available at https://github.com/PoorOtterBob/AirCade.

## In ModernTSF
Default config: `configs/models/AirCade.toml`; model specification: `spec.py`; local implementation: `model.py`.

The runtime consumes `x_enc [B, seq_len, N]`, historical marks/covariates `[B, seq_len, N, cov_dim]`, and future covariates `[B, pred_len, N, cov_dim]`; raw six-column marks are converted to two calendar covariates. It returns `[B, pred_len, N]`, permits unequal history/horizon lengths, and intentionally learns spatial matrices rather than consuming adjacency. Equation (1) maps to prompted embeddings, equations (2)-(7) to `DomainKnowledgeAttention`, equations (8)-(11) to Cade/Cadi, equation (12) to the predictor, and equation (13) to relaxed multi-environment masks.

## Source and verification

Pinned source inspection: `src/moderntsf/models/AirCade.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

- Local implementation: confirmed from the paper; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Structure evidence maps equations (1)--(13) to prompts, DK-MSA, Cade/Cadi, interventions, and the forecast head. Runtime evidence covers covariates, gradients, serialization, CPU, batch/sequence/node boundaries, and the graph-free adjacency contract.

## Citation

```bibtex
@inproceedings{DBLP:conf/icassp/MaWHYWWW25,
  author       = {Jiaming Ma and
                  Guanjun Wang and
                  Sheng Huang and
                  Kuo Yang and
                  Binwu Wang and
                  Pengkun Wang and
                  Yang Wang},
  title        = {Spatiotemporal Causal Decoupling Model for Air Quality Forecasting},
  booktitle    = {2025 {IEEE} International Conference on Acoustics, Speech and Signal
                  Processing, {ICASSP} 2025, Hyderabad, India, April 6-11, 2025},
  pages        = {1--5},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/ICASSP49660.2025.11099015},
  doi          = {10.1109/ICASSP49660.2025.11099015},
  timestamp    = {Wed, 11 Feb 2026 11:45:24 +0100},
  biburl       = {https://dblp.org/rec/conf/icassp/MaWHYWWW25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
