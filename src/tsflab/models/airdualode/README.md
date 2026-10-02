---
name: "AirDualODE"
summary: "Air-DualODE combines explicit open-system pollutant dynamics with a complementary data-driven latent ODE. This local implementation retains BA-DAE diffusion, directed advection and source/sink correction, masked learned dynamics, temporal rollout, and geographic graph fusion."
paper: "https://openreview.net/forum?id=kOJf7Dklyv"
paper_title: "Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems"
venue: "ICLR 2025"
year: 2025
code: "https://github.com/decisionintelligence/Air-DualODE"
revision: "3accfef5d3ab40f685ea29f302f76287706ba821"
license: "NOASSERTION"
---
# AirDualODE

Air-DualODE models an open pollutant system with a physical BA-DAE branch and a separate data-driven latent ODE, then fuses both node representations on the geographic graph. This is a paper-derived local implementation; the unlicensed reference code was inspected at the pinned revision; no external source code was copied.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://openreview.net/forum?id=kOJf7Dklyv); title: Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems; venue/year: ICLR 2025 / 2025
- [codebase](https://github.com/decisionintelligence/Air-DualODE); revision: `3accfef5d3ab40f685ea29f302f76287706ba821`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AirDualODE.toml`](../../../../configs/models/AirDualODE.toml).

## Differences

Pinned source inspection: `models/Air_DualODE.py`, `models/layers/Explicit_odefunc.py`, `models/layers/Unk_Dynamics.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

- Local implementation: confirmed from equations (6)--(10) and the graph-fusion description; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Structure and runtime evidence verify BA-DAE terms independently, both dynamics paths, graph sensitivity, covariates, all active gradients, serialization, CPU, and boundary cases.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `phy_latent_dim=16`, `unk_latent_dim=16`, `gcn_hidden_dim=32`, `n_heads=4`, `ode_method='euler'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems
- **Venue**: ICLR 2025
- **Published**: 2025 (arXiv: 2024-10)
- **arXiv**: https://arxiv.org/abs/2410.19892

## Abstract
Air pollution significantly threatens human health and ecosystems, necessitating effective air quality prediction to inform public policy. Traditional approaches are generally categorized into physics-based and data-driven models. Physics-based models usually struggle with high computational demands and closed-system assumptions, while data-driven models may overlook essential physical dynamics, confusing the capturing of spatiotemporal correlations. Although some physics-guided approaches combine the strengths of both models, they often face a mismatch between explicit physical equations and implicit learned representations. To address these challenges, we propose Air-DualODE, a novel physics-guided approach that integrates dual branches of Neural ODEs for air quality prediction. The first branch applies open-system physical equations to capture spatiotemporal dependencies for learning physics dynamics, while the second branch identifies the dependencies not addressed by the first in a fully data-driven way. These dual representations are temporally aligned and fused to enhance prediction accuracy. Our experimental results demonstrate that Air-DualODE achieves state-of-the-art performance in predicting pollutant concentrations across various spatial scales, thereby offering a promising solution for real-world air quality challenges.

## In TSFLab
Default config: `configs/models/AirDualODE.toml`; model specification: `spec.py`; local implementation: `model.py`.

Inputs are `x_enc [B, seq_len, N]` plus raw or node-structured meteorology; distance adjacency and directed wind/flow adjacency are construction inputs. Output is `[B, pred_len, N]`. Equation (6) maps to `BoundaryAwareDynamics`, equations (7)-(8) to the explicit rollout/projection, equations (9)-(10) to the GRU and masked-attention latent ODE, and the described GNN fusion to `graph_fusion`. Decay-TCL remains a separately declared training loss rather than hidden forward behavior.

## Source and verification

Pinned source inspection: `models/Air_DualODE.py`, `models/layers/Explicit_odefunc.py`, `models/layers/Unk_Dynamics.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

- Local implementation: confirmed from equations (6)--(10) and the graph-fusion description; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Structure and runtime evidence verify BA-DAE terms independently, both dynamics paths, graph sensitivity, covariates, all active gradients, serialization, CPU, and boundary cases.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/TianL0CGZPRY25,
  author       = {Jindong Tian and
                  Yuxuan Liang and
                  Ronghui Xu and
                  Peng Chen and
                  Chenjuan Guo and
                  Aoying Zhou and
                  Lujia Pan and
                  Zhongwen Rao and
                  Bin Yang},
  title        = {Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open
                  Systems},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025, Singapore, April 24-28, 2025},
  publisher    = {OpenReview.net},
  year         = {2025},
  url          = {https://openreview.net/forum?id=kOJf7Dklyv},
  timestamp    = {Fri, 14 Nov 2025 07:30:22 +0100},
  biburl       = {https://dblp.org/rec/conf/iclr/TianL0CGZPRY25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
