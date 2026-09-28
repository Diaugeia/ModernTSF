---
name: "DynamicTMoE"
summary: "DynamicTMoE is a clean-room fixed-capacity realization of drift-aware temporal MoE routing with RBF-MMD, recurrent memory, heterogeneous experts, and cyclic relations."
paper: "https://arxiv.org/abs/2605.20678"
paper_title: "Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting"
venue: "ICML 2026"
year: 2026
code: "https://github.com/andone-07/Dynamic-TMoE"
revision: "3e4123530d40c8463cb9487992da49cd967fd9d7"
license: "NOASSERTION"
---
# DynamicTMoE

Dynamic TMoE models non-stationarity through drift perception, temporally coherent expert routing, heterogeneous inductive biases, and training-time expert-pool evolution.

<!-- model-card:canonical:start -->
## Method overview

DynamicTMoE is a clean-room fixed-capacity realization of drift-aware temporal MoE routing with RBF-MMD, recurrent memory, heterogeneous experts, and cyclic relations.

## Core architecture

DynamicTMoE is a clean-room fixed-capacity realization of drift-aware temporal MoE routing with RBF-MMD, recurrent memory, heterogeneous experts, and cyclic relations.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2605.20678); title: Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting; venue/year: ICML 2026 / 2026
- [codebase](https://github.com/andone-07/Dynamic-TMoE); revision: `3e4123530d40c8463cb9487992da49cd967fd9d7`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/DynamicTMoE.toml`](../../../../configs/models/DynamicTMoE.toml).

## Differences

Pinned source inspection: `models/Dynamic_TMoE/model.py`, `models/Dynamic_TMoE/memory_router.py`, `models/Dynamic_TMoE/cyclic_relation.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. The rewrite maps paper equations (1)--(10) to RBF-MMD perception,
GRU/anomaly-memory routing, five heterogeneous experts, concentrated top-k
weights, and cyclic channel-relation refinement.

The paper's training orchestrator creates, aligns, and prunes modules and mutates
the anomaly gallery; `forward` deliberately does none of those stateful actions.
This compact entry uses a fixed five-expert pool, learnable repository, and a
small routing floor to preserve gradients. Evidence is in
`../../../verification/evidence/DynamicTMoE.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `patch_len=16`, `stride=8`, `top_k=3`, `memory_slots=4`, `relation_period=24`, `routing_floor=0.0001`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026
- **arXiv**: https://arxiv.org/abs/2605.20678

## Abstract
Dynamic TMoE introduces an adaptive Mixture of Experts framework designed for time series forecasting in non-stationary environments. The method uses Maximum Mean Discrepancy (MMD) to detect distribution shifts and responds by dynamically expanding or pruning a heterogeneous expert pool, overcoming the rigidity of traditional fixed-capacity MoE designs. A drift-aware routing mechanism selects or allocates experts based on detected statistical changes in the input distribution, enabling robust forecasting under concept drift. The framework was accepted as a poster at the Forty-third International Conference on Machine Learning (ICML 2026) and demonstrates notable improvements in MSE and MAE across nine standard benchmarks compared to prior state-of-the-art methods. The official implementation is available at https://github.com/andone-07/Dynamic-TMoE.

## Source and verification

Pinned source inspection: `models/Dynamic_TMoE/model.py`, `models/Dynamic_TMoE/memory_router.py`, `models/Dynamic_TMoE/cyclic_relation.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. The rewrite maps paper equations (1)--(10) to RBF-MMD perception,
GRU/anomaly-memory routing, five heterogeneous experts, concentrated top-k
weights, and cyclic channel-relation refinement.

The paper's training orchestrator creates, aligns, and prunes modules and mutates
the anomaly gallery; `forward` deliberately does none of those stateful actions.
This compact entry uses a fixed five-expert pool, learnable repository, and a
small routing floor to preserve gradients. Evidence is in
`../../../verification/evidence/DynamicTMoE.json`.

## In ModernTSF
Default config: `configs/models/DynamicTMoE.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@misc{zhu2026dynamictmoe,
  author        = {Jiawen Zhu and Shuhan Liu and Di Weng and Yingcai Wu},
  title         = {Dynamic TMoE: {A} Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting},
  year          = {2026},
  eprint        = {2605.20678},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.20678}
}
```
