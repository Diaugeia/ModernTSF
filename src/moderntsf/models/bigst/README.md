---
name: "BigST"
summary: "BigST is a spatiotemporal learning model designed for large-scale traffic forecasting on road networks. It models both temporal dynamics and spatial dependencies among nodes, scaling to graphs with up to one hundred thousand nodes by replacing the conventional quadratic-complexity graph attention with a linearized random-feature approximation and a pre-computable long-range temporal encoder."
paper: "https://www.vldb.org/pvldb/vol17/p1081-han.pdf"
paper_title: "BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks"
venue: "PVLDB 2024"
year: 2024
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# BigST

BigST is a spatiotemporal learning model designed for large-scale traffic forecasting on road networks. It models both temporal dynamics and spatial dependencies among nodes, scaling to graphs with up to one hundred thousand nodes by replacing the conventional quadratic-complexity graph attention with a linearized random-feature approximation and a pre-computable long-range temporal encoder.

<!-- model-card:canonical:start -->
## Method overview

BigST is a spatiotemporal learning model designed for large-scale traffic forecasting on road networks.

## Core architecture

It models both temporal dynamics and spatial dependencies among nodes, scaling to graphs with up to one hundred thousand nodes by replacing the conventional quadratic-complexity graph attention with a linearized random-feature approximation and a pre-computable long-range temporal encoder.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://www.vldb.org/pvldb/vol17/p1081-han.pdf); title: BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks; venue/year: PVLDB 2024 / 2024
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/BigST.toml`](../../../../configs/models/BigST.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

- **Paper**: the PVLDB article identifies `usail-hkust/BigST` as its released artifact.
- **Implementation**: independent clean-room rewrite from the paper equations; the reference-only repository source was not copied.
- **Formula map**: positive random features evaluate global node attention as `phi(Q)(phi(K)^T V) / phi(Q)(phi(K)^T 1)`, conditioned by learned node and calendar embeddings.
- **Differences**: the separately pretrained long-history extractor, spatial regularization loss, official data pipeline and masked-MAE recipe are omitted. The supplied graph is used only as a residual prior, and the runner owns the objective.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `hid_dim=16`, `node_dim=8`, `time_dim=8`, `tod_size=24`, `dow_size=7`, `tau=1.0`, `random_feature_dim=16`, `dropout=0.1`, `use_residual=True`, `use_bn=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks
- **Venue**: Proceedings of the VLDB Endowment (PVLDB), Vol. 17, No. 5, pp. 1081–1090
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Spatio-Temporal Graph Neural Network (STGNN) has been used as a common workhorse for traffic forecasting. However, most of them require prohibitive quadratic computational complexity to capture long-range spatio-temporal dependencies, thus hindering their applications to long historical sequences on large-scale road networks in the real-world. To this end, in this paper, we propose BigST, a linear complexity spatio-temporal graph neural network, to efficiently exploit long-range spatio-temporal dependencies for large-scale traffic forecasting. Specifically, we first propose a scalable long sequence feature extractor to encode node-wise longrange inputs (e.g., thousands of time-steps in the past week) into low-dimensional representations encompassing rich temporal dynamics. The resulting representations can be pre-computed and hence significantly reduce the computational overhead for prediction. Then, we build a linearized global spatial convolution network to adaptively distill time-varying graph structures, which enables fast runtime message passing along spatial dimensions in linear complexity. We empirically evaluate our model on two large-scale real-world traffic datasets. Extensive experiments demonstrate that BigST can scale to road networks with up to one hundred thousand nodes, while significantly improving prediction accuracy and efficiency compared to state-of-the-art traffic forecasting models.

## In ModernTSF
Default config: `configs/models/BigST.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

- **Paper**: the PVLDB article identifies `usail-hkust/BigST` as its released artifact.
- **Implementation**: independent clean-room rewrite from the paper equations; the reference-only repository source was not copied.
- **Formula map**: positive random features evaluate global node attention as `phi(Q)(phi(K)^T V) / phi(Q)(phi(K)^T 1)`, conditioned by learned node and calendar embeddings.
- **Differences**: the separately pretrained long-history extractor, spatial regularization loss, official data pipeline and masked-MAE recipe are omitted. The supplied graph is used only as a residual prior, and the runner owns the objective.

## Citation

```bibtex
@article{DBLP:journals/pvldb/HanZLTTX24,
  author       = {Jindong Han and
                  Weijia Zhang and
                  Hao Liu and
                  Tao Tao and
                  Naiqiang Tan and
                  Hui Xiong},
  title        = {BigST: Linear Complexity Spatio-Temporal Graph Neural Network for
                  Traffic Forecasting on Large-Scale Road Networks},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {17},
  number       = {5},
  pages        = {1081--1090},
  year         = {2024},
  url          = {https://www.vldb.org/pvldb/vol17/p1081-han.pdf},
  doi          = {10.14778/3641204.3641217},
  timestamp    = {Sun, 19 Jan 2025 13:44:31 +0100},
  biburl       = {https://dblp.org/rec/journals/pvldb/HanZLTTX24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
