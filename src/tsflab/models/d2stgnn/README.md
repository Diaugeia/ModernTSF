---
name: "D2STGNN"
summary: "D2STGNN (Decoupled Dynamic Spatial-Temporal Graph Neural Network) is a spatiotemporal learning model designed for node-structured graph data such as road-sensor traffic networks. It explicitly separates traffic signals into diffusion signals (vehicles propagating through the network) and inherent signals (local non-diffusion patterns) via a learned estimation gate and residual decomposition, then processes each component with a dedicated module while a dynamic graph learning sub-network captures time-varying spatial topology."
paper: "https://www.vldb.org/pvldb/vol15/p2733-shao.pdf"
paper_title: "Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting"
venue: "VLDB 2022"
year: 2022
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "79641b1c75246ab2d8c53bb52f2ac72588be0cdc"
license: "Apache-2.0"
tagline: "Decouples diffusion and inherent signals with an estimation gate and residual backcasts on a dynamic graph."
tags: ["gnn", "rnn", "cnn", "decomposition", "graph-learning", "spatiotemporal", "covariates"]
composition: ["normalization=none", "decomposition=local:diffusion-inherent-estimation-gate-residual-decomposition", "temporal=local:gru-conv-inherent-branch", "channel=component:graph_utils+component:adaptive_node_embedding_adjacency+local:dynamic-hidden-state-graph", "head=local:accumulated-layer-forecasts-mlp", "loss=loss:mse"]
---
# D2STGNN

## Key ideas

- `DecoupledLayer` splits each hidden state into a diffusion part (multi-step graph propagation over four graphs) and an inherent part (`GRU` plus temporal `Conv1d`), mixed by a sigmoid estimation gate.
- The four graphs are forward and reverse transition supports (`graph_utils`), a self-adaptive graph (`adaptive_node_embedding_adjacency`) and a `DynamicGraphConstructor` graph built from the hidden state.
- Each layer subtracts a `backcast` of both signals from the residual before the next layer, and emits a partial forecast from its last step.
- Partial forecasts from all layers are concatenated and decoded by an MLP; time-of-day and day-of-week embeddings come from calendar marks.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://www.vldb.org/pvldb/vol15/p2733-shao.pdf); title: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting; venue/year: VLDB 2022 / 2022
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `79641b1c75246ab2d8c53bb52f2ac72588be0cdc`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/D2STGNN.toml`](../../../../configs/models/D2STGNN.toml).

## Differences

TSFLab rewrites D2STGNN locally after reviewing the paper and pinned official codebase. The implementation retains hidden-state dynamic graph construction, diffusion/inherent branches, estimation gating, residual backcasts, and accumulated partial forecasts. Dataset loading and training schedules remain outside the model. Canonical evidence is stored in [`verification/evidence/D2STGNN.json`](../../../../verification/evidence/D2STGNN.json).

## Shared components

- [`adaptive_node_embedding_adjacency`](../_components/adaptive_node_embedding_adjacency/README.md)
- [`graph_utils`](../_components/graph_utils/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `num_feat=1`, `num_hidden=16`, `node_hidden=8`, `time_emb_dim=8`, `k_s=2`, `k_t=3`, `gap=1`, `num_layers=2`, `dropout=0.1`, `time_in_day_size=288`, `day_in_week_size=7`, `forecast_dim=64`, `output_hidden=128`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting
- **Venue**: VLDB 2022
- **Published**: 2022 (arXiv: 2022-06)
- **arXiv**: https://arxiv.org/abs/2206.09112

## Abstract
We all depend on mobility, and vehicular transportation affects the daily lives of most of us. Thus, the ability to forecast the state of traffic in a road network is an important functionality and a challenging task. Traffic data is often obtained from sensors deployed in a road network. Recent proposals on spatial-temporal graph neural networks have achieved great progress at modeling complex spatial-temporal correlations in traffic data, by modeling traffic data as a diffusion process. However, intuitively, traffic data encompasses two different kinds of hidden time series signals, namely the diffusion signals and inherent signals. Unfortunately, nearly all previous works coarsely consider traffic signals entirely as the outcome of the diffusion, while neglecting the inherent signals, which impacts model performance negatively. To improve modeling performance, we propose a novel Decoupled Spatial-Temporal Framework (DSTF) that separates the diffusion and inherent traffic information in a data-driven manner, which encompasses a unique estimation gate and a residual decomposition mechanism. The separated signals can be handled subsequently by the diffusion and inherent modules separately. Further, we propose an instantiation of DSTF, Decoupled Dynamic Spatial-Temporal Graph Neural Network (D2STGNN), that captures spatial-temporal correlations and also features a dynamic graph learning module that targets the learning of the dynamic characteristics of traffic networks. Extensive experiments with four real-world traffic datasets demonstrate that the framework is capable of advancing the state-of-the-art.

## In TSFLab
Default config: `configs/models/D2STGNN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

TSFLab rewrites D2STGNN locally after reviewing the paper and pinned official codebase. The implementation retains hidden-state dynamic graph construction, diffusion/inherent branches, estimation gating, residual backcasts, and accumulated partial forecasts. Dataset loading and training schedules remain outside the model. Canonical evidence is stored in [`verification/evidence/D2STGNN.json`](../../../../verification/evidence/D2STGNN.json).

## Citation

```bibtex
@article{DBLP:journals/pvldb/ShaoZWWXCJ22,
  author       = {Zezhi Shao and
                  Zhao Zhang and
                  Wei Wei and
                  Fei Wang and
                  Yongjun Xu and
                  Xin Cao and
                  Christian S. Jensen},
  title        = {Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic
                  Forecasting},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {15},
  number       = {11},
  pages        = {2733--2746},
  year         = {2022},
  url          = {https://www.vldb.org/pvldb/vol15/p2733-shao.pdf},
  doi          = {10.14778/3551793.3551827},
  timestamp    = {Sat, 06 Sep 2025 20:28:21 +0200},
  biburl       = {https://dblp.org/rec/journals/pvldb/ShaoZWWXCJ22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
