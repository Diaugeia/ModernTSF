---
name: "PM25_GNN"
summary: "PM25_GNN is a graph neural network model for air quality (PM2.5 concentration) forecasting that integrates domain knowledge about pollutant diffusion processes to construct the graph topology and combines GNN layers with GRU-based temporal modeling to capture both fine-grained and long-term spatial-temporal dependencies across monitoring stations."
paper: "https://doi.org/10.1145/3397536.3422208"
paper_title: "PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting"
venue: "ACM SIGSPATIAL 2020"
year: 2020
code: "https://github.com/shuowang-ai/PM2.5-GNN"
revision: "471fc60775f80492f4f224203d172868bc6eebac"
license: "MIT"
tagline: "Graph-GRU encoder over station history and graph-GRU decoder rolled out hour by hour with known-future covariates."
tags: ["gnn", "rnn", "spatiotemporal", "covariates", "autoregressive", "air-quality"]
composition: ["normalization=none", "decomposition=none", "temporal=local:graph-gru-encoder-decoder", "channel=local:fixed-adjacency-graph-propagation", "head=local:autoregressive-linear-readout", "loss=loss:mse"]
---
# PM25_GNN

## Key ideas

- `GraphGRUCell` concatenates each node's input and state with its neighbours' (row-normalized `adj_mx` plus self loops) in both gates.
- The encoder runs over the lookback with station values and time covariates built by `to_spatiotemporal`.
- The decoder rolls out `pred_len` steps, feeding its previous prediction and the known-future covariates (`future_time_features`) into the next step.
- Graph topology is fixed and supplied through `adj_mx` (identity when absent); nothing is learned about the graph.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1145/3397536.3422208); title: PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting; venue/year: ACM SIGSPATIAL 2020 / 2020
- [codebase](https://github.com/shuowang-ai/PM2.5-GNN); revision: `471fc60775f80492f4f224203d172868bc6eebac`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PM25_GNN.toml`](../../../../configs/models/PM25_GNN.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper description; reference-only source
code was not copied. Graph messages enter both gates of a history encoder and an
autoregressive future-covariate decoder. Repository adjacency cannot supply the
paper's geographic distance/direction and wind-conditioned transport features,
and shared calendar marks replace the KnowAir meteorological variables. The
paper's data pipeline and published metric reference comparison are not claimed.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `hid_dim=64`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting
- **Venue**: ACM SIGSPATIAL 2020
- **Published**: 2020 (arXiv: 2020-02)
- **arXiv**: https://arxiv.org/abs/2002.12898

## Abstract
When predicting PM2.5 concentrations, it is necessary to consider complex information sources since the concentrations are influenced by various factors within a long period. In this paper, we identify a set of critical domain knowledge for PM2.5 forecasting and develop a novel graph based model, PM2.5-GNN, being capable of capturing long-term dependencies. On a real-world dataset, we validate the effectiveness of the proposed model and examine its abilities of capturing both fine-grained and long-term influences in PM2.5 process. The proposed PM2.5-GNN has also been deployed online to provide free forecasting service.

## In TSFLab
Default config: `configs/models/PM25_GNN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper description; reference-only source
code was not copied. Graph messages enter both gates of a history encoder and an
autoregressive future-covariate decoder. Repository adjacency cannot supply the
paper's geographic distance/direction and wind-conditioned transport features,
and shared calendar marks replace the KnowAir meteorological variables. The
paper's data pipeline and published metric reference comparison are not claimed.

## Citation

```bibtex
@inproceedings{DBLP:conf/gis/WangLZMMG20,
  author       = {Shuo Wang and
                  Yanran Li and
                  Jiang Zhang and
                  Qingye Meng and
                  Lingwei Meng and
                  Fei Gao},
  editor       = {Chang{-}Tien Lu and
                  Fusheng Wang and
                  Goce Trajcevski and
                  Yan Huang and
                  Shawn D. Newsam and
                  Li Xiong},
  title        = {{PM2.5-GNN:} {A} Domain Knowledge Enhanced Graph Neural Network For
                  {PM2.5} Forecasting},
  booktitle    = {{SIGSPATIAL} '20: 28th International Conference on Advances in Geographic
                  Information Systems, Seattle, WA, USA, November 3-6, 2020},
  pages        = {163--166},
  publisher    = {{ACM}},
  year         = {2020},
  url          = {https://doi.org/10.1145/3397536.3422208},
  doi          = {10.1145/3397536.3422208},
  timestamp    = {Sat, 30 Sep 2023 09:41:50 +0200},
  biburl       = {https://dblp.org/rec/conf/gis/WangLZMMG20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
