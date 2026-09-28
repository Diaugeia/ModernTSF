---
name: "STGODE"
summary: "STGODE is a spatiotemporal learning model for node-structured traffic and graph data that captures continuous spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE). By coupling a semantic adjacency matrix with a temporal dilated convolution structure, it overcomes the over-smoothing limitation of shallow GNNs and captures both structural and semantic long-range dependencies between nodes."
paper: "https://doi.org/10.1145/3447548.3467430"
paper_title: "Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting"
venue: "KDD 2021"
year: 2021
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# STGODE

STGODE is a spatiotemporal learning model for node-structured traffic and graph data that captures continuous spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE). By coupling a semantic adjacency matrix with a temporal dilated convolution structure, it overcomes the over-smoothing limitation of shallow GNNs and captures both structural and semantic long-range dependencies between nodes.

<!-- model-card:canonical:start -->
## Method overview

STGODE is a spatiotemporal learning model for node-structured traffic and graph data that captures continuous spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE).

## Core architecture

By coupling a semantic adjacency matrix with a temporal dilated convolution structure, it overcomes the over-smoothing limitation of shallow GNNs and captures both structural and semantic long-range dependencies between nodes.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1145/3447548.3467430); title: Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting; venue/year: KDD 2021 / 2021
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/STGODE.toml`](../../../../configs/models/STGODE.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper equations; reference-only source
code was not copied. Separate spatial and semantic branches apply causal dilated
temporal filters and explicit Euler steps to simultaneous graph, channel and
temporal dynamics before gated fusion. Because the catalog contract has no
training-series DTW matrix, semantic similarity is derived from graph-neighborhood
profiles. The integration depth is configurable; official preprocessing,
`torchdiffeq` solver details and published metric reference comparison are not reproduced.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `hidden_dim=32`, `ode_steps=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting
- **Venue**: KDD 2021
- **Published**: 2021 (arXiv: 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.12931

## Abstract
Spatial-temporal forecasting has attracted tremendous attention in a wide range of applications, and traffic flow prediction is a canonical and typical example. The complex and long-range spatial-temporal correlations of traffic flow bring it to a most intractable challenge. Existing works typically utilize shallow graph convolution networks (GNNs) and temporal extracting modules to model spatial and temporal dependencies respectively. However, the representation ability of such models is limited due to: (1) shallow GNNs are incapable to capture long-range spatial correlations, (2) only spatial connections are considered and a mass of semantic connections are ignored, which are of great importance for a comprehensive understanding of traffic networks. To this end, we propose Spatial-Temporal Graph Ordinary Differential Equation Networks (STGODE). Specifically, we capture spatial-temporal dynamics through a tensor-based ordinary differential equation (ODE), as a result, deeper networks can be constructed and spatial-temporal features are utilized synchronously. To understand the network more comprehensively, semantical adjacency matrix is considered in our model, and a well-design temporal dialated convolution structure is used to capture long term temporal dependencies. We evaluate our model on multiple real-world traffic datasets and superior performance is achieved over state-of-the-art baselines.

## In ModernTSF
Default config: `configs/models/STGODE.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper equations; reference-only source
code was not copied. Separate spatial and semantic branches apply causal dilated
temporal filters and explicit Euler steps to simultaneous graph, channel and
temporal dynamics before gated fusion. Because the catalog contract has no
training-series DTW matrix, semantic similarity is derived from graph-neighborhood
profiles. The integration depth is configurable; official preprocessing,
`torchdiffeq` solver details and published metric reference comparison are not reproduced.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/FangLSX21,
  author       = {Zheng Fang and
                  Qingqing Long and
                  Guojie Song and
                  Kunqing Xie},
  editor       = {Feida Zhu and
                  Beng Chin Ooi and
                  Chunyan Miao},
  title        = {Spatial-Temporal Graph {ODE} Networks for Traffic Flow Forecasting},
  booktitle    = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
  pages        = {364--373},
  publisher    = {{ACM}},
  year         = {2021},
  url          = {https://doi.org/10.1145/3447548.3467430},
  doi          = {10.1145/3447548.3467430},
  timestamp    = {Tue, 29 Nov 2022 09:04:02 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/FangLSX21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
