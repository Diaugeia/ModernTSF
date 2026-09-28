---
name: "HimNet"
summary: "HimNet (Heterogeneity-Informed Spatiotemporal Meta-Network) is a spatiotemporal learning model designed for node-structured or graph-structured data. It captures spatiotemporal heterogeneity by learning spatial and temporal embeddings as a clustering process, then derives location- and time-specific parameters from meta-parameter pools using a hierarchical meta-graph GRU encoder-decoder with an adaptively learned graph topology."
paper: "https://doi.org/10.1145/3637528.3671961"
paper_title: "Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting"
venue: "KDD 2024"
year: 2024
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# HimNet

HimNet (Heterogeneity-Informed Spatiotemporal Meta-Network) is a spatiotemporal learning model designed for node-structured or graph-structured data. It captures spatiotemporal heterogeneity by learning spatial and temporal embeddings as a clustering process, then derives location- and time-specific parameters from meta-parameter pools using a hierarchical meta-graph GRU encoder-decoder with an adaptively learned graph topology.

<!-- model-card:canonical:start -->
## Method overview

HimNet (Heterogeneity-Informed Spatiotemporal Meta-Network) is a spatiotemporal learning model designed for node-structured or graph-structured data.

## Core architecture

It captures spatiotemporal heterogeneity by learning spatial and temporal embeddings as a clustering process, then derives location- and time-specific parameters from meta-parameter pools using a hierarchical meta-graph GRU encoder-decoder with an adaptively learned graph topology.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1145/3637528.3671961); title: Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting; venue/year: KDD 2024 / 2024
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/HimNet.toml`](../../../../configs/models/HimNet.toml).

## Differences

ModernTSF rewrites HimNet locally after reviewing the paper and pinned official codebase. Encoder and autoregressive decoder graph-GRU cells generate node-specific filters from hierarchical node, calendar, and horizon meta embeddings. Canonical evidence is stored in [`verification/evidence/HimNet.json`](../../../../verification/evidence/HimNet.json).

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `output_dim=1`, `hidden_dim=16`, `num_layers=1`, `cheb_k=2`, `node_embedding_dim=8`, `st_embedding_dim=8`, `tod_embedding_dim=8`, `dow_embedding_dim=8`, `steps_per_day=24`, `use_teacher_forcing=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting
- **Venue**: KDD 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.10800

## Abstract
Spatiotemporal time series forecasting plays a key role in a wide range of real-world applications. While significant progress has been made in this area, fully capturing and leveraging spatiotemporal heterogeneity remains a fundamental challenge. Therefore, we propose a novel Heterogeneity-Informed Meta-Parameter Learning scheme. Specifically, our approach implicitly captures spatiotemporal heterogeneity through learning spatial and temporal embeddings, which can be viewed as a clustering process. Then, a novel spatiotemporal meta-parameter learning paradigm is proposed to learn spatiotemporal-specific parameters from meta-parameter pools, which is informed by the captured heterogeneity. Based on these ideas, we develop a Heterogeneity-Informed Spatiotemporal Meta-Network (HimNet) for spatiotemporal time series forecasting. Extensive experiments on five widely-used benchmarks demonstrate our method achieves state-of-the-art performance while exhibiting superior interpretability.

## In ModernTSF
Default config: `configs/models/HimNet.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

ModernTSF rewrites HimNet locally after reviewing the paper and pinned official codebase. Encoder and autoregressive decoder graph-GRU cells generate node-specific filters from hierarchical node, calendar, and horizon meta embeddings. Canonical evidence is stored in [`verification/evidence/HimNet.json`](../../../../verification/evidence/HimNet.json).

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/DongJGLDW024,
  author       = {Zheng Dong and
                  Renhe Jiang and
                  Haotian Gao and
                  Hangchen Liu and
                  Jinliang Deng and
                  Qingsong Wen and
                  Xuan Song},
  editor       = {Ricardo Baeza{-}Yates and
                  Francesco Bonchi},
  title        = {Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal
                  Time Series Forecasting},
  booktitle    = {Proceedings of the 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2024, Barcelona, Spain, August 25-29, 2024},
  pages        = {631--641},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3637528.3671961},
  doi          = {10.1145/3637528.3671961},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/DongJGLDW024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
