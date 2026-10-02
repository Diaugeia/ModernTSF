---
name: "GAGNN"
summary: "GAGNN is a covariate prediction model for node-level air quality forecasting, corresponding to the original air quality prediction setting. It constructs both a city graph and a city group graph to capture spatial and latent dependencies between cities, using hierarchical group-aware attention and message-passing to predict future air quality indices at each node."
paper: "https://doi.org/10.1145/3631713"
paper_title: "Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting"
venue: "ACM TKDD 2024"
year: 2024
code: "https://github.com/Friger/GAGNN"
revision: "509ac7d6eb55914979fc45f6d23e967021cfd270"
license: "MIT"
---
# GAGNN

GAGNN is a covariate prediction model for node-level air quality forecasting, corresponding to the original air quality prediction setting. It constructs both a city graph and a city group graph to capture spatial and latent dependencies between cities, using hierarchical group-aware attention and message-passing to predict future air quality indices at each node.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1145/3631713); title: Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting; venue/year: ACM TKDD 2024 / 2024
- [codebase](https://github.com/Friger/GAGNN); revision: `509ac7d6eb55914979fc45f6d23e967021cfd270`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GAGNN.toml`](../../../../configs/models/GAGNN.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper description; the reference-only
source was not copied. A GRU encodes each city's history, soft assignments pool
cities into latent groups, learned group correlations propagate between groups,
and city-, group-, and residual features are fused. Location attributes and the
paper's full air-quality feature set are unavailable; the supplied adjacency and
calendar covariates are used instead, with a direct multi-horizon head. Published
metric reference comparison is not claimed.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `d_model=64`, `num_layers=3`, `dropout=0.1`, `group_num=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting
- **Venue**: ACM Transactions on Knowledge Discovery from Data (TKDD), Vol. 18, No. 3, Article 55
- **Published**: 2024 (arXiv: 2021-08)
- **arXiv**: https://arxiv.org/abs/2108.12238

## Abstract
The problem of air pollution threatens public health. Air quality forecasting can provide the air quality index hours or even days later, which can help the public to prevent air pollution in advance. Previous works focus on citywide air quality forecasting and cannot solve nationwide city forecasting problem, whose difficulties lie in capturing the latent dependencies between geographically distant but highly correlated cities. In this paper, we propose the group-aware graph neural network (GAGNN), a hierarchical model for nationwide city air quality forecasting. The model constructs a city graph and a city group graph to model the spatial and latent dependencies between cities, respectively. GAGNN introduces differentiable grouping network to discover the latent dependencies among cities and generate city groups. Based on the generated city groups, a group correlation encoding module is introduced to learn the correlations between them, which can effectively capture the dependencies between city groups. After the graph construction, GAGNN implements message passing mechanism to model the dependencies between cities and city groups. The evaluation experiments on Chinese city air quality dataset indicate that our GAGNN outperforms existing forecasting models.

## In TSFLab
Default config: `configs/models/GAGNN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper description; the reference-only
source was not copied. A GRU encodes each city's history, soft assignments pool
cities into latent groups, learned group correlations propagate between groups,
and city-, group-, and residual features are fused. Location attributes and the
paper's full air-quality feature set are unavailable; the supplied adjacency and
calendar covariates are used instead, with a direct multi-horizon head. Published
metric reference comparison is not claimed.

## Citation

```bibtex
@article{DBLP:journals/tkdd/ChenXWH24,
  author       = {Ling Chen and
                  Jiahui Xu and
                  Binqing Wu and
                  Jianlong Huang},
  title        = {Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting},
  journal      = {{ACM} Trans. Knowl. Discov. Data},
  volume       = {18},
  number       = {3},
  pages        = {55:1--55:20},
  year         = {2024},
  url          = {https://doi.org/10.1145/3631713},
  doi          = {10.1145/3631713},
  timestamp    = {Sun, 19 Jan 2025 14:58:36 +0100},
  biburl       = {https://dblp.org/rec/journals/tkdd/ChenXWH24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
