---
name: "STID"
summary: "STID (Spatial-Temporal IDentity) is an MLP-based spatiotemporal forecasting model designed for node-structured or graph-structured data. It attaches learnable spatial identity embeddings (one per node) and temporal identity embeddings (time-of-day and day-of-week) to the input, then encodes all features with simple multi-layer perceptrons to predict future node values, achieving strong performance with minimal complexity."
paper: "https://arxiv.org/abs/2208.05233"
paper_title: "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
venue: "CIKM 2022"
year: 2022
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# STID

STID (Spatial-Temporal IDentity) is an MLP-based spatiotemporal forecasting model designed for node-structured or graph-structured data. It attaches learnable spatial identity embeddings (one per node) and temporal identity embeddings (time-of-day and day-of-week) to the input, then encodes all features with simple multi-layer perceptrons to predict future node values, achieving strong performance with minimal complexity.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2208.05233); title: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting; venue/year: CIKM 2022 / 2022
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STID.toml`](../../../../configs/models/STID.toml).

## Differences

ModernTSF rewrites STID locally after reviewing the paper and pinned official codebase. Flattened node histories are combined with node, time-of-day, and day-of-week identities, processed by pointwise residual blocks, and projected directly to the forecast horizon. Canonical evidence is stored in [`verification/evidence/STID.json`](../../../../verification/evidence/STID.json).

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `embed_dim=32`, `num_layers=1`, `num_time_in_day=24`, `num_day_in_week=7`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
- **Venue**: CIKM 2022
- **Published**: 2022 (arXiv: 2022-08)
- **arXiv**: https://arxiv.org/abs/2208.05233

## Abstract
Multivariate Time Series (MTS) forecasting plays a vital role in a wide range of applications. Recently, Spatial-Temporal Graph Neural Networks (STGNNs) have become increasingly popular MTS forecasting methods due to their state-of-the-art performance. However, recent works are becoming more sophisticated with limited performance improvements. This phenomenon motivates us to explore the critical factors of MTS forecasting and design a model that is as powerful as STGNNs, but more concise and efficient. In this paper, we identify the indistinguishability of samples in both spatial and temporal dimensions as a key bottleneck, and propose a simple yet effective baseline for MTS forecasting by attaching Spatial and Temporal IDentity information (STID), which achieves the best performance and efficiency simultaneously based on simple Multi-Layer Perceptrons (MLPs). These results suggest that we can design efficient and effective models as long as they solve the indistinguishability of samples, without being limited to STGNNs.

## In ModernTSF
Default config: `configs/models/STID.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

ModernTSF rewrites STID locally after reviewing the paper and pinned official codebase. Flattened node histories are combined with node, time-of-day, and day-of-week identities, processed by pointwise residual blocks, and projected directly to the forecast horizon. Canonical evidence is stored in [`verification/evidence/STID.json`](../../../../verification/evidence/STID.json).

## Citation

```bibtex
@inproceedings{DBLP:conf/cikm/ShaoZ00X22,
  author       = {Zezhi Shao and
                  Zhao Zhang and
                  Fei Wang and
                  Wei Wei and
                  Yongjun Xu},
  editor       = {Mohammad Al Hasan and
                  Li Xiong},
  title        = {Spatial-Temporal Identity: {A} Simple yet Effective Baseline for Multivariate
                  Time Series Forecasting},
  booktitle    = {Proceedings of the 31st {ACM} International Conference on Information
                  {\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
  pages        = {4454--4458},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3511808.3557702},
  doi          = {10.1145/3511808.3557702},
  timestamp    = {Sun, 02 Nov 2025 21:27:39 +0100},
  biburl       = {https://dblp.org/rec/conf/cikm/ShaoZ00X22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
