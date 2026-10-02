---
name: "MGSFformer"
summary: "MGSFformer is a Multi-Granularity Spatiotemporal Fusion Transformer designed for node-level air quality prediction. It consists of three specialised sub-modules: a residual de-redundant block that eliminates information redundancy between data of different temporal granularities, a spatiotemporal attention block that captures correlations across monitoring stations and time, and a dynamic fusion block that adaptively weights and integrates multi-granularity predictions."
paper: "https://doi.org/10.1016/j.inffus.2024.102607"
paper_title: "MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction"
venue: "Information Fusion 2025"
year: 2025
code: "https://github.com/GestaltCogTeam/MGSFformer"
revision: "ff665a422a0ae001cfdd1b60ec9b4338a5ab406e"
license: "NOASSERTION"
---
# MGSFformer

MGSFformer is a Multi-Granularity Spatiotemporal Fusion Transformer designed for node-level air quality prediction. It consists of three specialised sub-modules: a residual de-redundant block that eliminates information redundancy between data of different temporal granularities, a spatiotemporal attention block that captures correlations across monitoring stations and time, and a dynamic fusion block that adaptively weights and integrates multi-granularity predictions.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, nodes]`. The
declared output contract is a `[batch, 24, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1016/j.inffus.2024.102607); title: MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction; venue/year: Information Fusion 2025 / 2025
- [codebase](https://github.com/GestaltCogTeam/MGSFformer); revision: `ff665a422a0ae001cfdd1b60ec9b4338a5ab406e`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MGSFformer.toml`](../../../../configs/models/MGSFformer.toml).

## Differences

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map retains five granularities, residual de-redundancy,
temporal/spatial attention, and dynamic fusion; private preprocessing and
auxiliary objectives are omitted.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `IE_dim=32`, `dropout=0.3`, `num_head=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction
- **Venue**: Information Fusion 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Air quality prediction is a critical task in environmental science. Air monitoring stations typically collect data at multiple sampling intervals (multiple granularities), each exhibiting distinct temporal patterns, and data from different stations exhibit strong spatiotemporal correlations. MGSFformer addresses both challenges simultaneously through three components: (1) a residual de-redundant block that removes redundant information across granularities, preventing the model from being misled by overlapping signals; (2) a spatiotemporal attention block that models correlations among stations and across time steps; and (3) a dynamic fusion block that assesses the relative importance of each granularity and integrates the resulting predictions. Experiments on three real-world air quality datasets demonstrate that MGSFformer outperforms 11 state-of-the-art baselines by approximately 5%.

## In ModernTSF
Default config: `configs/models/MGSFformer.toml`; model specification: `spec.py`; implementation: `model.py`.

The independent implementation keeps the paper's three named modules and
RevIN. It consumes historical targets only and requires `seq_len` to be a
multiple of 24. The unlicensed author repository is reference-only and its
source was not copied.

## Verification

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map retains five granularities, residual de-redundancy,
temporal/spatial attention, and dynamic fusion; private preprocessing and
auxiliary objectives are omitted.

## Citation

```bibtex
@article{DBLP:journals/inffus/YuWWSSYX25,
  author       = {Chengqing Yu and
                  Fei Wang and
                  Yilun Wang and
                  Zezhi Shao and
                  Tao Sun and
                  Di Yao and
                  Yongjun Xu},
  title        = {MGSFformer: {A} Multi-Granularity Spatiotemporal Fusion Transformer
                  for air quality prediction},
  journal      = {Inf. Fusion},
  volume       = {113},
  pages        = {102607},
  year         = {2025},
  url          = {https://doi.org/10.1016/j.inffus.2024.102607},
  doi          = {10.1016/J.INFFUS.2024.102607},
  timestamp    = {Sat, 31 May 2025 23:16:07 +0200},
  biburl       = {https://dblp.org/rec/journals/inffus/YuWWSSYX25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
