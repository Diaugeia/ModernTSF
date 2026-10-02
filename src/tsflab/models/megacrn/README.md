---
name: "MegaCRN"
summary: "MegaCRN (Meta-Graph Convolutional Recurrent Network) is a spatiotemporal forecasting model designed for graph-structured node data such as road-network traffic. It addresses the heterogeneity and non-stationarity inherent in traffic streams by learning dynamic graph structures through a Meta-Graph Learner backed by a learnable Meta-Node Bank, plugged into a GCRN encoder-decoder. This allows the model to disentangle locations and time slots with different patterns and adapt robustly to anomalous conditions."
paper: "https://arxiv.org/abs/2211.14701"
paper_title: "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting"
venue: "AAAI 2023"
year: 2023
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# MegaCRN

MegaCRN (Meta-Graph Convolutional Recurrent Network) is a spatiotemporal forecasting model designed for graph-structured node data such as road-network traffic. It addresses the heterogeneity and non-stationarity inherent in traffic streams by learning dynamic graph structures through a Meta-Graph Learner backed by a learnable Meta-Node Bank, plugged into a GCRN encoder-decoder. This allows the model to disentangle locations and time slots with different patterns and adapt robustly to anomalous conditions.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2211.14701); title: Spatio-Temporal Meta-Graph Learning for Traffic Forecasting; venue/year: AAAI 2023 / 2023
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MegaCRN.toml`](../../../../configs/models/MegaCRN.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper structure; reference-only source
code was not copied. Hidden-state queries attend a trainable meta-node bank,
memory-derived node embeddings form a dynamic meta-graph, and graph-polynomial
GRU cells drive an autoregressive encoder-decoder. The supplied adjacency is a
soft prior. Contrastive memory losses, curriculum teacher forcing, the official
data pipeline and metric reference comparison are outside this module.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `rnn_units=32`, `num_layers=1`, `cheb_k=3`, `mem_num=8`, `mem_dim=16`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatio-Temporal Meta-Graph Learning for Traffic Forecasting
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.14701

## Abstract
Traffic forecasting as a canonical task of multivariate time series forecasting has been a significant research topic in AI community. To address the spatio-temporal heterogeneity and non-stationarity implied in the traffic stream, in this study, we propose Spatio-Temporal Meta-Graph Learning as a novel Graph Structure Learning mechanism on spatio-temporal data. Specifically, we implement this idea into Meta-Graph Convolutional Recurrent Network (MegaCRN) by plugging the Meta-Graph Learner powered by a Meta-Node Bank into GCRN encoder-decoder. We conduct a comprehensive evaluation on two benchmark datasets (i.e., METR-LA and PEMS-BAY) and a new large-scale traffic speed dataset called EXPY-TKY that covers 1843 expressway road links in Tokyo. Our model outperformed the state-of-the-arts on all three datasets. Besides, through a series of qualitative evaluations, we demonstrate that our model can explicitly disentangle the road links and time slots with different patterns and be robustly adaptive to any anomalous traffic situations.

## In TSFLab
Default config: `configs/models/MegaCRN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper structure; reference-only source
code was not copied. Hidden-state queries attend a trainable meta-node bank,
memory-derived node embeddings form a dynamic meta-graph, and graph-polynomial
GRU cells drive an autoregressive encoder-decoder. The supplied adjacency is a
soft prior. Contrastive memory losses, curriculum teacher forcing, the official
data pipeline and metric reference comparison are outside this module.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/Jiang0YJCK0FS23,
  author       = {Renhe Jiang and
                  Zhaonan Wang and
                  Jiawei Yong and
                  Puneet Jeph and
                  Quanjun Chen and
                  Yasumasa Kobayashi and
                  Xuan Song and
                  Shintaro Fukushima and
                  Toyotaro Suzumura},
  editor       = {Brian Williams and
                  Yiling Chen and
                  Jennifer Neville},
  title        = {Spatio-Temporal Meta-Graph Learning for Traffic Forecasting},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {8078--8086},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i7.25976},
  doi          = {10.1609/AAAI.V37I7.25976},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/Jiang0YJCK0FS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
