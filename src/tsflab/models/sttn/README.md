---
name: "STTN"
summary: "STTN (Spatial-Temporal Transformer Networks) is a spatiotemporal forecasting model designed for node-structured traffic and sensor-network data. It combines a spatial Transformer that dynamically models directed spatial dependencies with a self-attention mechanism — capturing real-time node-to-node relationships without a fixed adjacency matrix — with a temporal Transformer that captures long-range bidirectional temporal dependencies, yielding competitive accuracy especially for long-horizon traffic flow forecasting."
paper: "https://arxiv.org/abs/2001.02908"
paper_title: "Spatial-Temporal Transformer Networks for Traffic Flow Forecasting"
venue: "arXiv preprint"
year: 2020
code: "https://github.com/xumingxingsjtu/STTN"
revision: "d24f8d331a6d81b819cfe0a9430793ae028d25ad"
license: "NOASSERTION"
---
# STTN

STTN (Spatial-Temporal Transformer Networks) is a spatiotemporal forecasting model designed for node-structured traffic and sensor-network data. It combines a spatial Transformer that dynamically models directed spatial dependencies with a self-attention mechanism — capturing real-time node-to-node relationships without a fixed adjacency matrix — with a temporal Transformer that captures long-range bidirectional temporal dependencies, yielding competitive accuracy especially for long-horizon traffic flow forecasting.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2001.02908); title: Spatial-Temporal Transformer Networks for Traffic Flow Forecasting; venue/year: arXiv preprint / 2020
- [codebase](https://github.com/xumingxingsjtu/STTN); revision: `d24f8d331a6d81b819cfe0a9430793ae028d25ad`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STTN.toml`](../../../../configs/models/STTN.toml).

## Differences

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map covers directed multi-head spatial attention, its
gated stationary graph path, bidirectional temporal attention, and stacked ST
blocks. The official data pipeline and metric reference comparison are not claimed.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `d_model=64`, `mlp_expand=4`, `num_layers=3`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatial-Temporal Transformer Networks for Traffic Flow Forecasting
- **Venue**: arXiv preprint
- **Published**: 2020 (arXiv: 2020-01)
- **arXiv**: https://arxiv.org/abs/2001.02908

## Abstract
Traffic forecasting has emerged as a core component of intelligent transportation systems. However, timely accurate traffic forecasting, especially long-term forecasting, still remains an open challenge due to the highly nonlinear and dynamic spatial-temporal dependencies of traffic flows. In this paper, we propose a novel paradigm of Spatial-Temporal Transformer Networks (STTNs) that leverages dynamical directed spatial dependencies and long-range temporal dependencies to improve the accuracy of long-term traffic forecasting. Specifically, we present a new variant of graph neural networks, named spatial transformer, by dynamically modeling directed spatial dependencies with self-attention mechanism to capture realtime traffic conditions as well as the directionality of traffic flows. Furthermore, different spatial dependency patterns can be jointly modeled with multi-heads attention mechanism to consider diverse relationships related to different factors (e.g. similarity, connectivity and covariance). On the other hand, the temporal transformer is utilized to model long-range bidirectional temporal dependencies across multiple time steps. Finally, they are composed as a block to jointly model the spatial-temporal dependencies for accurate traffic prediction. Compared to existing works, the proposed model enables fast and scalable training over a long range spatial-temporal dependencies. Experiment results demonstrate that the proposed model achieves competitive results compared with the state-of-the-arts, especially forecasting long-term traffic flows on real-world PeMS-Bay and PeMSD7(M) datasets.

## In TSFLab
Default config: `configs/models/STTN.toml`; model specification: `spec.py`; implementation: `model.py`.

This is an independent implementation based on equations and architecture in
the paper. Four attention heads represent diverse directed spatial relations;
the supplied graph contributes a separate stationary path. Both unlicensed
repositories remain reference-only and no source from either is included.

## Verification

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map covers directed multi-head spatial attention, its
gated stationary graph path, bidirectional temporal attention, and stacked ST
blocks. The official data pipeline and metric reference comparison are not claimed.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2001-02908,
  author       = {Mingxing Xu and
                  Wenrui Dai and
                  Chunmiao Liu and
                  Xing Gao and
                  Weiyao Lin and
                  Guo{-}Jun Qi and
                  Hongkai Xiong},
  title        = {Spatial-Temporal Transformer Networks for Traffic Flow Forecasting},
  journal      = {CoRR},
  volume       = {abs/2001.02908},
  year         = {2020},
  url          = {http://arxiv.org/abs/2001.02908},
  eprinttype   = {arXiv},
  eprint       = {2001.02908},
  timestamp    = {Tue, 14 Jan 2020 10:25:48 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2001-02908.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
