---
name: "STGCN"
summary: "STGCN (Spatio-Temporal Graph Convolutional Network) is a deep learning framework for node-level spatiotemporal forecasting, originally developed for traffic speed prediction. It combines graph convolution layers that capture spatial dependencies between nodes on a road network with temporal convolution layers that model short- and long-range time patterns, using fully convolutional structures to achieve fast training and compact parameterisation compared to recurrent alternatives."
paper: "https://arxiv.org/abs/1709.04875"
paper_title: "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"
venue: "IJCAI 2018"
year: 2018
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# STGCN

STGCN (Spatio-Temporal Graph Convolutional Network) is a deep learning framework for node-level spatiotemporal forecasting, originally developed for traffic speed prediction. It combines graph convolution layers that capture spatial dependencies between nodes on a road network with temporal convolution layers that model short- and long-range time patterns, using fully convolutional structures to achieve fast training and compact parameterisation compared to recurrent alternatives.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/1709.04875); title: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting; venue/year: IJCAI 2018 / 2018
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STGCN.toml`](../../../../configs/models/STGCN.toml).

## Differences

ModernTSF rewrites STGCN locally after reviewing the paper and pinned official codebase. Each block follows temporal GLU, fixed Chebyshev graph convolution, and temporal GLU order using the injected adjacency and shared spectral support builder. Canonical evidence is stored in [`verification/evidence/STGCN.json`](../../../../verification/evidence/STGCN.json).

## Shared components

- [`graph_spectral`](../_components/graph_spectral/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `Kt=3`, `Ks=3`, `hidden_dim=32`, `bottleneck_dim=8`, `out_hidden_dim=32`, `act_func='glu'`, `graph_conv_type='cheb_graph_conv'`, `bias=True`, `droprate=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
- **Venue**: IJCAI 2018
- **Published**: 2018 (arXiv: 2017-09)
- **arXiv**: https://arxiv.org/abs/1709.04875

## Abstract
Timely accurate traffic forecast is crucial for urban traffic control and guidance. Due to the high nonlinearity and complexity of traffic flow, traditional methods cannot satisfy the requirements of mid-and-long term prediction tasks and often neglect spatial and temporal dependencies. In this paper, we propose a novel deep learning framework, Spatio-Temporal Graph Convolutional Networks (STGCN), to tackle the time series prediction problem in traffic domain. Instead of applying regular convolutional and recurrent units, we formulate the problem on graphs and build the model with complete convolutional structures, which enable much faster training speed with fewer parameters. Experiments show that our model STGCN effectively captures comprehensive spatio-temporal correlations through modeling multi-scale traffic networks and consistently outperforms state-of-the-art baselines on various real-world traffic datasets.

## In ModernTSF
Default config: `configs/models/STGCN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

ModernTSF rewrites STGCN locally after reviewing the paper and pinned official codebase. Each block follows temporal GLU, fixed Chebyshev graph convolution, and temporal GLU order using the injected adjacency and shared spectral support builder. Canonical evidence is stored in [`verification/evidence/STGCN.json`](../../../../verification/evidence/STGCN.json).

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcai/YuYZ18,
  author       = {Bing Yu and
                  Haoteng Yin and
                  Zhanxing Zhu},
  editor       = {J{\'{e}}r{\^{o}}me Lang},
  title        = {Spatio-Temporal Graph Convolutional Networks: {A} Deep Learning Framework
                  for Traffic Forecasting},
  booktitle    = {Proceedings of the Twenty-Seventh International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2018, July 13-19, 2018, Stockholm,
                  Sweden},
  pages        = {3634--3640},
  publisher    = {ijcai.org},
  year         = {2018},
  url          = {https://doi.org/10.24963/ijcai.2018/505},
  doi          = {10.24963/IJCAI.2018/505},
  timestamp    = {Sun, 04 Aug 2024 19:36:39 +0200},
  biburl       = {https://dblp.org/rec/conf/ijcai/YuYZ18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
