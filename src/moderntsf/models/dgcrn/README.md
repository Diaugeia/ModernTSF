---
name: "DGCRN"
summary: "The DGCRN paper uses hyper-networks to generate time-varying graph filters and combines the resulting dynamic adjacency with a predefined graph inside a recurrent encoder-decoder. This clean-room implementation generates directed graphs from hidden state and node embeddings at every step, mixes dynamic and static propagation in graph-GRU gates, and uses known time marks without future targets."
paper: "https://doi.org/10.1145/3532611"
paper_title: "Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution"
venue: "ACM TKDD 2023"
year: 2023
code: "https://github.com/tsinghua-fib-lab/Traffic-Benchmark"
revision: "b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228"
license: "MIT"
---
# DGCRN

The DGCRN paper uses hyper-networks to generate time-varying graph filters and combines the resulting dynamic adjacency with a predefined graph inside a recurrent encoder-decoder. This clean-room implementation generates directed graphs from hidden state and node embeddings at every step, mixes dynamic and static propagation in graph-GRU gates, and uses known time marks without future targets.

<!-- model-card:canonical:start -->
## Method overview

The DGCRN paper uses hyper-networks to generate time-varying graph filters and combines the resulting dynamic adjacency with a predefined graph inside a recurrent encoder-decoder.

## Core architecture

This clean-room implementation generates directed graphs from hidden state and node embeddings at every step, mixes dynamic and static propagation in graph-GRU gates, and uses known time marks without future targets.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1145/3532611); title: Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution; venue/year: ACM TKDD 2023 / 2023
- [codebase](https://github.com/tsinghua-fib-lab/Traffic-Benchmark); revision: `b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/DGCRN.toml`](../../../../configs/models/DGCRN.toml).

## Differences

- Clean-room implementation: confirmed. The module was designed from the DGCRN method description and equations. The official source is reference-only; the removed BasicTS-derived file was not a basis for this replacement.
- Formula mapping: `DynamicGraphGenerator` is the hidden-state-conditioned hyper-network; `DynamicGraphConvolution` concatenates static forward/reverse and learned directed multi-hop propagation; `DynamicGraphGRUCell` inserts the filters into recurrent gates.
- Adjacency and marks: `adj_mx` is shape-checked and row-normalized in both directions. Historical and future raw or node-structured marks contribute one known time driver; future target values are never consumed.
- Differences and limits: the default dimensions are reduced, one recurrent cell is used for encoder and decoder, and task-level curriculum, target teacher forcing, official preprocessing, and published-metric reference comparison are not reproduced. Missing adjacency uses identity transitions.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `gcn_depth=1`, `rnn_size=16`, `node_dim=8`, `hyper_gnn_dim=8`, `middle_dim=2`, `tanhalpha=3.0`, `dropout=0.3`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution
- **Venue**: ACM Transactions on Knowledge Discovery from Data (TKDD), Vol. 17, No. 1, Article 9
- **Published**: 2023 (arXiv: 2021-04)
- **arXiv**: https://arxiv.org/abs/2104.14917

## Abstract
Traffic prediction is the cornerstone of an intelligent transportation system. Accurate traffic forecasting is essential for the applications of smart cities, i.e., intelligent traffic management and urban planning. Although various methods are proposed for spatio-temporal modeling, they ignore the dynamic characteristics of correlations among locations on road networks. Meanwhile, most Recurrent Neural Network (RNN) based works are not efficient enough due to their recurrent operations. Additionally, there is a severe lack of fair comparison among different methods on the same datasets. To address the above challenges, in this paper, we propose a novel traffic prediction framework, named Dynamic Graph Convolutional Recurrent Network (DGCRN). In DGCRN, hyper-networks are designed to leverage and extract dynamic characteristics from node attributes, while the parameters of dynamic filters are generated at each time step. We filter the node embeddings and then use them to generate a dynamic graph, which is integrated with a pre-defined static graph. As far as we know, we are the first to employ a generation method to model fine topology of dynamic graph at each time step. Further, to enhance efficiency and performance, we employ a training strategy for DGCRN by restricting the iteration number of decoder during forward and backward propagation. Finally, a reproducible standardized benchmark and a brand new representative traffic dataset are opened for fair comparison and further research. Extensive experiments on three datasets demonstrate that our model outperforms 15 baselines consistently.

## In ModernTSF
Default config: `configs/models/DGCRN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

- Clean-room implementation: confirmed. The module was designed from the DGCRN method description and equations. The official source is reference-only; the removed BasicTS-derived file was not a basis for this replacement.
- Formula mapping: `DynamicGraphGenerator` is the hidden-state-conditioned hyper-network; `DynamicGraphConvolution` concatenates static forward/reverse and learned directed multi-hop propagation; `DynamicGraphGRUCell` inserts the filters into recurrent gates.
- Adjacency and marks: `adj_mx` is shape-checked and row-normalized in both directions. Historical and future raw or node-structured marks contribute one known time driver; future target values are never consumed.
- Differences and limits: the default dimensions are reduced, one recurrent cell is used for encoder and decoder, and task-level curriculum, target teacher forcing, official preprocessing, and published-metric reference comparison are not reproduced. Missing adjacency uses identity transitions.

## Citation

```bibtex
@article{DBLP:journals/tkdd/LiFYJYSJL23,
  author       = {Fuxian Li and
                  Jie Feng and
                  Huan Yan and
                  Guangyin Jin and
                  Fan Yang and
                  Funing Sun and
                  Depeng Jin and
                  Yong Li},
  title        = {Dynamic Graph Convolutional Recurrent Network for Traffic Prediction:
                  Benchmark and Solution},
  journal      = {{ACM} Trans. Knowl. Discov. Data},
  volume       = {17},
  number       = {1},
  pages        = {9:1--9:21},
  year         = {2023},
  url          = {https://doi.org/10.1145/3532611},
  doi          = {10.1145/3532611},
  timestamp    = {Fri, 27 Feb 2026 23:29:38 +0100},
  biburl       = {https://dblp.org/rec/journals/tkdd/LiFYJYSJL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
