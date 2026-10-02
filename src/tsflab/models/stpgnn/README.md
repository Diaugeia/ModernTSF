---
name: "STPGNN"
summary: "STPGNN (Spatio-Temporal Pivotal Graph Neural Network) is a spatiotemporal learning model for node-structured traffic forecasting that explicitly identifies and models pivotal nodes — nodes with a large number of connections to other nodes — which are disproportionately difficult to predict with standard graph neural networks. It consists of a Pivotal Node Identification Module, a Pivotal Graph Convolution Module for capturing complex spatio-temporal dependencies around these high-connectivity nodes, and a parallel architecture that simultaneously processes both pivotal and non-pivotal nodes."
paper: "https://doi.org/10.1609/aaai.v38i8.28707"
paper_title: "Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting"
venue: "AAAI 2024"
year: 2024
code: "https://github.com/Kongwy5689/STPGNN"
revision: "df199624259776515a2d287c32b5db459a629f5d"
license: "MIT"
tagline: "Parallel graph layers where a learned pivotal-node membership gates a temporally windowed graph convolution."
tags: ["gnn", "spatiotemporal", "graph-learning", "covariates", "cnn"]
composition: ["normalization=none", "decomposition=none", "temporal=local:temporal-span-window-and-conv", "channel=local:pivotal-node-graph-convolution+local:adaptive-graph", "head=local:flatten-mlp-readout", "loss=loss:mse"]
---
# STPGNN

## Key ideas

- `PivotalNodeIdentification` scores nodes by physical plus learned-affinity degree, exposes the top-k pivotal set and a smooth sigmoid membership, and returns a learned adaptive graph added to the physical one.
- `PivotalGraphConvolution` averages a sliding temporal window with learned weights, propagates over the graph, and is multiplied by pivotal membership.
- `ParallelSTLayer` runs the pivotal path, an ordinary graph path and a temporal conv in parallel, fuses them linearly with a residual LayerNorm; a flatten MLP reads out the horizon.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1609/aaai.v38i8.28707); title: Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting; venue/year: AAAI 2024 / 2024
- [codebase](https://github.com/Kongwy5689/STPGNN); revision: `df199624259776515a2d287c32b5db459a629f5d`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STPGNN.toml`](../../../../configs/models/STPGNN.toml).

## Differences

Clean-room implementation: confirmed. The implementation was derived independently from the paper's pivotal-node identification, Equation 7 pivotal graph convolution, Equation 8 ordinary diffusion, and parallel temporal branch; reference source code was not copied or reused. It uses smooth pivotal membership for trainability and retains an inspectable top-k pivotal set.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `dropout=0.1`, `topk=4`, `residual_channels=16`, `end_channels=64`, `kernel_size=2`, `blocks=2`, `layers=2`, `dims=16`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Traffic flow forecasting is a classical spatio-temporal data mining problem with many real-world applications. Graph Neural Networks (GNNs) are currently the mainstream approach to solving this problem. However, the majority of existing methods disregard the importance of certain nodes (referred to as pivotal nodes) that naturally exhibit extensive connections with multiple other nodes. Predicting on pivotal nodes poses a challenge due to their complex spatio-temporal dependencies compared to other nodes. In this paper, we propose Spatio-Temporal Pivotal Graph Neural Networks (STPGNN) to address this challenge. Specifically, we first introduce a pivotal node identification module for identifying pivotal nodes. We then propose a novel pivotal graph convolution module, enabling precise capture of spatio-temporal dependencies centered around pivotal nodes. We further propose a parallel framework capable of extracting spatio-temporal traffic features on both pivotal and non-pivotal nodes. Experiments on seven real-world traffic datasets verify the effectiveness and efficiency of our proposed method compared to state-of-the-art baselines.

## In TSFLab
Default config: `configs/models/STPGNN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. The implementation was derived independently from the paper's pivotal-node identification, Equation 7 pivotal graph convolution, Equation 8 ordinary diffusion, and parallel temporal branch; reference source code was not copied or reused. It uses smooth pivotal membership for trainability and retains an inspectable top-k pivotal set.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/KongGL24,
  author       = {Weiyang Kong and
                  Ziyu Guo and
                  Yubao Liu},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {8627--8635},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i8.28707},
  doi          = {10.1609/AAAI.V38I8.28707},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/KongGL24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
