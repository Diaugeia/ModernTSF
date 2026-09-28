---
name: "GWNet"
summary: "GWNet (Graph WaveNet) is a spatiotemporal graph neural network that serves the spatiotemporal forecasting setting on node-structured data. It jointly models hidden spatial dependencies via a learned adaptive adjacency matrix and long-range temporal trends via stacked dilated 1D causal convolutions whose receptive field grows exponentially with depth — enabling end-to-end, scalable traffic and sensor-network forecasting."
paper: "https://www.ijcai.org/proceedings/2019/264"
paper_title: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
venue: "IJCAI 2019"
year: 2019
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# GWNet

GWNet (Graph WaveNet) is a spatiotemporal graph neural network that serves the spatiotemporal forecasting setting on node-structured data. It jointly models hidden spatial dependencies via a learned adaptive adjacency matrix and long-range temporal trends via stacked dilated 1D causal convolutions whose receptive field grows exponentially with depth — enabling end-to-end, scalable traffic and sensor-network forecasting.

<!-- model-card:canonical:start -->
## Method overview

GWNet (Graph WaveNet) is a spatiotemporal graph neural network that serves the spatiotemporal forecasting setting on node-structured data.

## Core architecture

It jointly models hidden spatial dependencies via a learned adaptive adjacency matrix and long-range temporal trends via stacked dilated 1D causal convolutions whose receptive field grows exponentially with depth — enabling end-to-end, scalable traffic and sensor-network forecasting.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://www.ijcai.org/proceedings/2019/264); title: Graph WaveNet for Deep Spatial-Temporal Graph Modeling; venue/year: IJCAI 2019 / 2019
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/GWNet.toml`](../../../../configs/models/GWNet.toml).

## Differences

ModernTSF rewrites Graph WaveNet locally after reviewing the paper and pinned official codebase. It retains causal dilated gated convolutions, forward/reverse random-walk supports, a learned adaptive graph, diffusion convolution, residual paths, and accumulated skip forecasts. Canonical evidence is stored in [`verification/evidence/GWNet.json`](../../../../verification/evidence/GWNet.json).

## Shared components

- [`diffusion_conv`](../_components/diffusion_conv/README.md)
- [`graph_utils`](../_components/graph_utils/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `dropout=0.3`, `residual_channels=16`, `dilation_channels=16`, `skip_channels=64`, `end_channels=128`, `kernel_size=2`, `blocks=2`, `layers=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
- **Venue**: IJCAI 2019
- **Published**: 2019 (arXiv: 2019-05)
- **arXiv**: https://arxiv.org/abs/1906.00121

## Abstract
Spatial-temporal graph modeling is an important task to analyze the spatial relations and temporal trends of components in a system. Existing approaches mostly capture the spatial dependency on a fixed graph structure, assuming that the underlying relation between entities is pre-determined. However, the explicit graph structure (relation) does not necessarily reflect the true dependency and genuine relation may be missing due to the incomplete connections in the data. Furthermore, existing methods are ineffective to capture the temporal trends as the RNNs or CNNs employed in these methods cannot capture long-range temporal sequences. To overcome these limitations, we propose in this paper a novel graph neural network architecture, Graph WaveNet, for spatial-temporal graph modeling. By developing a novel adaptive dependency matrix and learn it through node embedding, our model can precisely capture the hidden spatial dependency in the data. With a stacked dilated 1D convolution component whose receptive field grows exponentially as the number of layers increases, Graph WaveNet is able to handle very long sequences. These two components are integrated seamlessly in a unified framework and the whole framework is learned in an end-to-end manner. Experimental results on two public traffic network datasets, METR-LA and PEMS-BAY, demonstrate the superior performance of our algorithm.

## In ModernTSF
Default config: `configs/models/GWNet.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

ModernTSF rewrites Graph WaveNet locally after reviewing the paper and pinned official codebase. It retains causal dilated gated convolutions, forward/reverse random-walk supports, a learned adaptive graph, diffusion convolution, residual paths, and accumulated skip forecasts. Canonical evidence is stored in [`verification/evidence/GWNet.json`](../../../../verification/evidence/GWNet.json).

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcai/WuPLJZ19,
  author       = {Zonghan Wu and
                  Shirui Pan and
                  Guodong Long and
                  Jing Jiang and
                  Chengqi Zhang},
  editor       = {Sarit Kraus},
  title        = {Graph WaveNet for Deep Spatial-Temporal Graph Modeling},
  booktitle    = {Proceedings of the Twenty-Eighth International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2019, Macao, China, August 10-16,
                  2019},
  pages        = {1907--1913},
  publisher    = {ijcai.org},
  year         = {2019},
  url          = {https://doi.org/10.24963/ijcai.2019/264},
  doi          = {10.24963/IJCAI.2019/264},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/ijcai/WuPLJZ19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
