---
name: "STDN"
summary: "STDN is a spatiotemporal learning model for node-structured graph data. It constructs a dynamic graph to represent traffic flow and captures global dynamics through novel spatio-temporal embeddings, then applies a trend-seasonality decomposition module to disentangle trend-cyclical and seasonal components for each node, before passing them through an encoder-decoder network."
paper: "https://doi.org/10.1609/aaai.v39i11.33247"
paper_title: "Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting"
venue: "AAAI 2025"
year: 2025
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# STDN

STDN is a spatiotemporal learning model for node-structured graph data. It constructs a dynamic graph to represent traffic flow and captures global dynamics through novel spatio-temporal embeddings, then applies a trend-seasonality decomposition module to disentangle trend-cyclical and seasonal components for each node, before passing them through an encoder-decoder network.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1609/aaai.v39i11.33247); title: Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting; venue/year: AAAI 2025 / 2025
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STDN.toml`](../../../../configs/models/STDN.toml).

## Differences

TSFLab rewrites STDN locally after reviewing the paper and pinned official codebase. Spatial Laplacian positions and calendar embeddings gate trend/seasonal decomposition; dynamic diffusion forecasts the trend while history-to-future attention forecasts the seasonal branch. Canonical evidence is stored in [`verification/evidence/STDN.json`](../../../../verification/evidence/STDN.json).

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `time_slice_size=60`, `K=4`, `d=8`, `L=1`, `order=2`, `reference=4`, `out_channels=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2025-02)
- **arXiv**: https://arxiv.org/abs/2502.12213

## Abstract
Traffic prediction is critical for optimizing travel scheduling and enhancing public safety, yet the complex spatial and temporal dynamics within traffic data present significant challenges for accurate forecasting. In this paper, we introduce a novel model, the Spatiotemporal-aware Trend-Seasonality Decomposition Network (STDN). This model begins by constructing a dynamic graph structure to represent traffic flow and incorporates novel spatio-temporal embeddings to jointly capture global traffic dynamics. The representations learned are further refined by a specially designed trend-seasonality decomposition module, which disentangles the trend-cyclical component and seasonal component for each traffic node at different times within the graph. These components are subsequently processed through an encoder-decoder network to generate the final predictions. Extensive experiments conducted on real-world traffic datasets demonstrate that STDN achieves superior performance with remarkable computation cost. Furthermore, we have released a new traffic dataset named JiNan, which features unique inner-city dynamics, thereby enriching the scenario comprehensiveness in traffic prediction evaluation.

## In TSFLab
Default config: `configs/models/STDN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

TSFLab rewrites STDN locally after reviewing the paper and pinned official codebase. Spatial Laplacian positions and calendar embeddings gate trend/seasonal decomposition; dynamic diffusion forecasts the trend while history-to-future attention forecasts the seasonal branch. Canonical evidence is stored in [`verification/evidence/STDN.json`](../../../../verification/evidence/STDN.json).

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/CaoWJYD25,
  author       = {Lingxiao Cao and
                  Bin Wang and
                  Guiyuan Jiang and
                  Yanwei Yu and
                  Junyu Dong},
  editor       = {Toby Walsh and
                  Julie Shah and
                  Zico Kolter},
  title        = {Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic
                  Flow Forecasting},
  booktitle    = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
                  Conference on Innovative Applications of Artificial Intelligence,
                  Fifteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages        = {11463--11471},
  publisher    = {{AAAI} Press},
  year         = {2025},
  url          = {https://doi.org/10.1609/aaai.v39i11.33247},
  doi          = {10.1609/AAAI.V39I11.33247},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/CaoWJYD25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
