---
name: "MTGNN"
summary: "MTGNN is a spatiotemporal graph neural network for multivariate time-series forecasting that jointly learns the graph structure and performs message passing. It uses a graph learning module to automatically extract uni-directed inter-variable relations, a mix-hop propagation layer for multi-hop spatial aggregation, and dilated inception layers for multi-scale temporal convolution, all trained end-to-end without requiring a pre-defined graph."
paper: "https://doi.org/10.1145/3394486.3403118"
paper_title: "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"
venue: "KDD 2020"
year: 2020
code: "https://github.com/nnzhan/MTGNN"
revision: "f811746fa7022ebf336f9ecd2434af5f365ecbf6"
license: "MIT"
---
# MTGNN

MTGNN is a spatiotemporal graph neural network for multivariate time-series forecasting that jointly learns the graph structure and performs message passing. It uses a graph learning module to automatically extract uni-directed inter-variable relations, a mix-hop propagation layer for multi-hop spatial aggregation, and dilated inception layers for multi-scale temporal convolution, all trained end-to-end without requiring a pre-defined graph.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1145/3394486.3403118); title: Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks; venue/year: KDD 2020 / 2020
- [codebase](https://github.com/nnzhan/MTGNN); revision: `f811746fa7022ebf336f9ecd2434af5f365ecbf6`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MTGNN.toml`](../../../../configs/models/MTGNN.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper equations; the reference-only
source was not copied. It implements the antisymmetric directed graph constructor,
top-k sparsification, forward/backward mix-hop propagation, causal dilated gated
temporal convolutions, skip path and forecast head. The implementation uses one
kernel size per layer instead of the paper's full dilated-inception kernel bank,
mixes supplied adjacency as a graph prior, and does not reproduce the official
training/data protocol or published metrics.

## Shared components

- [`gated_dilated_conv`](../_components/gated_dilated_conv/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `gcn_depth=2`, `subgraph_size=8`, `node_dim=16`, `conv_channels=16`, `residual_channels=16`, `skip_channels=32`, `end_channels=64`, `layers=3`, `dropout=0.3`, `propalpha=0.05`, `tanhalpha=3.0`, `dilation_exponential=1`, `build_adj=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks
- **Venue**: KDD 2020
- **Published**: 2020 (arXiv: 2020-05)
- **arXiv**: https://arxiv.org/abs/2005.11650

## Abstract
Modeling multivariate time series has long been a subject that has attracted researchers from a diverse range of fields including economics, finance, and traffic. A basic assumption behind multivariate time series forecasting is that its variables depend on one another but, upon looking closely, it is fair to say that existing methods fail to fully exploit latent spatial dependencies between pairs of variables. In recent years, meanwhile, graph neural networks (GNNs) have shown high capability in handling relational dependencies. GNNs require well-defined graph structures for information propagation which means they cannot be applied directly for multivariate time series where the dependencies are not known in advance. In this paper, we propose a general graph neural network framework designed specifically for multivariate time series data. Our approach automatically extracts the uni-directed relations among variables through a graph learning module, into which external knowledge like variable attributes can be easily integrated. A novel mix-hop propagation layer and a dilated inception layer are further proposed to capture the spatial and temporal dependencies within the time series. The graph learning, graph convolution, and temporal convolution modules are jointly learned in an end-to-end framework. Experimental results show that our proposed model outperforms the state-of-the-art baseline methods on 3 of 4 benchmark datasets and achieves on-par performance with other approaches on two traffic datasets which provide extra structural information.

## In ModernTSF
Default config: `configs/models/MTGNN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room rewrite from the paper equations; the reference-only
source was not copied. It implements the antisymmetric directed graph constructor,
top-k sparsification, forward/backward mix-hop propagation, causal dilated gated
temporal convolutions, skip path and forecast head. The implementation uses one
kernel size per layer instead of the paper's full dilated-inception kernel bank,
mixes supplied adjacency as a graph prior, and does not reproduce the official
training/data protocol or published metrics.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/WuPL0CZ20,
  author       = {Zonghan Wu and
                  Shirui Pan and
                  Guodong Long and
                  Jing Jiang and
                  Xiaojun Chang and
                  Chengqi Zhang},
  editor       = {Rajesh Gupta and
                  Yan Liu and
                  Jiliang Tang and
                  B. Aditya Prakash},
  title        = {Connecting the Dots: Multivariate Time Series Forecasting with Graph
                  Neural Networks},
  booktitle    = {{KDD} '20: The 26th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, Virtual Event, CA, USA, August 23-27, 2020},
  pages        = {753--763},
  publisher    = {{ACM}},
  year         = {2020},
  url          = {https://doi.org/10.1145/3394486.3403118},
  doi          = {10.1145/3394486.3403118},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/WuPL0CZ20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
