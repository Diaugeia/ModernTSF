---
name: "DSTAGNN"
summary: "The DSTAGNN paper combines a data-derived pattern-aware graph, spatial-temporal attention with residual attention, Chebyshev graph convolution, and multi-scale gated temporal convolution. This clean-room implementation couples dense temporal and spatial multi-head attention to attention-modulated Chebyshev filtering and three gated temporal receptive fields."
paper: "https://proceedings.mlr.press/v162/lan22a.html"
paper_title: "DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting"
venue: "ICML 2022"
year: 2022
code: "https://github.com/SYLan2019/DSTAGNN"
revision: "10da0e08ec3cf8845841741b8434fd76fd48ff84"
license: "NOASSERTION"
---
# DSTAGNN

The DSTAGNN paper combines a data-derived pattern-aware graph, spatial-temporal attention with residual attention, Chebyshev graph convolution, and multi-scale gated temporal convolution. This clean-room implementation couples dense temporal and spatial multi-head attention to attention-modulated Chebyshev filtering and three gated temporal receptive fields.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://proceedings.mlr.press/v162/lan22a.html); title: DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting; venue/year: ICML 2022 / 2022
- [codebase](https://github.com/SYLan2019/DSTAGNN); revision: `10da0e08ec3cf8845841741b8434fd76fd48ff84`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DSTAGNN.toml`](../../../../configs/models/DSTAGNN.toml).

## Differences

- Clean-room implementation: confirmed. The replacement follows the public paper description; neither the unlicensed official repository nor the prior CauAir-derived file was used as implementation source.
- Formula mapping: `AxisAttention` provides temporal and dynamic spatial multi-head attention; `DynamicChebyshevConvolution` modulates each graph polynomial by spatial attention; `MultiScaleGatedTemporalConvolution` uses parallel receptive fields 3, 5, and 7.
- Adjacency and marks: a shape-checked `adj_mx` supplies Chebyshev supports; attention remains dense. Timestamp marks are intentionally not consumed because this entry models the paper's value stream only.
- Differences and limits: the paper's data-derived pattern-aware adjacency, temporal-distance matrix, residual-attention accumulation across multiple blocks, preprocessing, and training objective are not reproduced. A missing graph uses identity adjacency.

## Shared components

- [`graph_spectral`](../_components/graph_spectral/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `d_model=64`, `d_k=8`, `d_v=8`, `n_heads=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting
- **Venue**: ICML 2022
- **Published**: 2022
- **arXiv**: N/A

## Abstract
As a typical problem in time series analysis, traffic flow prediction is one of the most important application fields of machine learning. However, achieving highly accurate traffic flow prediction is a challenging task, due to the presence of complex dynamic spatial-temporal dependencies within a road network. This paper proposes a novel Dynamic Spatial-Temporal Aware Graph Neural Network (DSTAGNN) to model the complex spatial-temporal interaction in road network. First, considering the fact that historical data carries intrinsic dynamic information about the spatial structure of road networks, we propose a new dynamic spatial-temporal aware graph based on a data-driven strategy to replace the pre-defined static graph usually used in traditional graph convolution. Second, we design a novel graph neural network architecture, which can not only represent dynamic spatial relevance among nodes with an improved multi-head attention mechanism, but also acquire the wide range of dynamic temporal dependency from multi-receptive field features via multi-scale gated convolution. Extensive experiments on real-world data sets demonstrate that our proposed method significantly outperforms the state-of-the-art methods.

## In TSFLab
Default config: `configs/models/DSTAGNN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

- Clean-room implementation: confirmed. The replacement follows the public paper description; neither the unlicensed official repository nor the prior CauAir-derived file was used as implementation source.
- Formula mapping: `AxisAttention` provides temporal and dynamic spatial multi-head attention; `DynamicChebyshevConvolution` modulates each graph polynomial by spatial attention; `MultiScaleGatedTemporalConvolution` uses parallel receptive fields 3, 5, and 7.
- Adjacency and marks: a shape-checked `adj_mx` supplies Chebyshev supports; attention remains dense. Timestamp marks are intentionally not consumed because this entry models the paper's value stream only.
- Differences and limits: the paper's data-derived pattern-aware adjacency, temporal-distance matrix, residual-attention accumulation across multiple blocks, preprocessing, and training objective are not reproduced. A missing graph uses identity adjacency.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/LanMHWYL22,
  author       = {Shiyong Lan and
                  Yitong Ma and
                  Weikang Huang and
                  Wenwu Wang and
                  Hongyu Yang and
                  Pyang Li},
  editor       = {Kamalika Chaudhuri and
                  Stefanie Jegelka and
                  Le Song and
                  Csaba Szepesv{\'{a}}ri and
                  Gang Niu and
                  Sivan Sabato},
  title        = {{DSTAGNN:} Dynamic Spatial-Temporal Aware Graph Neural Network for
                  Traffic Flow Forecasting},
  booktitle    = {International Conference on Machine Learning, {ICML} 2022, 17-23 July
                  2022, Baltimore, Maryland, {USA}},
  series       = {Proceedings of Machine Learning Research},
  pages        = {11906--11917},
  publisher    = {{PMLR}},
  year         = {2022},
  url          = {https://proceedings.mlr.press/v162/lan22a.html},
  timestamp    = {Thu, 05 Jan 2023 08:20:54 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/LanMHWYL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
