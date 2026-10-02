---
name: "AdaMSHyper"
summary: "Ada-MSHyper builds a strided-convolution multi-scale pyramid of the normalized series, learns a sparse binary node-to-hyperedge incidence matrix at every scale from node and hyperedge embeddings, regularizes it with node and hyperedge constraint losses, updates nodes with intra-scale hypergraph convolution attention, mixes all scales' hyperedges with self-attention, and predicts with one linear layer over the concatenated node and hyperedge tokens."
paper: "https://arxiv.org/abs/2410.23992"
paper_title: "Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting"
venue: "NeurIPS"
year: 2024
code: "https://github.com/shangzongjiang/Ada-MSHyper"
revision: "8efd34c8737b0e4dafbd247794213d0278e1dc06"
license: "unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)"
tagline: "Per-scale learned hypergraphs over a conv pyramid, node/hyperedge constraint loss, intra-scale and hyperedge attention."
tags: ["gnn", "hypergraph", "multi-scale", "graph-learning", "attention-variant", "normalization", "time-series", "revin"]
composition: ["normalization=component:revin", "decomposition=local:strided-conv-multi-scale-pyramid", "temporal=component:adaptive_node_embedding_adjacency+local:intra-scale-hypergraph-attention-and-inter-scale-hyperedge-attention", "channel=local:channel-mixing-feature-projections", "head=local:concat-node-hyperedge-linear-head", "loss=loss:mse+local:node-hyperedge-constraint"]
---
# AdaMSHyper

## Key ideas

- `Model.multi_scale` aggregates the series with strided `ScaleConv` layers (Conv1d, BatchNorm, ELU; window equals stride), giving node sets of length `L`, `L/w1`, `L/(w1 w2)`.
- `ScaleHypergraph.incidence` learns an incidence matrix per scale: `adaptive_node_embedding_adjacency` (`softmax(relu(E_node E_hyper^T))`), top-`eta` per node, threshold `beta`, binary in the forward value.
- Hyperedge features are the mean of member nodes; the node constraint (distance of a node to its hyperedges) and the hyperedge constraint (cosine-weighted attract/repel by Euclidean distance with margin `gamma`) form `L_const`, exposed as `aux_loss` while training.
- Nodes are updated by attention-enriched, degree-normalized hypergraph convolution; all scales' hyperedges then exchange information by one self-attention (`hyperedge_attention`), and a single Linear maps the concatenated node and hyperedge tokens to the horizon.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2410.23992); title: Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting; venue/year: NeurIPS / 2024
- [codebase](https://github.com/shangzongjiang/Ada-MSHyper); revision: `8efd34c8737b0e4dafbd247794213d0278e1dc06`; license: `unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AdaMSHyper.toml`](../../../../configs/models/AdaMSHyper.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`adaptive_node_embedding_adjacency`](../_components/adaptive_node_embedding_adjacency/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `window_size=[4, 4]`, `hyper_num=[50, 20, 10]`, `d_embed=16`, `eta=3`, `beta=0.5`, `gamma=4.2`, `lambda_balance=0.5`, `const_weight=1.0`, `dropout=0.1`
<!-- model-card:canonical:end -->
