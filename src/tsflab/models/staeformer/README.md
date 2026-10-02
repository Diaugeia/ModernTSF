---
name: "STAEformer"
summary: "STAEformer is a spatiotemporal Transformer for node-structured graph data such as traffic networks. It introduces a novel spatio-temporal adaptive embedding that jointly encodes intrinsic spatial relations between nodes and chronological temporal patterns, enabling a standard (vanilla) Transformer encoder—without complex graph convolutions—to achieve state-of-the-art performance on traffic forecasting benchmarks."
paper: "https://arxiv.org/abs/2308.10425"
paper_title: "STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting"
venue: "CIKM 2023"
year: 2023
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# STAEformer

STAEformer is a spatiotemporal Transformer for node-structured graph data such as traffic networks. It introduces a novel spatio-temporal adaptive embedding that jointly encodes intrinsic spatial relations between nodes and chronological temporal patterns, enabling a standard (vanilla) Transformer encoder—without complex graph convolutions—to achieve state-of-the-art performance on traffic forecasting benchmarks.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2308.10425); title: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting; venue/year: CIKM 2023 / 2023
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STAEformer.toml`](../../../../configs/models/STAEformer.toml).

## Differences

TSFLab rewrites STAEformer locally after reviewing the paper and pinned official codebase. Value, calendar, optional spatial, and adaptive spatiotemporal embeddings feed alternating temporal-axis and spatial-axis self-attention blocks before direct horizon projection. Canonical evidence is stored in [`verification/evidence/STAEformer.json`](../../../../verification/evidence/STAEformer.json).

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `steps_per_day=24`, `input_embedding_dim=8`, `tod_embedding_dim=4`, `dow_embedding_dim=4`, `spatial_embedding_dim=0`, `adaptive_embedding_dim=8`, `feed_forward_dim=16`, `num_heads=2`, `num_layers=1`, `dropout=0.1`, `use_mixed_proj=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
- **Venue**: CIKM 2023
- **Published**: 2023 (arXiv: 2023-08)
- **arXiv**: https://arxiv.org/abs/2308.10425

## Abstract
With the rapid development of the Intelligent Transportation System (ITS), accurate traffic forecasting has emerged as a critical challenge. The key bottleneck lies in capturing the intricate spatio-temporal traffic patterns. In recent years, numerous neural networks with complicated architectures have been proposed to address this issue. However, the advancements in network architectures have encountered diminishing performance gains. In this study, we present a novel component called spatio-temporal adaptive embedding that can yield outstanding results with vanilla transformers. Our proposed Spatio-Temporal Adaptive Embedding transformer (STAEformer) achieves state-of-the-art performance on five real-world traffic forecasting datasets. Further experiments demonstrate that spatio-temporal adaptive embedding plays a crucial role in traffic forecasting by effectively capturing intrinsic spatio-temporal relations and chronological information in traffic time series.

## In TSFLab
Default config: `configs/models/STAEformer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

TSFLab rewrites STAEformer locally after reviewing the paper and pinned official codebase. Value, calendar, optional spatial, and adaptive spatiotemporal embeddings feed alternating temporal-axis and spatial-axis self-attention blocks before direct horizon projection. Canonical evidence is stored in [`verification/evidence/STAEformer.json`](../../../../verification/evidence/STAEformer.json).

## Citation

```bibtex
@misc{liu2023staeformer,
  author        = {Hangchen Liu and
                  Zheng Dong and
                  Renhe Jiang and
                  Jiewen Deng and
                  Jinliang Deng and
                  Quanjun Chen and
                  Xuan Song},
  title         = {STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting},
  year          = {2023},
  eprint        = {2308.10425},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2308.10425}
}
```
