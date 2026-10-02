---
name: "RAGC"
summary: "RAGC forecasts node-structured spatiotemporal series through a stochastically regularized, node-embedding adaptive graph convolution at its core. It embeds each lookback window per node, concatenates it with a stochastic-shared-embedding-regularized node embedding and time-of-day/day-of-week embeddings, and stacks feed-forward + adaptive-graph-convolution blocks whose graph operator (the 'Efficient Cosine Operator') builds a gated cosine-normalized support from the node embeddings and diffuses it with a linear-time kernel trick that never materializes a dense adjacency; each block subtracts the graph-smoothed signal from the running state (a residual-difference mechanism) while accumulating the smoothed signal in a skip path, and both paths are projected to the forecast horizon and summed."
paper: "https://arxiv.org/abs/2506.07179"
paper_title: "Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution"
venue: "arXiv"
year: 2025
code: "https://github.com/wkq-wukaiqi/RAGC"
revision: "ee9cbb112b31ca704971a6b1d1c8ffdce6412b95"
license: "Unlicensed (no LICENSE file in official repo; all rights reserved by default)"
tagline: "Node-embedding adaptive graph convolution with a linear-time cosine operator and residual-difference blocks."
tags: ["gnn", "spatiotemporal", "graph-learning", "covariates", "regularization", "traffic"]
composition: ["normalization=none", "decomposition=local:residual-difference-graph-smoothing", "temporal=local:per-node-window-embedding-feedforward", "channel=component:regularized_adaptive_graph_conv", "head=local:residual-plus-skip-linear-heads", "loss=loss:mse"]
---
# RAGC

## Key ideas

- Each node's lookback window is embedded by one linear layer and concatenated with a learned node embedding and time-of-day and day-of-week embeddings read from the calendar covariates.
- `EfficientCosineGraphConv` (the cosine operator) builds a gated, cosine-normalized support from node embeddings and diffuses without materializing a dense adjacency, so cost is linear in the node count.
- `StochasticSharedEmbedding` regularizes the node embedding during training (`sse_p`, `use_sse`).
- Each block subtracts the graph-smoothed signal from the hidden state and accumulates it in a skip path; two linear heads (`regression_layer`, `regression_skip`) are summed.
- `graph_regularization_loss` (Laplacian smoothness on `adj_mx`) is exposed as an optional ablation but is not called by `forward` or the trainer.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2506.07179); title: Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution; venue/year: arXiv / 2025
- [codebase](https://github.com/wkq-wukaiqi/RAGC); revision: `ee9cbb112b31ca704971a6b1d1c8ffdce6412b95`; license: `Unlicensed (no LICENSE file in official repo; all rights reserved by default)`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/RAGC.toml`](../../../../configs/models/RAGC.toml).

## Differences

TSFLab rewrites RAGC locally after reading the paper (arXiv:2506.07179) and inspecting the pinned official codebase (`wkq-wukaiqi/RAGC@ee9cbb112b31ca704971a6b1d1c8ffdce6412b95`, specifically `models/RAGC.py` and `main.py`). The local implementation keeps the paper's defining operations, rewritten as the reusable `regularized_adaptive_graph_conv` component: Stochastic Shared Embedding (`StochasticSharedEmbedding`, matching the official `SSE` module's row-swap-by-probability behavior exactly) and the Efficient Cosine Operator's gated cosine support plus kernel-trick diffusion (`EfficientCosineGraphConv`, matching the official `cos_gconv` branch's `emb_w1`/`emb_w2` gate/filter and `LinearizedSematicGraphConvolutional`'s two-einsum degree-normalized hop). The per-block residual-difference wiring (`hidden = hidden - smoothed; skips += smoothed`) and the dual regression heads (`regression_layer` on the residual state, `regression_skip` on the accumulated skip) also match the official `forward`.

**No explicit auxiliary regularization loss is wired into the runner's `forward` contract.** The official repository does contain one explicit graph-regularization loss term — `RAGC.laplacian_smoothing()`, a Laplacian-smoothness penalty `trace(E^T L E)` on the node embedding `E` against the *static* road-network adjacency `L`, added to the training loss as `loss_mae + loss_lap * 0.001` — but this is gated behind an ablation flag (`--lap_sm`) that **none** of the paper's released per-dataset training commands (SD, GBA, GLA, CA in `README.md`) enable; the paper's headline "regularized adaptive graph convolution" refers to SSE (wired into `forward` above), not this Laplacian term. Following the same pattern used for ST-SSDL's auxiliary self-supervised loss, this optional ablation term is still exposed as a separate method, `Model.graph_regularization_loss()`, computed from the runner-injected static `adj_mx` (identity when none is supplied) and the raw node embedding; it is documented as **not** called by `forward` or by this repository's runner/training loop, matching its role as an optional ablation in the official code rather than RAGC's core mechanism.

Further differences: `num_nodes`/`adj_mx` are runner-injected, not user parameters, matching the STID/AGCRN convention in this repository. The official `softmax_gconv`/`geo_gconv` ablation branches (plain softmax adjacency, or the raw static adjacency, in place of ECO) are not implemented; only the paper's proposed `cos_gconv` (ECO) path is. Dropout inside `EfficientCosineGraphConv`'s output projection defaults to `0.0` here (vs. the official `0.1`) to keep the smoke contract's forward pass deterministic under `model.eval()`-independent unit tests; it can be set via a future parameter if needed. `configs/models/RAGC.toml` uses small smoke-scale widths (`spatial_dim=16`, `embed_dim=16`, `temp_dim_tid=8`, `temp_dim_diw=8`, `num_layer=2`) for CPU-only contract/smoke checks, versus the official per-dataset configs (`spatial_dim=64`, `num_layer=4-8`). The official `time_of_day_size` (`steps_per_day`, e.g. 96/288) is generalized to a configurable `num_time_in_day` (default 24) to match this repository's normalized `[0, 1)` time-in-day convention (`STID`'s convention), and `day_of_week_size` defaults to 7 as in the official code.

Canonical evidence is stored in [`verification/evidence/RAGC.json`](../../../../verification/evidence/RAGC.json).

## Shared components

- [`marks`](../_components/marks/README.md)
- [`regularized_adaptive_graph_conv`](../_components/regularized_adaptive_graph_conv/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `spatial_dim=16`, `embed_dim=16`, `temp_dim_tid=8`, `temp_dim_diw=8`, `num_time_in_day=24`, `num_day_in_week=7`, `num_layer=2`, `order=1`, `sse_p=0.1`, `use_sse=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution
- **Venue**: arXiv
- **Published**: 2025 (arXiv: 2025-06)
- **arXiv**: https://arxiv.org/abs/2506.07179

## Abstract
Traffic prediction is a critical task in spatial-temporal forecasting with broad applications in travel planning and urban management. To model the complex spatial-temporal dependencies in traffic data, Spatial-Temporal Graph Convolutional Networks (STGCNs) have been widely employed, achieving advanced performance. However, when applied to large-scale road networks, the quadratic computational complexity of traditional graph convolution operations severely limits their scalability. Several methods attempt to address this issue through approximation, compression, or spatial partitioning. Nevertheless, these methods often either fail to achieve sufficient computational efficiency or compromise prediction accuracy. To address these challenges, we propose a Regularized Adaptive Graph Convolution (RAGC) model. First, to ensure scalability on large road networks, we develop the Efficient Cosine Operator (ECO), which performs graph convolution based on the cosine similarity of node embeddings with linear time complexity. Second, we introduce a regularized adaptive graph convolution framework that combines Stochastic Shared Embedding (SSE) and adaptive graph convolution through a residual difference mechanism. This design enables the model to learn high-quality node embeddings, thereby improving prediction accuracy while maintaining computational efficiency.

## In TSFLab
Default config: `configs/models/RAGC.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

TSFLab rewrites RAGC locally after reading the paper (arXiv:2506.07179) and inspecting the pinned official codebase (`wkq-wukaiqi/RAGC@ee9cbb112b31ca704971a6b1d1c8ffdce6412b95`, specifically `models/RAGC.py` and `main.py`). The local implementation keeps the paper's defining operations, rewritten as the reusable `regularized_adaptive_graph_conv` component: Stochastic Shared Embedding (`StochasticSharedEmbedding`, matching the official `SSE` module's row-swap-by-probability behavior exactly) and the Efficient Cosine Operator's gated cosine support plus kernel-trick diffusion (`EfficientCosineGraphConv`, matching the official `cos_gconv` branch's `emb_w1`/`emb_w2` gate/filter and `LinearizedSematicGraphConvolutional`'s two-einsum degree-normalized hop). The per-block residual-difference wiring (`hidden = hidden - smoothed; skips += smoothed`) and the dual regression heads (`regression_layer` on the residual state, `regression_skip` on the accumulated skip) also match the official `forward`.

**No explicit auxiliary regularization loss is wired into the runner's `forward` contract.** The official repository does contain one explicit graph-regularization loss term — `RAGC.laplacian_smoothing()`, a Laplacian-smoothness penalty `trace(E^T L E)` on the node embedding `E` against the *static* road-network adjacency `L`, added to the training loss as `loss_mae + loss_lap * 0.001` — but this is gated behind an ablation flag (`--lap_sm`) that **none** of the paper's released per-dataset training commands (SD, GBA, GLA, CA in `README.md`) enable; the paper's headline "regularized adaptive graph convolution" refers to SSE (wired into `forward` above), not this Laplacian term. Following the same pattern used for ST-SSDL's auxiliary self-supervised loss, this optional ablation term is still exposed as a separate method, `Model.graph_regularization_loss()`, computed from the runner-injected static `adj_mx` (identity when none is supplied) and the raw node embedding; it is documented as **not** called by `forward` or by this repository's runner/training loop, matching its role as an optional ablation in the official code rather than RAGC's core mechanism.

Further differences: `num_nodes`/`adj_mx` are runner-injected, not user parameters, matching the STID/AGCRN convention in this repository. The official `softmax_gconv`/`geo_gconv` ablation branches (plain softmax adjacency, or the raw static adjacency, in place of ECO) are not implemented; only the paper's proposed `cos_gconv` (ECO) path is. Dropout inside `EfficientCosineGraphConv`'s output projection defaults to `0.0` here (vs. the official `0.1`) to keep the smoke contract's forward pass deterministic under `model.eval()`-independent unit tests; it can be set via a future parameter if needed. `configs/models/RAGC.toml` uses small smoke-scale widths (`spatial_dim=16`, `embed_dim=16`, `temp_dim_tid=8`, `temp_dim_diw=8`, `num_layer=2`) for CPU-only contract/smoke checks, versus the official per-dataset configs (`spatial_dim=64`, `num_layer=4-8`). The official `time_of_day_size` (`steps_per_day`, e.g. 96/288) is generalized to a configurable `num_time_in_day` (default 24) to match this repository's normalized `[0, 1)` time-in-day convention (`STID`'s convention), and `day_of_week_size` defaults to 7 as in the official code.

Canonical evidence is stored in [`verification/evidence/RAGC.json`](../../../../verification/evidence/RAGC.json).
