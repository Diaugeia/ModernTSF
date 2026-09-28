---
name: "VisiFold"
summary: "VisiFold is a spatiotemporal graph forecaster for long-term traffic prediction on node-structured data. It folds each node's entire input history into a single per-node token with a small MLP (a temporal folding graph, avoiding cross-step message passing across stacked snapshots) and, during training, applies a node-visibility mechanism — node-level random masking followed by shuffle-then-subgraph-grouped self-attention — to bound the cost of all-pairs node mixing before projecting each surviving node token directly to the forecast horizon."
paper: "https://arxiv.org/abs/2603.11816"
paper_title: "VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility"
venue: "ICDE 2026"
year: 2026
code: "https://github.com/PlanckChang/VisiFold"
revision: "cc035f7e803ca793b132bec3a834e8e2b39a9f99"
license: "MIT"
---
# VisiFold

<!-- model-card:canonical:start -->
## Method overview

VisiFold is a spatiotemporal graph forecaster for long-term traffic prediction on node-structured data.

## Core architecture

It folds each node's entire input history into a single per-node token with a small MLP (a temporal folding graph, avoiding cross-step message passing across stacked snapshots) and, during training, applies a node-visibility mechanism — node-level random masking followed by shuffle-then-subgraph-grouped self-attention — to bound the cost of all-pairs node mixing before projecting each surviving node token directly to the forecast horizon.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2603.11816); title: VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility; venue/year: ICDE 2026 / 2026
- [codebase](https://github.com/PlanckChang/VisiFold); revision: `cc035f7e803ca793b132bec3a834e8e2b39a9f99`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/VisiFold.toml`](../../../../configs/models/VisiFold.toml).

## Differences

ModernTSF rewrites VisiFold locally after reading the paper (arXiv:2603.11816) and inspecting the pinned official codebase (`PlanckChang/VisiFold@cc035f7e803ca793b132bec3a834e8e2b39a9f99`, specifically `model/VisiFold.py`, `model/config.yaml`, and `README.md`). The local implementation keeps the paper's two defining operations:

- **Temporal folding graph**: `history[..., 0].transpose(1, 2)` reshapes `(B, T, N)` values to `(B, N, T)` and a two-layer MLP (`fold_input`) projects the folded per-node history to `input_embedding_dim`, matching the official `s_input` block exactly (`nn.Linear(in_steps, D) -> ReLU -> nn.Linear(D, D)`). Time-of-day, day-of-week, and adaptive node embeddings are concatenated exactly as in the official `forward`.
- **Node visibility (training only)**: `random_mask_tokens` keeps a sorted random `(1 - mask_ratio)` subset of node tokens (official `random_mask_token`), `shuffle_tokens` applies a per-sample random permutation (official `random_shuffle`), and `group_into_subgraphs` reshapes the surviving tokens into zero-padded `subgraph_size` groups (official `split_group`) before self-attention; `ungroup_subgraphs` and `unshuffle_tokens` invert the transform afterward (official `recover_group` / `unshuffle`), mirroring the official `forward`'s `if self.training:` branch.

Canonical evidence is stored in [`verification/evidence/VisiFold.json`](../../../../verification/evidence/VisiFold.json).

**Differences from the official implementation.**

- **Attention block simplified to `nn.MultiheadAttention`.** The official `MultiHeadAttention` supports an additive `edge_emb` bias, but VisiFold's own `forward` never passes `edge_emb` to `attn_layers_s`, so that code path is dead in the official model too. The local `SelfAttentionLayer` uses `nn.MultiheadAttention` (batch-first) instead of re-deriving a custom scaled-dot-product implementation; the residual/LayerNorm order (post-norm: attend, add, norm, feed-forward, add, norm) is preserved exactly.
- **Masked-node targets are scattered back to a full `(B, N, pred_len)` tensor at train time** (`full.scatter_` on `keep_indices`) so the executable contract's output shape matches `(B, pred_len, N)` regardless of `mask_ratio`. The official reference code returns only the visible/kept nodes during training and expects the training script to apply the same drop to the target `y` (`random_token_dropout_y`); ModernTSF's runner contract requires a full-shaped output, so masked-out node positions are zero-filled here. Any training loop that uses `mask_ratio > 0` with this contract should mask the loss to the same `keep_indices` rather than relying on the zero-filled positions as targets — this repository's runner currently does not do so, since VisiFold is registered for smoke/contract verification, not for a masked-loss training loop.
- **`num_nodes`/`adj_mx` are runner-injected, not user parameters**, matching the STID/STAEformer convention in this repository; VisiFold's official code does not consume a fixed adjacency at all (attention operates over the free node-token set), so `adj_mx` is accepted only for interface uniformity and discarded.
- **Smoke-scale defaults.** `configs/models/VisiFold.toml` uses small embedding widths (`input_embedding_dim=8`, `feed_forward_dim=16`, `num_heads=2`, `num_layers=1`) and `subgraph_size=4` for CPU-only contract/smoke checks, versus the paper's PEMS04/PEMS08/SEATTLE presets (`model/config.yaml`) with widths up to 64-256 and `subgraph_size` 30-50.
- **Dropout wiring.** The local implementation exposes a single `dropout` parameter applied inside `SelfAttentionLayer`; the official `SelfAttentionLayer`/`MultiHeadAttention` also default `dropout=0`, so behavior at the smoke-scale default (`dropout=0.0`) is identical.

## Shared components

- [`marks`](../_components/marks/README.md)
- [`node_visibility`](../_components/node_visibility/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `steps_per_day=24`, `input_embedding_dim=8`, `tod_embedding_dim=4`, `dow_embedding_dim=4`, `spatial_embedding_dim=8`, `feed_forward_dim=16`, `num_heads=2`, `num_layers=1`, `mask_ratio=0.2`, `subgraph_size=4`, `dropout=0.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility
- **Venue**: ICDE 2026
- **Published**: 2026 (arXiv: 2026-03)
- **arXiv**: https://arxiv.org/abs/2603.11816

## Abstract
Traffic forecasting is a cornerstone of intelligent transportation systems. While existing research has made significant progress in short-term prediction, long-term forecasting remains a largely uncharted and challenging frontier. Extending the prediction horizon intensifies two critical issues: escalating computational resource consumption and increasingly complex spatial-temporal dependencies. Current approaches, which rely on spatial-temporal graphs and process temporal and spatial dimensions separately, suffer from snapshot-stacking inflation and cross-step fragmentation. To overcome these limitations, we propose VisiFold. Our framework introduces a novel temporal folding graph that consolidates a sequence of temporal snapshots into a single graph. Furthermore, we present a node visibility mechanism that incorporates node-level masking and subgraph sampling to overcome the computational bottleneck imposed by large node counts. Extensive experiments show that VisiFold not only drastically reduces resource consumption but also outperforms existing baselines in long-term forecasting tasks. Remarkably, even with a high mask ratio of 80%, VisiFold maintains its performance advantage. By effectively breaking the resource constraints in both temporal and spatial dimensions, our work paves the way for more accurate long-term traffic forecasting.

## In ModernTSF
Default config: `configs/models/VisiFold.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

ModernTSF rewrites VisiFold locally after reading the paper (arXiv:2603.11816) and inspecting the pinned official codebase (`PlanckChang/VisiFold@cc035f7e803ca793b132bec3a834e8e2b39a9f99`, specifically `model/VisiFold.py`, `model/config.yaml`, and `README.md`). The local implementation keeps the paper's two defining operations:

- **Temporal folding graph**: `history[..., 0].transpose(1, 2)` reshapes `(B, T, N)` values to `(B, N, T)` and a two-layer MLP (`fold_input`) projects the folded per-node history to `input_embedding_dim`, matching the official `s_input` block exactly (`nn.Linear(in_steps, D) -> ReLU -> nn.Linear(D, D)`). Time-of-day, day-of-week, and adaptive node embeddings are concatenated exactly as in the official `forward`.
- **Node visibility (training only)**: `random_mask_tokens` keeps a sorted random `(1 - mask_ratio)` subset of node tokens (official `random_mask_token`), `shuffle_tokens` applies a per-sample random permutation (official `random_shuffle`), and `group_into_subgraphs` reshapes the surviving tokens into zero-padded `subgraph_size` groups (official `split_group`) before self-attention; `ungroup_subgraphs` and `unshuffle_tokens` invert the transform afterward (official `recover_group` / `unshuffle`), mirroring the official `forward`'s `if self.training:` branch.

Canonical evidence is stored in [`verification/evidence/VisiFold.json`](../../../../verification/evidence/VisiFold.json).

**Differences from the official implementation.**

- **Attention block simplified to `nn.MultiheadAttention`.** The official `MultiHeadAttention` supports an additive `edge_emb` bias, but VisiFold's own `forward` never passes `edge_emb` to `attn_layers_s`, so that code path is dead in the official model too. The local `SelfAttentionLayer` uses `nn.MultiheadAttention` (batch-first) instead of re-deriving a custom scaled-dot-product implementation; the residual/LayerNorm order (post-norm: attend, add, norm, feed-forward, add, norm) is preserved exactly.
- **Masked-node targets are scattered back to a full `(B, N, pred_len)` tensor at train time** (`full.scatter_` on `keep_indices`) so the executable contract's output shape matches `(B, pred_len, N)` regardless of `mask_ratio`. The official reference code returns only the visible/kept nodes during training and expects the training script to apply the same drop to the target `y` (`random_token_dropout_y`); ModernTSF's runner contract requires a full-shaped output, so masked-out node positions are zero-filled here. Any training loop that uses `mask_ratio > 0` with this contract should mask the loss to the same `keep_indices` rather than relying on the zero-filled positions as targets — this repository's runner currently does not do so, since VisiFold is registered for smoke/contract verification, not for a masked-loss training loop.
- **`num_nodes`/`adj_mx` are runner-injected, not user parameters**, matching the STID/STAEformer convention in this repository; VisiFold's official code does not consume a fixed adjacency at all (attention operates over the free node-token set), so `adj_mx` is accepted only for interface uniformity and discarded.
- **Smoke-scale defaults.** `configs/models/VisiFold.toml` uses small embedding widths (`input_embedding_dim=8`, `feed_forward_dim=16`, `num_heads=2`, `num_layers=1`) and `subgraph_size=4` for CPU-only contract/smoke checks, versus the paper's PEMS04/PEMS08/SEATTLE presets (`model/config.yaml`) with widths up to 64-256 and `subgraph_size` 30-50.
- **Dropout wiring.** The local implementation exposes a single `dropout` parameter applied inside `SelfAttentionLayer`; the official `SelfAttentionLayer`/`MultiHeadAttention` also default `dropout=0`, so behavior at the smoke-scale default (`dropout=0.0`) is identical.

## Citation

```bibtex
@article{zhang2026visifold,
  title   = {VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility},
  author  = {Zhang, Zhiwei and Du, Xinyi and Wang, Weihao and Guo, Xuanchi and Han, Wenjuan},
  journal = {arXiv preprint arXiv:2603.11816},
  year    = {2026}
}
```
