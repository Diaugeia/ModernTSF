---
name: "Extralonger"
summary: "Extralonger is a spatiotemporal graph forecaster for extra-long-term traffic prediction. It runs three parallel routes over the same value/calendar input — a node-axis-compressed temporal-attention route, a time-axis-compressed spatial route using a global/local adjacency-blended attention, and a mixed route that applies temporal attention followed by a second attention pass over the transposed feature axis — and fuses their per-route forecasts with a fixed weighting so long-range, data-driven affinity and fixed graph structure both shape the forecast."
paper: "https://arxiv.org/abs/2411.00844"
paper_title: "Extralonger: Toward a Unified Perspective of Spatial-Temporal Factors for Extra-Long-Term Traffic Forecasting"
venue: "NeurIPS 2024 Workshop"
year: 2024
code: "https://github.com/ZhuoLinLi-shu/Extralonger"
revision: "4adb3ec1f0d0a844e9726273a225e640802417cd"
license: "MIT"
---
# Extralonger

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2411.00844); title: Extralonger: Toward a Unified Perspective of Spatial-Temporal Factors for Extra-Long-Term Traffic Forecasting; venue/year: NeurIPS 2024 Workshop / 2024
- [codebase](https://github.com/ZhuoLinLi-shu/Extralonger); revision: `4adb3ec1f0d0a844e9726273a225e640802417cd`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Extralonger.toml`](../../../../configs/models/Extralonger.toml).

## Differences

ModernTSF rewrites Extralonger locally after reading the paper
(arXiv:2411.00844) and inspecting the pinned official codebase
(`ZhuoLinLi-shu/Extralonger@4adb3ec1f0d0a844e9726273a225e640802417cd`,
specifically `model/Extralonger.py`, `model/config.yaml`, and `train.py`).
The local implementation keeps the paper's defining operations:

- **Global-local spatial attention (`GLSAtt`)**: the paper's equation
  combining a dense ("global") softmax attention matrix with an
  adjacency-masked ("local") softmax attention matrix by averaging the two
  probability matrices before applying them to `value` — `GLSAtt =
  (Softmax(alpha_local) + Softmax(alpha_global)) @ V / 2` — matches the
  official `AttentionLayer` with `mode="spatial"` exactly and is the
  extracted, paper-neutral
  [`graph_masked_attention`](../_components/graph_masked_attention/README.md)
  component (`GlobalLocalGraphAttention`).
- **Three-route fusion**: `out = (2 * mixed + spatial + temporal) / 4`,
  matching the official `forward`'s hand-set weights (temporal 0.25, spatial
  0.25, mixed 0.50).
- **Temporal route**: `t_input` (`nn.Linear(num_nodes, input_embedding_dim)`)
  compresses the node axis per time step; time-of-day/day-of-week embeddings
  are concatenated; a stack of self-attention blocks mixes across the time
  axis; `t_output` projects `model_dim -> num_nodes` and (when
  `seq_len != pred_len`) an extra `Linear(seq_len, pred_len)` retimes the
  horizon — matching the official `t_input`/`attn_layers_t`/`t_output`.
- **Spatial route**: `s_input`
  (`nn.Linear(seq_len, model_dim - spatial_embedding_dim)`) compresses the
  time axis per node; a learned node-identity embedding is concatenated; a
  stack of global-local attention blocks mixes across the node axis using the
  adjacency mask; `s_output` projects to `pred_len` — matching the official
  `s_input`/`attn_layers_s`/`s_output`.
- **Mixed route**: the pre-attention temporal-route features are passed
  through a second temporal-attention stack (`mix_temporal_layers`),
  transposed so the feature axis becomes the token axis and the time axis
  becomes the per-token width, run through a second attention stack whose
  token width is `seq_len` (`mix_spatial_layers`, fixed at 2 heads, matching
  the official code's hardcoded `num_heads=2` for this specific pass), and
  projected through `mix_proj` (`Linear(model_dim, num_nodes)` then
  `Linear(seq_len, pred_len)` with transposes in between) — matching the
  official `attn_layers_mix_t`/`attn_layers_mix_s`/`mix_proj`.

Canonical evidence is stored in
[`verification/evidence/Extralonger.json`](../../../../verification/evidence/Extralonger.json).

**Differences from the official implementation.**

- **Adjacency mask is derived locally from the runner-injected dense
  `adj_mx`** (`(adj_mx != 0) | eye(N)`, boolean, with self-loops), rather than
  the official `train.py::get_mask`, which builds the same kind of symmetric
  boolean, self-looped mask but from a raw `data/<DATASET>/adj.csv` edge list
  outside the model. The resulting mask semantics (symmetric boolean
  adjacency with self-loops) are the same; only the source format differs,
  since this repository's runner supplies a dense adjacency matrix rather
  than an edge-list CSV.
- **No input-noise parameter.** The official model adds a learned
  `input_noise` parameter (`nn.Parameter` broadcast onto the raw input before
  the temporal/spatial branches split) that is initialized once and trained
  like a bias; it is not described as a defining mechanism in the paper text
  and its effect at initialization is a fixed additive offset. It is omitted
  here as an undocumented, non-essential detail; removing it does not change
  the three-route/fusion architecture the paper describes.
- **`GlobalLocalGraphAttention` uses reshape-based multi-head splitting**
  (`view` + `movedim`) instead of the official's `torch.cat(torch.split(...))`
  head-splitting idiom. Both compute identical scaled dot-product multi-head
  attention; the reshape form is the standard idiom used elsewhere in this
  repository (e.g. `self_attention_family`).
- **`num_nodes`/`adj_mx` are runner-injected, not user parameters**, matching
  the STID/STAEformer/VisiFold convention in this repository.
- **Smoke-scale defaults.** `configs/models/Extralonger.toml` uses small
  embedding widths (`input_embedding_dim=8`, `tod_embedding_dim=4`,
  `dow_embedding_dim=4`, `spatial_embedding_dim=8`, `feed_forward_dim=16`,
  `num_heads=2`, `num_layers=1`) and `seq_len=pred_len=12` for CPU-only
  contract/smoke checks, versus the paper's PEMS04/PEMS08/SEATTLE presets
  (`model/config.yaml`, widths up to 128-256, `feed_forward_dim` up to 1024)
  and, crucially, versus the paper's actual "extra-long-term" horizons of
  144-2016 steps (0.5 day to 1 week at 5-minute sampling) exercised via
  `train.py`'s `-i`/`-o` flags. This repository verifies Extralonger's
  architecture and executable contract at smoke scale only; it does not
  reproduce the paper's extra-long-horizon training results.

## Shared components

- [`graph_masked_attention`](../_components/graph_masked_attention/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `steps_per_day=24`, `input_embedding_dim=8`, `tod_embedding_dim=4`, `dow_embedding_dim=4`, `spatial_embedding_dim=8`, `feed_forward_dim=16`, `num_heads=2`, `num_layers=1`, `dropout=0.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Extralonger: Toward a Unified Perspective of Spatial-Temporal Factors for Extra-Long-Term Traffic Forecasting
- **Venue**: NeurIPS 2024 Workshop
- **Published**: 2024 (arXiv: 2024-10)
- **arXiv**: https://arxiv.org/abs/2411.00844

## Abstract
Traffic forecasting is fundamental to intelligent transportation systems. Existing methods typically process temporal and spatial factors separately, which incurs high computational cost as the horizon extends, and are largely confined to short- or long-term prediction (up to a few hours). Drawing on the unification of space and time in relativity theory, we propose Extralonger, which treats temporal and spatial factors from a unified perspective rather than modeling them independently. Extralonger fuses temporal, spatial, and mixed spatial-temporal attention routes with hand-crafted weights, extending the feasible prediction horizon to a week on real-world traffic benchmarks while improving training time, inference time, and memory usage over prior long-term forecasting baselines, setting new standards for long-term and extra-long-term traffic forecasting.

## In ModernTSF
Default config: `configs/models/Extralonger.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

ModernTSF rewrites Extralonger locally after reading the paper
(arXiv:2411.00844) and inspecting the pinned official codebase
(`ZhuoLinLi-shu/Extralonger@4adb3ec1f0d0a844e9726273a225e640802417cd`,
specifically `model/Extralonger.py`, `model/config.yaml`, and `train.py`).
The local implementation keeps the paper's defining operations:

- **Global-local spatial attention (`GLSAtt`)**: the paper's equation
  combining a dense ("global") softmax attention matrix with an
  adjacency-masked ("local") softmax attention matrix by averaging the two
  probability matrices before applying them to `value` — `GLSAtt =
  (Softmax(alpha_local) + Softmax(alpha_global)) @ V / 2` — matches the
  official `AttentionLayer` with `mode="spatial"` exactly and is the
  extracted, paper-neutral
  [`graph_masked_attention`](../_components/graph_masked_attention/README.md)
  component (`GlobalLocalGraphAttention`).
- **Three-route fusion**: `out = (2 * mixed + spatial + temporal) / 4`,
  matching the official `forward`'s hand-set weights (temporal 0.25, spatial
  0.25, mixed 0.50).
- **Temporal route**: `t_input` (`nn.Linear(num_nodes, input_embedding_dim)`)
  compresses the node axis per time step; time-of-day/day-of-week embeddings
  are concatenated; a stack of self-attention blocks mixes across the time
  axis; `t_output` projects `model_dim -> num_nodes` and (when
  `seq_len != pred_len`) an extra `Linear(seq_len, pred_len)` retimes the
  horizon — matching the official `t_input`/`attn_layers_t`/`t_output`.
- **Spatial route**: `s_input`
  (`nn.Linear(seq_len, model_dim - spatial_embedding_dim)`) compresses the
  time axis per node; a learned node-identity embedding is concatenated; a
  stack of global-local attention blocks mixes across the node axis using the
  adjacency mask; `s_output` projects to `pred_len` — matching the official
  `s_input`/`attn_layers_s`/`s_output`.
- **Mixed route**: the pre-attention temporal-route features are passed
  through a second temporal-attention stack (`mix_temporal_layers`),
  transposed so the feature axis becomes the token axis and the time axis
  becomes the per-token width, run through a second attention stack whose
  token width is `seq_len` (`mix_spatial_layers`, fixed at 2 heads, matching
  the official code's hardcoded `num_heads=2` for this specific pass), and
  projected through `mix_proj` (`Linear(model_dim, num_nodes)` then
  `Linear(seq_len, pred_len)` with transposes in between) — matching the
  official `attn_layers_mix_t`/`attn_layers_mix_s`/`mix_proj`.

Canonical evidence is stored in
[`verification/evidence/Extralonger.json`](../../../../verification/evidence/Extralonger.json).

**Differences from the official implementation.**

- **Adjacency mask is derived locally from the runner-injected dense
  `adj_mx`** (`(adj_mx != 0) | eye(N)`, boolean, with self-loops), rather than
  the official `train.py::get_mask`, which builds the same kind of symmetric
  boolean, self-looped mask but from a raw `data/<DATASET>/adj.csv` edge list
  outside the model. The resulting mask semantics (symmetric boolean
  adjacency with self-loops) are the same; only the source format differs,
  since this repository's runner supplies a dense adjacency matrix rather
  than an edge-list CSV.
- **No input-noise parameter.** The official model adds a learned
  `input_noise` parameter (`nn.Parameter` broadcast onto the raw input before
  the temporal/spatial branches split) that is initialized once and trained
  like a bias; it is not described as a defining mechanism in the paper text
  and its effect at initialization is a fixed additive offset. It is omitted
  here as an undocumented, non-essential detail; removing it does not change
  the three-route/fusion architecture the paper describes.
- **`GlobalLocalGraphAttention` uses reshape-based multi-head splitting**
  (`view` + `movedim`) instead of the official's `torch.cat(torch.split(...))`
  head-splitting idiom. Both compute identical scaled dot-product multi-head
  attention; the reshape form is the standard idiom used elsewhere in this
  repository (e.g. `self_attention_family`).
- **`num_nodes`/`adj_mx` are runner-injected, not user parameters**, matching
  the STID/STAEformer/VisiFold convention in this repository.
- **Smoke-scale defaults.** `configs/models/Extralonger.toml` uses small
  embedding widths (`input_embedding_dim=8`, `tod_embedding_dim=4`,
  `dow_embedding_dim=4`, `spatial_embedding_dim=8`, `feed_forward_dim=16`,
  `num_heads=2`, `num_layers=1`) and `seq_len=pred_len=12` for CPU-only
  contract/smoke checks, versus the paper's PEMS04/PEMS08/SEATTLE presets
  (`model/config.yaml`, widths up to 128-256, `feed_forward_dim` up to 1024)
  and, crucially, versus the paper's actual "extra-long-term" horizons of
  144-2016 steps (0.5 day to 1 week at 5-minute sampling) exercised via
  `train.py`'s `-i`/`-o` flags. This repository verifies Extralonger's
  architecture and executable contract at smoke scale only; it does not
  reproduce the paper's extra-long-horizon training results.

## Citation

```bibtex
@article{zhang2024extralonger,
  title   = {Extralonger: Toward a Unified Perspective of Spatial-Temporal Factors for Extra-Long-Term Traffic Forecasting},
  author  = {Zhang, Zhiwei and E, Shaojun and Meng, Fandong and Zhou, Jie and Han, Wenjuan},
  journal = {arXiv preprint arXiv:2411.00844},
  year    = {2024}
}
```
