---
name: "ST-SSDL"
summary: "ST-SSDL forecasts node-structured spatiotemporal series through a self-supervised deviation-learning mechanism at its core. It encodes the lookback window with a Chebyshev graph-convolutional GRU, projects the resulting hidden state into a learnable bank of prototype vectors via softmax attention to retrieve a soft 'expected pattern' value and the two nearest prototypes, builds a data-driven decoding graph from the prototype-augmented state, and autoregressively decodes the forecast horizon with a second Chebyshev graph-convolutional GRU; the official method additionally trains the prototype bank with contrastive and deviation losses computed against a distinct historical reference window."
paper: "https://arxiv.org/abs/2510.04908"
paper_title: "How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/Jimmy-7664/ST-SSDL"
revision: "a98331d3a45a081778f3abb608897902a60e301d"
license: "MIT"
---
# ST-SSDL

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2510.04908); title: How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/Jimmy-7664/ST-SSDL); revision: `a98331d3a45a081778f3abb608897902a60e301d`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ST-SSDL.toml`](../../../../configs/models/ST-SSDL.toml).

## Differences

ModernTSF rewrites ST-SSDL locally after reading the paper (arXiv:2510.04908) and inspecting the pinned official codebase (`Jimmy-7664/ST-SSDL@a98331d3a45a081778f3abb608897902a60e301d`, specifically `model_STSSDL/STSSDL.py`, `model_STSSDL/train_STSSDL.py`, and `model_STSSDL/utils.py`). The local implementation keeps the paper's defining operations: the Chebyshev graph-conv GRU encoder/decoder (`ChebGraphConv`, `ChebGRUCell`, including the official gate-split naming where `z` mixes into the candidate and `r` blends the previous state with the candidate), the prototype memory with attention retrieval and top-2 nearest-prototype lookup (rewritten as the reusable `deviation_memory.PrototypeMemory` component, matching the official `construct_prototypes`/`query_prototypes`), and the data-driven decoding graph (`graph_hypernet` + `softmax(relu(e @ e^T))`, matching the official `hypernet` + `support` construction).

**The paper's self-supervised auxiliary losses are not wired into the runner's `forward` contract.** ST-SSDL's central novelty is training the prototype memory with two auxiliary objectives computed against a *second*, distinct historical reference window `x_his` (e.g. the same lookback window sampled one period earlier) that the official `STSSDL.forward(x, x_cov, x_his, y_cov, ...)` takes as an explicit argument: a triplet contrastive loss (`nn.TripletMarginLoss`) pulling the current window's query toward its nearest prototype and away from the second-nearest one, and a deviation loss (`F.l1_loss`) matching the current-vs-historical query distance (`latent_dis`) to the current-vs-historical nearest-prototype distance (`prototype_dis`) — the paper's namesake "how different from the past" signal. ModernTSF's canonical forward signature is `forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)` and carries only one lookback window, so it cannot express a second historical window without a new public input (disallowed by the `implement-model` skill's contract rule). Because the official forecast decode itself only needs the *current* window's encoding (the official decoder input `h_de = cat([h_t, v_t])` depends only on `h_t`/`v_t`, not on the historical `h_a`/`v_a`), `forward` here implements the full forecasting path faithfully using only the current window, and exposes the auxiliary losses as a separate method, `Model.auxiliary_losses(x_enc, x_mark_enc, x_enc_hist, x_mark_enc_hist=None, margin=0.5)`, that takes an explicit historical window and is **not** called by `forward` or by this repository's runner/training loop. Any caller that wants to reproduce the paper's training objective must supply `x_enc_hist` itself (e.g. from a custom training script); this is a documented limitation, not a silent omission.

A second, related simplification follows from the same constraint: the official adaptive decoding graph is built from `h_aug = cat([h_t, v_t, h_a, v_a])` (both current and historical prototype retrievals). Since only the current window is available in `forward`, the local `graph_hypernet` is built from `h_de = cat([h_t, v_t])` alone, which changes the fitted weight's effective input distribution but preserves the same architectural shape and role.

Further differences: `num_nodes`/`adj_mx` are runner-injected, not user parameters, matching the STID/AGCRN convention in this repository; the default adjacency normalization is `symadj` (one static support), matching the official `--adj_type` default, via `graph_utils.adj_to_supports` (Apache-2.0 BasicTS-derived math, already used by other local graph models) in place of the official `utils.load_adj`. The official time-of-day/adaptive/node-embedding front-end (`use_STE`) is not implemented; the official released configs default `use_STE=False`, so the local implementation only needs to match that path, which it does. Curriculum learning (scheduled sampling) is not implemented: the official decoder optionally feeds ground-truth targets instead of its own previous prediction during training (`use_curriculum_learning`, `compute_sampling_threshold`), which requires `labels` and a `batches_seen` counter the runner's forward contract does not provide; the local decoder always feeds back its own prediction (equivalent to `use_curriculum_learning=False` / inference-time behavior). `configs/models/ST-SSDL.toml` uses small smoke-scale widths (`rnn_units=16`, `prototype_num=8`, `prototype_dim=8`, `cheb_k=2`, `rnn_layers=1`) for CPU-only contract/smoke checks, versus the official per-dataset configs (`rnn_units=64-128`, `prototype_num=20`, `prototype_dim=64`).

Canonical evidence is stored in [`verification/evidence/ST-SSDL.json`](../../../../verification/evidence/ST-SSDL.json).

## Shared components

- [`deviation_memory`](../_components/deviation_memory/README.md)
- [`graph_utils`](../_components/graph_utils/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `rnn_units=16`, `rnn_layers=1`, `cheb_k=2`, `prototype_num=8`, `prototype_dim=8`, `output_dim=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.04908

## Abstract
Spatio-temporal forecasting is essential for real-world applications such as traffic management and urban computing. Although recent methods have shown improved accuracy, they often fail to account for dynamic deviations between current inputs and historical patterns, which limits their robustness and adaptability. To address this, we propose ST-SSDL, a Spatio-Temporal framework that incorporates Self-Supervised Deviation Learning to capture such deviations. ST-SSDL discretizes the latent space into a set of prototypes and applies a deviation-aware learning objective, guided by contrastive and deviation losses, to better distinguish the current status of the system from historical patterns. Extensive experiments on six real-world spatio-temporal datasets show that ST-SSDL consistently outperforms state-of-the-art baselines across multiple metrics, and could be flexibly integrated into other backbones as a general enhancement.

## In ModernTSF
Default config: `configs/models/ST-SSDL.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

ModernTSF rewrites ST-SSDL locally after reading the paper (arXiv:2510.04908) and inspecting the pinned official codebase (`Jimmy-7664/ST-SSDL@a98331d3a45a081778f3abb608897902a60e301d`, specifically `model_STSSDL/STSSDL.py`, `model_STSSDL/train_STSSDL.py`, and `model_STSSDL/utils.py`). The local implementation keeps the paper's defining operations: the Chebyshev graph-conv GRU encoder/decoder (`ChebGraphConv`, `ChebGRUCell`, including the official gate-split naming where `z` mixes into the candidate and `r` blends the previous state with the candidate), the prototype memory with attention retrieval and top-2 nearest-prototype lookup (rewritten as the reusable `deviation_memory.PrototypeMemory` component, matching the official `construct_prototypes`/`query_prototypes`), and the data-driven decoding graph (`graph_hypernet` + `softmax(relu(e @ e^T))`, matching the official `hypernet` + `support` construction).

**The paper's self-supervised auxiliary losses are not wired into the runner's `forward` contract.** ST-SSDL's central novelty is training the prototype memory with two auxiliary objectives computed against a *second*, distinct historical reference window `x_his` (e.g. the same lookback window sampled one period earlier) that the official `STSSDL.forward(x, x_cov, x_his, y_cov, ...)` takes as an explicit argument: a triplet contrastive loss (`nn.TripletMarginLoss`) pulling the current window's query toward its nearest prototype and away from the second-nearest one, and a deviation loss (`F.l1_loss`) matching the current-vs-historical query distance (`latent_dis`) to the current-vs-historical nearest-prototype distance (`prototype_dis`) — the paper's namesake "how different from the past" signal. ModernTSF's canonical forward signature is `forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)` and carries only one lookback window, so it cannot express a second historical window without a new public input (disallowed by the `implement-model` skill's contract rule). Because the official forecast decode itself only needs the *current* window's encoding (the official decoder input `h_de = cat([h_t, v_t])` depends only on `h_t`/`v_t`, not on the historical `h_a`/`v_a`), `forward` here implements the full forecasting path faithfully using only the current window, and exposes the auxiliary losses as a separate method, `Model.auxiliary_losses(x_enc, x_mark_enc, x_enc_hist, x_mark_enc_hist=None, margin=0.5)`, that takes an explicit historical window and is **not** called by `forward` or by this repository's runner/training loop. Any caller that wants to reproduce the paper's training objective must supply `x_enc_hist` itself (e.g. from a custom training script); this is a documented limitation, not a silent omission.

A second, related simplification follows from the same constraint: the official adaptive decoding graph is built from `h_aug = cat([h_t, v_t, h_a, v_a])` (both current and historical prototype retrievals). Since only the current window is available in `forward`, the local `graph_hypernet` is built from `h_de = cat([h_t, v_t])` alone, which changes the fitted weight's effective input distribution but preserves the same architectural shape and role.

Further differences: `num_nodes`/`adj_mx` are runner-injected, not user parameters, matching the STID/AGCRN convention in this repository; the default adjacency normalization is `symadj` (one static support), matching the official `--adj_type` default, via `graph_utils.adj_to_supports` (Apache-2.0 BasicTS-derived math, already used by other local graph models) in place of the official `utils.load_adj`. The official time-of-day/adaptive/node-embedding front-end (`use_STE`) is not implemented; the official released configs default `use_STE=False`, so the local implementation only needs to match that path, which it does. Curriculum learning (scheduled sampling) is not implemented: the official decoder optionally feeds ground-truth targets instead of its own previous prediction during training (`use_curriculum_learning`, `compute_sampling_threshold`), which requires `labels` and a `batches_seen` counter the runner's forward contract does not provide; the local decoder always feeds back its own prediction (equivalent to `use_curriculum_learning=False` / inference-time behavior). `configs/models/ST-SSDL.toml` uses small smoke-scale widths (`rnn_units=16`, `prototype_num=8`, `prototype_dim=8`, `cheb_k=2`, `rnn_layers=1`) for CPU-only contract/smoke checks, versus the official per-dataset configs (`rnn_units=64-128`, `prototype_num=20`, `prototype_dim=64`).

Canonical evidence is stored in [`verification/evidence/ST-SSDL.json`](../../../../verification/evidence/ST-SSDL.json).
