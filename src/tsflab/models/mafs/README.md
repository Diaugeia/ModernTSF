---
name: "MAFS"
summary: "MAFS (Multi-Agent Forecasting System) is a time series forecasting framework that replaces the conventional single-model paradigm with a cooperative system of specialized agents. The forecasting task is decomposed into multiple sub-tasks — covering different temporal perspectives such as varying resolutions or signal characteristics — each handled by a dedicated iTransformer-based agent. Agents communicate through learnable topology graphs (ring, star, chain, or fully connected), and a lightweight voting aggregator integrates their outputs into the final prediction for each channel."
paper: "https://papers.nips.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html"
paper_title: "Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/h505023992/MAFS"
revision: "4fb26b02824a144d149964b372da98071fc79687"
license: "MIT"
tagline: "Variate-token Transformer agents exchange messages over a learnable topology graph; a voter aggregates forecasts."
tags: ["transformer", "multi-agent", "channel-mixing", "graph-learning", "ensemble"]
composition: ["normalization=none", "decomposition=none", "temporal=local:per-agent-variate-token-embedding", "channel=local:variate-token-attention-agents+local:agent-topology-graph-communication", "head=local:voting-aggregation-of-agent-heads", "loss=loss:mse"]
---
# MAFS

## Key ideas

- Each of `num_agents` agents embeds every variate's whole window into a token and runs its own `AgentEncoderLayer` stack (attention across variates).
- After every layer agents exchange messages through a symmetric, self-looped, degree-normalized adjacency built from learnable `edge_logits` masked by a star, ring, chain or fully-connected topology (`normalized_adjacency`).
- A confidence gate blends each agent's state with its neighbourhood context, and an input-conditioned softmax `voter` weights the agents.
- The output is a head on the voted representation plus the voted sum of per-agent heads; `specialization_loss` exposes the paper's multi-scale prefix pre-training stage but is not used by the default runner.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://papers.nips.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html); title: Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/h505023992/MAFS); revision: `4fb26b02824a144d149964b372da98071fc79687`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MAFS.toml`](../../../../configs/models/MAFS.toml).

## Differences

Pinned source inspection: `mafs_hetegenous_sub_task/models/Agent_iTrans_Cooperation.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This paper-derived rewrite retains iTransformer-style variate-token agents,
multi-scale specialization targets, layer-wise graph communication (Eq. (4)),
masked symmetric normalized topology weights (Eq. (5)), confidence blending,
and an input-conditioned global voter (Eqs. (6)--(7)). Four agents and a star
topology are the compact defaults. The common runner optimizes the complete
point forecast end to end; it does not automatically reproduce the paper's
separate ten-epoch specialization and frozen-agent collaboration stages.
`specialization_targets` and `specialization_loss` expose the fixed-graph
homogeneous prefix stage for experiment harnesses. The reference implementation was inspected at the pinned revision; no external source code was copied. Evidence
is in `../../../verification/evidence/MAFS.json`.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_agents=4`, `num_layers=2`, `num_heads=4`, `topology='star'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **Proceedings**: https://papers.nips.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html

## Abstract
Time series forecasting is a critical and complex task, characterized by diverse temporal patterns, varying statistical properties, and different prediction horizons across datasets and domains. Conventional approaches typically rely on a single, unified model architecture to handle all forecasting scenarios, but such monolithic models struggle to generalize across dynamically evolving time series with shifting patterns. In this paper, we propose a Multi-Agent Forecasting System (MAFS) that abandons the one-size-fits-all paradigm by decomposing the forecasting task into multiple sub-tasks, each handled by a dedicated agent trained on specific temporal perspectives. Agents share and refine information through different communication topology, enabling cooperative reasoning across different temporal views, and a lightweight voting aggregator then integrates their outputs into consistent final predictions. Extensive experiments across 11 benchmarks demonstrate that MAFS significantly outperforms traditional single-model approaches, yielding more robust and adaptable forecasts.

## Source and verification

Pinned source inspection: `mafs_hetegenous_sub_task/models/Agent_iTrans_Cooperation.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This paper-derived rewrite retains iTransformer-style variate-token agents,
multi-scale specialization targets, layer-wise graph communication (Eq. (4)),
masked symmetric normalized topology weights (Eq. (5)), confidence blending,
and an input-conditioned global voter (Eqs. (6)--(7)). Four agents and a star
topology are the compact defaults. The common runner optimizes the complete
point forecast end to end; it does not automatically reproduce the paper's
separate ten-epoch specialization and frozen-agent collaboration stages.
`specialization_targets` and `specialization_loss` expose the fixed-graph
homogeneous prefix stage for experiment harnesses. The reference implementation was inspected at the pinned revision; no external source code was copied. Evidence
is in `../../../verification/evidence/MAFS.json`.

## In TSFLab
Default config: `configs/models/MAFS.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@inproceedings{huang2025mafs,
  author       = {Qihe Huang and Zhengyang Zhou and Yangze Li and Kuo Yang and Binwu Wang and Yang Wang},
  title        = {Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation},
  booktitle    = {Advances in Neural Information Processing Systems},
  year         = {2025},
  url          = {https://papers.nips.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html}
}
```
