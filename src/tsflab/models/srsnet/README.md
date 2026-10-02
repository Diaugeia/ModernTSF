---
name: "SRSNet"
summary: "SRSNet is a patch-based time series forecasting model that introduces the Selective Representation Space (SRS) module, which uses learnable Selective Patching and Dynamic Reassembly techniques to adaptively select and reorder patches from the input context window, paired with an MLP prediction head, to achieve state-of-the-art forecasting performance."
paper: "https://arxiv.org/abs/2510.14510"
paper_title: "Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/decisionintelligence/SRSNet"
revision: "6ee35d498f48eefecf84530b362b137de38e6592"
license: "MIT"
tagline: "Learnable patch scoring and gating, then differentiable soft-rank reassembly of patches before a linear head."
tags: ["mlp", "patching", "selective-representation", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:selective-representation-space", "channel=local:channel-independent-shared-weights", "head=component:flatten_forecast_head", "loss=loss:mse"]
---
# SRSNet

## Key ideas

- `SelectivePatching` embeds each patch and scores it with a small MLP, gating patch embeddings with a sigmoid of the mean-centered score (`alpha`).
- `DynamicReassembly` converts scores into a soft rank and a soft permutation (`assignment`) that reorders patches by score in a differentiable way.
- `SelectiveRepresentationSpace` adds optional positions and LayerNorm; there is no attention.
- `flatten_forecast_head` maps the flattened patch representation to the horizon, inside `revin`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.14510); title: Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/decisionintelligence/SRSNet); revision: `6ee35d498f48eefecf84530b362b137de38e6592`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SRSNet.toml`](../../../../configs/models/SRSNet.toml).

## Differences

Clean-room implementation: confirmed. Paper mapping: Selective Patching → `SelectivePatching`; Dynamic Reassembly → `DynamicReassembly`; SRS → `SelectiveRepresentationSpace`; MLP head → shared `FlattenForecastHead`. Reference-only source code was not copied. The soft-sort relaxation and forecast-only interface are disclosed differences.

## Shared components

- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=512`, `patch_len=24`, `stride=24`, `hidden_size=128`, `dropout=0.2`, `head_dropout=0.1`, `alpha=2.0`, `pos=True`, `head_mode='linear'`, `affine=True`, `subtract_last=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.14510

## Abstract
Time Series Forecasting has made significant progress with the help of Patching technique, which partitions time series into multiple patches to effectively retain contextual semantic information into a representation space beneficial for modeling long-term dependencies. However, conventional patching partitions a time series into adjacent patches, which causes a fixed representation space, thus resulting in insufficiently expressful representations. In this paper, we pioneer the exploration of constructing a selective representation space to flexibly include the most informative patches for forecasting. Specifically, we propose the Selective Representation Space (SRS) module, which utilizes the learnable Selective Patching and Dynamic Reassembly techniques to adaptively select and shuffle the patches from the contextual time series, aiming at fully exploiting the information of contextual time series to enhance the forecasting performance of patch-based models. To demonstrate the effectiveness of SRS module, we propose a simple yet effective SRSNet consisting of SRS and an MLP head, which achieves state-of-the-art performance on real-world datasets from multiple domains. Furthermore, as a novel plug-and-play module, SRS can also enhance the performance of existing patch-based models.

## In TSFLab
Default config: `configs/models/SRSNet.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. Paper mapping: Selective Patching → `SelectivePatching`; Dynamic Reassembly → `DynamicReassembly`; SRS → `SelectiveRepresentationSpace`; MLP head → shared `FlattenForecastHead`. Reference-only source code was not copied. The soft-sort relaxation and forecast-only interface are disclosed differences.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2510-14510,
  author       = {Xingjian Wu and
                  Xiangfei Qiu and
                  Hanyin Cheng and
                  Zhengyu Li and
                  Jilin Hu and
                  Chenjuan Guo and
                  Bin Yang},
  title        = {Enhancing Time Series Forecasting through Selective Representation
                  Spaces: {A} Patch Perspective},
  journal      = {CoRR},
  volume       = {abs/2510.14510},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2510.14510},
  doi          = {10.48550/ARXIV.2510.14510},
  eprinttype   = {arXiv},
  eprint       = {2510.14510},
  timestamp    = {Fri, 14 Nov 2025 15:17:45 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2510-14510.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
