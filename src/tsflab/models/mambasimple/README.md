---
name: "MambaSimple"
summary: "MambaSimple is a time series forecasting model built on the Mamba selective state space architecture. It adapts Mamba's selective scan mechanism — where SSM parameters are functions of the input, allowing the model to selectively propagate or forget information — into a pure PyTorch implementation that requires no custom CUDA operators, making it portable across CPU, CUDA, and MPS backends."
paper: "https://arxiv.org/abs/2312.00752"
paper_title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
venue: "arXiv preprint"
year: 2023
code: "https://github.com/thuml/Time-Series-Library"
revision: "4e938a1767106324dd753b2a44832bf870a0252e"
license: "MIT"
tagline: "Pure-PyTorch selective-scan Mamba blocks over time tokens, then a linear map from history length to horizon."
tags: ["ssm", "channel-mixing", "normalization", "portable"]
composition: ["normalization=local:instance-mean-std-normalization", "decomposition=none", "temporal=component:mamba", "channel=local:channel-mixing-input-projection", "head=local:linear-horizon-projection", "loss=loss:mse"]
---
# MambaSimple

## Key ideas

- Each time step's channel vector is projected to `d_model` and processed by `MambaResidualBlock`s (`mamba`), the selective state-space scan written in plain PyTorch with no CUDA kernels.
- A final RMSNorm and channel projection restore `enc_in` features, then a `Linear(seq_len, pred_len)` along time produces the horizon.
- Each variate is standardized over its window and restored on the output; time marks are ignored.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2312.00752); title: Mamba: Linear-Time Sequence Modeling with Selective State Spaces; venue/year: arXiv preprint / 2023
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `4e938a1767106324dd753b2a44832bf870a0252e`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MambaSimple.toml`](../../../../configs/models/MambaSimple.toml).

## Differences

**Clean-room implementation: confirmed.** The Mamba equations are provided by the canonical shared pure-PyTorch block and the forecast wrapper was independently designed. Inputs are `[B, seq_len, enc_in]`, outputs are `[B, pred_len, enc_in]`, and marks are ignored. The Time-Series-Library link is reference-only; no source or checkpoint reference comparison is claimed.

## Shared components

- [`mamba`](../_components/mamba/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_state=16`, `e_layers=2`, `expand=2`, `d_conv=4`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-12)
- **arXiv**: https://arxiv.org/abs/2312.00752

## Abstract
Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

## In TSFLab
Default config: `configs/models/MambaSimple.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** The Mamba equations are provided by the canonical shared pure-PyTorch block and the forecast wrapper was independently designed. Inputs are `[B, seq_len, enc_in]`, outputs are `[B, pred_len, enc_in]`, and marks are ignored. The Time-Series-Library link is reference-only; no source or checkpoint reference comparison is claimed.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2312-00752,
  author       = {Albert Gu and
                  Tri Dao},
  title        = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  journal      = {CoRR},
  volume       = {abs/2312.00752},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2312.00752},
  doi          = {10.48550/ARXIV.2312.00752},
  eprinttype   = {arXiv},
  eprint       = {2312.00752},
  timestamp    = {Sun, 19 Jan 2025 13:42:18 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2312-00752.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
