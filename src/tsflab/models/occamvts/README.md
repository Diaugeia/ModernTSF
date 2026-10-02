---
name: "OccamVTS"
summary: "OccamVTS is a knowledge-distillation-based time series forecasting model for the standard time-series setting. It reveals that 99% of large vision model (LVM) parameters are unnecessary for time series tasks and proposes a pyramid-style feature alignment combined with correlation and feature distillation to transfer only the essential low-level textural patterns from pre-trained LVMs into a compact lightweight network — improving accuracy by eliminating overfitting to irrelevant visual features while preserving essential temporal patterns."
paper: "https://arxiv.org/abs/2508.01727"
paper_title: "OccamVTS: Distilling Vision Models to 1% Parameters for Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/sisuolv/OccamVTS"
revision: "a534df297263f0c0f087b513f6bc4acdf7eb1bf2"
license: "NOASSERTION"
tagline: "Compact distilled student: patch Transformer cross-attending a conv encoder over raw, spectrum, and periodic channels."
tags: ["transformer", "patching", "frequency", "cross-modal", "distillation", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:patch-transformer+local:conv-visual-branch", "channel=local:channel-independent-shared-weights", "head=local:cross-modal-fused-linear-head", "loss=loss:mse"]
---
# OccamVTS

## Key ideas

- Implements only the inference-time student; the large vision teacher and the distillation losses are not part of this module.
- `temporal_encoder` runs a Transformer over overlapping patch tokens (`patch_len`, `stride`).
- `visual_augmentation` stacks the raw series, its normalized FFT magnitude, and sine and cosine at `period`; `visual_encoder` convolves that stack.
- `cross_modal` attention lets temporal tokens query the visual features, then pooled tokens go to a linear horizon head; `revin` wraps the model.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2508.01727); title: OccamVTS: Distilling Vision Models to 1% Parameters for Time Series Forecasting; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/sisuolv/OccamVTS); revision: `a534df297263f0c0f087b513f6bc4acdf7eb1bf2`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/OccamVTS.toml`](../../../../configs/models/OccamVTS.toml).

## Differences

Clean-room implementation: confirmed. This is the retained deployment student
described by equations (1), (2), (8), (9), and (12): overlapping temporal patch
tokens are fused with compact visual features built from raw, FFT-magnitude, and
periodic channels. The linked repository is reference-only; its source was not
inspected or copied.

The large pretrained vision teacher, pseudo-image resizing, pyramid feature
alignment, and correlation/feature distillation objectives are training-only and
are not included. Consequently this preset is a compact student trained directly
for forecasting; it is not a reproduction of the paper's teacher-distilled
weights, few-shot results, or zero-shot protocol.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=32`, `patch_len=16`, `stride=8`, `period=24`, `num_heads=4`, `num_layers=1`, `dropout=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: OccamVTS: Distilling Vision Models to 1% Parameters for Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-08)
- **arXiv**: https://arxiv.org/abs/2508.01727

## Abstract
Time series forecasting is fundamental to diverse applications, with recent approaches leverage large vision models (LVMs) to capture temporal patterns through visual representations. We reveal that while vision models enhance forecasting performance, 99% of their parameters are unnecessary for time series tasks. Through cross-modal analysis, we find that time series align with low-level textural features but not high-level semantics, which can impair forecasting accuracy. We propose OccamVTS, a knowledge distillation framework that extracts only the essential 1% of predictive information from LVMs into lightweight networks. Using pre-trained LVMs as privileged teachers, OccamVTS employs pyramid-style feature alignment combined with correlation and feature distillation to transfer beneficial patterns while filtering out semantic noise. Counterintuitively, this aggressive parameter reduction improves accuracy by eliminating overfitting to irrelevant visual features while preserving essential temporal patterns. Extensive experiments across multiple benchmark datasets demonstrate that OccamVTS consistently achieves state-of-the-art performance with only 1% of the original parameters, particularly excelling in few-shot and zero-shot scenarios.

## Source and verification

Clean-room implementation: confirmed. This is the retained deployment student
described by equations (1), (2), (8), (9), and (12): overlapping temporal patch
tokens are fused with compact visual features built from raw, FFT-magnitude, and
periodic channels. The linked repository is reference-only; its source was not
inspected or copied.

The large pretrained vision teacher, pseudo-image resizing, pyramid feature
alignment, and correlation/feature distillation objectives are training-only and
are not included. Consequently this preset is a compact student trained directly
for forecasting; it is not a reproduction of the paper's teacher-distilled
weights, few-shot results, or zero-shot protocol.

## In TSFLab
Default config: `configs/models/OccamVTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/LyuZRLWXL26,
  author       = {Sisuo Lyu and
                  Siru Zhong and
                  Weilin Ruan and
                  Qingxiang Liu and
                  Qingsong Wen and
                  Hui Xiong and
                  Yuxuan Liang},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {OccamVTS: Distilling Vision Models to 1{\%} Parameters for Time Series
                  Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {24216--24225},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i29.39601},
  doi          = {10.1609/AAAI.V40I29.39601},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/LyuZRLWXL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
