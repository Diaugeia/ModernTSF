---
name: "PMDformer"
summary: "PMDformer is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It decouples patch-level local shape fluctuations from their mean (trend) level through Patch-Mean Decoupling (PMD), combines Proximal Variable Attention (PVA) to focus on the most relevant inter-variable interactions, and applies Trend Recovery Attention (TRA) to restore long-term trend information, improving both forecasting accuracy and computational efficiency."
paper: "https://arxiv.org/abs/2606.26549"
paper_title: "PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/aohu1105/PMDformer"
revision: "d9296b7b857d8e1075838759ec5d0aa3f3539f7e"
license: "NOASSERTION"
---
# PMDformer

PMDformer is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It decouples patch-level local shape fluctuations from their mean (trend) level through Patch-Mean Decoupling (PMD), combines Proximal Variable Attention (PVA) to focus on the most relevant inter-variable interactions, and applies Trend Recovery Attention (TRA) to restore long-term trend information, improving both forecasting accuracy and computational efficiency.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2606.26549); title: PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/aohu1105/PMDformer); revision: `d9296b7b857d8e1075838759ec5d0aa3f3539f7e`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PMDformer.toml`](../../../../configs/models/PMDformer.toml).

## Differences

Pinned source inspection: `model/PMDformer.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Equations (1)--(3) map to
`patch_mean_decouple()` and `patch_projection`; equations (4)--(5) map to the
last-patch cross-variable attention; equations (6)--(8) map to
`TrendRestorationAttention`; and equation (9) maps to the restored flattened
projection. The linked repository is reference-only; its source was inspected at the pinned revision; no external source code was copied.

This compact rewrite uses one PVA and one parameter-shared TRA block, left-pads
non-divisible histories by replicating the first observation, and uses single-head
trend restoration rather than the full paper training configuration. It does not
claim numerical reference comparison or reproduce the reported hyperparameter sweep.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `patch_len=16`, `num_heads=4`, `dropout=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2026-06)
- **arXiv**: https://arxiv.org/abs/2606.26549

## Abstract
PMDformer introduces three core innovations: (1) Patch-Mean Decoupling (PMD), which separates local shape fluctuations from their absolute magnitude (mean level) to reduce bias and better capture underlying patterns; (2) Proximal Variable Attention (PVA), which strengthens focus on the most relevant and temporally proximal inter-variable interactions; and (3) Trend Restoration Attention (TRA), which restores long-term trend information to improve both responsiveness and stability in forecasting. Together, these components deliver stronger forecasting accuracy and stability while reducing memory usage compared to previous patch-based Transformer methods.

## Source and verification

Pinned source inspection: `model/PMDformer.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Equations (1)--(3) map to
`patch_mean_decouple()` and `patch_projection`; equations (4)--(5) map to the
last-patch cross-variable attention; equations (6)--(8) map to
`TrendRestorationAttention`; and equation (9) maps to the restored flattened
projection. The linked repository is reference-only; its source was inspected at the pinned revision; no external source code was copied.

This compact rewrite uses one PVA and one parameter-shared TRA block, left-pads
non-divisible histories by replicating the first observation, and uses single-head
trend restoration rather than the full paper training configuration. It does not
claim numerical reference comparison or reproduce the reported hyperparameter sweep.

## In ModernTSF
Default config: `configs/models/PMDformer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{hu2026pmdformer,
  author    = {Ao Hu and Liangjian Wen and Jiang Duan and Yong Dai and Yan He and Dongkai Wang and Jun Wang and Yukun Zhang and Ruoxi Jiang and Zenglin Xu},
  title     = {{PMD}former: Patch-Mean Decoupling Information Transformer for Long-term Forecasting},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rfJ41gK9Ct}
}
```
