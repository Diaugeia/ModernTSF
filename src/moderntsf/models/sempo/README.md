---
name: "SEMPO"
summary: "SEMPO is a lightweight time-series foundation model accepted at NeurIPS 2025. It combines an energy-aware spectral decomposition module that captures both high- and low-energy frequency signals with a Mixture-of-Prompts enabled Transformer that routes tokens to small dataset-specific prompt-based experts, enabling strong zero-shot and few-shot generalization across diverse datasets while requiring far less pre-training data and a smaller model size than existing foundation models."
paper: "https://arxiv.org/abs/2510.19710"
paper_title: "SEMPO: Lightweight Foundation Models for Time Series Forecasting"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/mala-lab/SEMPO"
revision: "59233d113f8e47fc32402ef8f371298a4d71fb21"
license: "Apache-2.0"
---
# SEMPO

SEMPO is a lightweight time-series foundation model accepted at NeurIPS 2025. It combines an energy-aware spectral decomposition module that captures both high- and low-energy frequency signals with a Mixture-of-Prompts enabled Transformer that routes tokens to small dataset-specific prompt-based experts, enabling strong zero-shot and few-shot generalization across diverse datasets while requiring far less pre-training data and a smaller model size than existing foundation models.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.19710); title: SEMPO: Lightweight Foundation Models for Time Series Forecasting; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/mala-lab/SEMPO); revision: `59233d113f8e47fc32402ef8f371298a4d71fb21`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SEMPO.toml`](../../../../configs/models/SEMPO.toml).

## Differences

Pinned source inspection: `models/SEMPO.py`, `layers/SEMPO_EncDec.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This local implementation follows the EASD partition and MoP routing and
key/value augmentation equations. It uses differentiable deterministic spectral
masks and one compact attention block. It does not ship SEMPO's pre-training
corpus or weights, stochastic multi-mask reconstruction objective, decoder
stack, or two-stage frozen-backbone tuning procedure. The reference-only
repository was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `patch_len=16`, `num_prompts=4`, `num_heads=4`, `dropout=0.1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: SEMPO: Lightweight Foundation Models for Time Series Forecasting
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.19710

## Abstract
The recent boom of large pre-trained models witnesses remarkable success in developing foundation models (FMs) for time series forecasting. Despite impressive performance across diverse downstream forecasting tasks, existing time series FMs possess massive network architectures and require substantial pre-training on large-scale datasets, which significantly hinders their deployment in resource-constrained environments. In response to this growing tension between versatility and affordability, we propose SEMPO, a novel lightweight foundation model that requires pretraining on relatively small-scale data, yet exhibits strong general time series forecasting. Concretely, SEMPO comprises two key modules: 1) energy-aware SpEctral decomposition module, that substantially improves the utilization of pre-training data by modeling not only the high-energy frequency signals but also the low-energy yet informative frequency signals that are ignored in current methods; and 2) Mixture-of-PrOmpts enabled Transformer, that learns heterogeneous temporal patterns through small dataset-specific prompts and adaptively routes time series tokens to prompt-based experts for parameter-efficient model adaptation across different datasets and domains. Equipped with these modules, SEMPO significantly reduces both pre-training data scale and model size, while achieving strong generalization. Extensive experiments on two large-scale benchmarks covering 16 datasets demonstrate the superior performance of SEMPO in both zero-shot and few-shot forecasting scenarios compared with state-of-the-art methods.

## Source and verification

Pinned source inspection: `models/SEMPO.py`, `layers/SEMPO_EncDec.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This local implementation follows the EASD partition and MoP routing and
key/value augmentation equations. It uses differentiable deterministic spectral
masks and one compact attention block. It does not ship SEMPO's pre-training
corpus or weights, stochastic multi-mask reconstruction objective, decoder
stack, or two-stage frozen-backbone tuning procedure. The reference-only
repository was inspected at the pinned revision; no external source code was copied.

## In ModernTSF
Default config: `configs/models/SEMPO.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@misc{he2025sempo,
  author        = {Hui He and
                  Kun Yi and
                  Yuanchi Ma and
                  Qi Zhang and
                  Zhendong Niu and
                  Guansong Pang},
  title         = {SEMPO: Lightweight Foundation Models for Time Series Forecasting},
  year          = {2025},
  eprint        = {2510.19710},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.19710}
}
```
