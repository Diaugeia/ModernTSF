---
name: "TimeO1"
summary: "Time-o1 is a model-agnostic transformation-augmented forecasting objective that aligns the most significant decorrelated label components. The local runtime provides per-variate SVD basis fitting, the published mixed objective, and a small independent temporal carrier model."
paper: "https://arxiv.org/abs/2505.17847"
paper_title: "Time-o1: Time-Series Forecasting Needs Transformed Label Alignment"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/Master-PLC/Time-o1"
revision: "c93d4c545ee0fe4929d0b8ba37268d0da161bb9d"
license: "MIT"
---
# TimeO1

Time-o1 is a model-agnostic transformation-augmented training objective. The local module exposes its per-variate SVD basis fitting and transformed-label loss alongside a small independent temporal backbone required by the repository's model contract.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.17847); title: Time-o1: Time-Series Forecasting Needs Transformed Label Alignment; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/Master-PLC/Time-o1); revision: `c93d4c545ee0fe4929d0b8ba37268d0da161bb9d`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeO1.toml`](../../../../configs/models/TimeO1.toml).

## Differences

Pinned source inspection: `utils/polynomial.py`, `README.md` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Time-o1 does not prescribe a forecasting architecture, so the local temporal MLP plus linear skip is only a runnable carrier. Experiment code must fit the projection on training labels and explicitly call `transformed_alignment_loss`; the generic MSE runner does not activate Time-o1 automatically. The reference-only codebase was inspected at the pinned revision; no external source code was copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `alpha=0.8`, `rank_ratio=0.5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Time-o1: Time-Series Forecasting Needs Transformed Label Alignment
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.17847

## Abstract
Training time-series forecast models presents unique challenges in designing effective learning objectives. Existing methods predominantly utilize the temporal mean squared error, which faces two critical challenges: (1) label autocorrelation, which leads to bias from the label sequence likelihood; (2) excessive amount of tasks, which increases with the forecast horizon and complicates optimization. To address these challenges, we propose Time-o1, a transformation-augmented learning objective tailored for time-series forecasting. The central idea is to transform the label sequence into decorrelated components with discriminated significance. Models are then trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Extensive experiments demonstrate that Time-o1 achieves state-of-the-art performance and is compatible with various forecast models. Code is available at https://github.com/Master-PLC/Time-o1.

## Source and verification

Pinned source inspection: `utils/polynomial.py`, `README.md` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Time-o1 does not prescribe a forecasting architecture, so the local temporal MLP plus linear skip is only a runnable carrier. Experiment code must fit the projection on training labels and explicitly call `transformed_alignment_loss`; the generic MSE runner does not activate Time-o1 automatically. The reference-only codebase was inspected at the pinned revision; no external source code was copied.

## In TSFLab
Default config: `configs/models/TimeO1.toml`; model specification: `spec.py`; clean-room objective/backbone: `model.py`.

## Citation

```bibtex
@misc{wang2025timeo,
  author        = {Hao Wang and
                  Licheng Pan and
                  Zhichao Chen and
                  Xu Chen and
                  Qingyang Dai and
                  Lei Wang and
                  Haoxuan Li and
                  Zhouchen Lin},
  title         = {Time-o1: Time-Series Forecasting Needs Transformed Label Alignment},
  year          = {2025},
  eprint        = {2505.17847},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.17847}
}
```
