---
name: "MoFo"
summary: "MoFo is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It explicitly models periodic patterns by constructing period-structured 2D patch tensors through discrete sampling and introduces a period-aware modulator that applies a learnable regulated relaxation function to guide attention coefficients toward periodic trends, achieving high memory efficiency and fast training speed."
paper: "https://proceedings.neurips.cc/paper_files/paper/2025/hash/7a99ad21706dec5b28f9ad715e12197f-Abstract-Conference.html"
paper_title: "MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/PoorOtterBob/MoFo"
revision: "2d14b47ea839c3809952b412340d72393f2521dc"
license: "MIT"
tagline: "Period-structured 2D patches with attention whose scores are biased by cycle distance through a learnable relaxation."
tags: ["transformer", "periodicity", "attention-variant", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:period-modulated-attention", "channel=local:channel-independent-shared-weights", "head=local:per-step-future-query-projection", "loss=loss:mse"]
---
# MoFo

## Key ideas

- `period_structured_patches` reshapes the series into cycles by phase using the `periodic` length (default 24).
- Each future step is a learned query that attends only to the history values at the same phase across cycles (`PeriodModulatedAttention`).
- `RegulatedRelaxation` converts cycle distance into a learnable additive log-bias on the attention scores.
- Channels share weights (`cias` must be 1); `revin` without affine wraps the model and `projection` maps each query to a scalar.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7a99ad21706dec5b28f9ad715e12197f-Abstract-Conference.html); title: MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/PoorOtterBob/MoFo); revision: `2d14b47ea839c3809952b412340d72393f2521dc`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MoFo.toml`](../../../../configs/models/MoFo.toml).

## Differences

**Paper-driven local implementation.** Discrete sampling rearranges the input
so every row contains period-aligned observations. Learned future queries
attend only to the matching phase history, while the paper's regulated
relaxation function supplies a trainable distance bias to attention scores.
The formula is checked at its exact boundary values. Calendar marks are not
part of this local model. The external repository is reference-only; no source
file was copied or adapted.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=6`, `d_model=64`, `periodic=24`, `head=4`, `d_layers=1`, `bias=1`, `cias=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
The stable periodic patterns present in the time series data serve as the foundation for long-term forecasting. However, existing models suffer from limitations such as continuous and chaotic input partitioning, as well as weak inductive biases, which restrict their ability to capture such recurring structures. In this paper, we propose MoFo, which interprets periodicity as both the correlation of period-aligned time steps and the trend of period-offset time steps. We first design period-structured patches—2D tensors generated through discrete sampling—where each row contains only period-aligned time steps, enabling direct modeling of periodic correlations. Period-offset time steps within a period are aligned in columns. To capture trends across these offset time steps, we introduce a period-aware modulator. This modulator introduces an adaptive strong inductive bias through a regulated relaxation function, encouraging the model to generate attention coefficients that align with periodic trends. This function is end-to-end trainable, enabling the model to adaptively capture the distinct periodic patterns across diverse datasets. Extensive empirical results on widely used benchmark datasets demonstrate that MoFo achieves competitive performance while maintaining high memory efficiency and fast training speed.

## In TSFLab
Default config: `configs/models/MoFo.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Paper-driven local implementation.** Discrete sampling rearranges the input
so every row contains period-aligned observations. Learned future queries
attend only to the matching phase history, while the paper's regulated
relaxation function supplies a trainable distance bias to attention scores.
The formula is checked at its exact boundary values. Calendar marks are not
part of this local model. The external repository is reference-only; no source
file was copied or adapted.

## Citation

```bibtex
@inproceedings{ma2025mofo,
  author    = {Jiaming Ma and Binwu Wang and Qihe Huang and Guanjun Wang and Pengkun Wang and Zhengyang Zhou and Yang Wang},
  title     = {{MoFo}: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://github.com/PoorOtterBob/MoFo}
}
```
