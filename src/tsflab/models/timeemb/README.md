---
name: "TimeEmb"
summary: "TimeEmb is a lightweight time-series forecasting model that disentangles static (time-invariant) and dynamic (time-varying) components of a series. A global timestamp-aware embedding bank captures recurring stable patterns, while a frequency-domain filtering mechanism handles short-term fluctuations — the two streams are combined to produce multi-step forecasts. The model can also serve as a plug-in module to enhance existing forecasters with minimal overhead."
paper: "https://arxiv.org/abs/2510.00461"
paper_title: "TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/showmeon/TimeEmb"
revision: "9adf3fba801b34642e7191b45e08aff224b26e67"
license: "NOASSERTION"
---
# TimeEmb

TimeEmb is a lightweight time-series forecasting model that disentangles static (time-invariant) and dynamic (time-varying) components of a series. A global timestamp-aware embedding bank captures recurring stable patterns, while a frequency-domain filtering mechanism handles short-term fluctuations — the two streams are combined to produce multi-step forecasts. The model can also serve as a plug-in module to enhance existing forecasters with minimal overhead.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.00461); title: TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/showmeon/TimeEmb); revision: `9adf3fba801b34642e7191b45e08aff224b26e67`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeEmb.toml`](../../../../configs/models/TimeEmb.toml).

## Differences

- Author source: https://github.com/showmeon/TimeEmb at `9adf3fba801b34642e7191b45e08aff224b26e67`; the repository declares no code license.
Clean-room implementation: confirmed. The implementation was derived independently from the paper's global static spectrum embedding and input-conditioned full-spectrum dynamic filter; source from the unlicensed reference repository was not copied or reused. Calendar marks are optional and fall back to a deterministic zero slot.
- Differences: first forecast-step hour and calendar-day indices come from TSFLab decoder marks, and disabled embedding tables are not registered as dead trainable parameters. Plug-in integrations, published training, and reported results are not reproduced here.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=512`, `use_revin=True`, `use_hour_index=True`, `use_day_index=False`, `scale=0.02`, `hour_length=24`, `day_length=7`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.00461

## Abstract
Temporal non-stationarity, the phenomenon that time series distributions change over time, poses fundamental challenges to reliable time series forecasting. Intuitively, the complex time series can be decomposed into two factors, i.e. time-invariant and time-varying components, which indicate static and dynamic patterns, respectively. Nonetheless, existing methods often conflate the time-varying and time-invariant components, and jointly learn the combined long-term patterns and short-term fluctuations, leading to suboptimal performance facing distribution shifts. To address this issue, we initiatively propose a lightweight static-dynamic decomposition framework, TimeEmb, for time series forecasting. TimeEmb innovatively separates time series into two complementary components: (1) time-invariant component, captured by a novel global embedding module that learns persistent representations across time series, and (2) time-varying component, processed by an efficient frequency-domain filtering mechanism inspired by full-spectrum analysis in signal processing. Experiments on real-world datasets demonstrate that TimeEmb outperforms state-of-the-art baselines and requires fewer computational resources. We conduct comprehensive quantitative and qualitative analyses to verify the efficacy of static-dynamic disentanglement. This lightweight framework can also improve existing time-series forecasting methods with simple integration.

## In TSFLab
Default config: `configs/models/TimeEmb.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

- Author source: https://github.com/showmeon/TimeEmb at `9adf3fba801b34642e7191b45e08aff224b26e67`; the repository declares no code license.
Clean-room implementation: confirmed. The implementation was derived independently from the paper's global static spectrum embedding and input-conditioned full-spectrum dynamic filter; source from the unlicensed reference repository was not copied or reused. Calendar marks are optional and fall back to a deterministic zero slot.
- Differences: first forecast-step hour and calendar-day indices come from TSFLab decoder marks, and disabled embedding tables are not registered as dead trainable parameters. Plug-in integrations, published training, and reported results are not reproduced here.

## Citation

```bibtex
@misc{xia2025timeemb,
  author        = {Mingyuan Xia and
                  Chunxu Zhang and
                  Zijian Zhang and
                  Hao Miao and
                  Qidong Liu and
                  Yuanshao Zhu and
                  Bo Yang},
  title         = {TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting},
  year          = {2025},
  eprint        = {2510.00461},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.00461}
}
```
