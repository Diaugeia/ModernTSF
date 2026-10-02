---
name: "RLinear"
summary: "RLinear is a time series forecasting model that combines Reversible Instance Normalisation (RevIN) with a single linear projection layer to perform long-term multivariate or univariate forecasting. Despite its simplicity, the model achieves competitive or state-of-the-art performance on standard benchmarks by exploiting the fact that affine mapping dominates forecasting accuracy and that RevIN transforms non-periodic trends into periodic-like patterns that a linear layer can capture effectively."
paper: "https://arxiv.org/abs/2305.10721"
paper_title: "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping"
venue: "arXiv preprint"
year: 2023
code: "https://github.com/plumprc/RTSF"
revision: "0fec00104f754f4fbf795b9b4da5fa2459b32e76"
license: "NOASSERTION"
---
# RLinear

RLinear is a time series forecasting model that combines Reversible Instance Normalisation (RevIN) with a single linear projection layer to perform long-term multivariate or univariate forecasting. Despite its simplicity, the model achieves competitive or state-of-the-art performance on standard benchmarks by exploiting the fact that affine mapping dominates forecasting accuracy and that RevIN transforms non-periodic trends into periodic-like patterns that a linear layer can capture effectively.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2305.10721); title: Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping; venue/year: arXiv preprint / 2023
- [codebase](https://github.com/plumprc/RTSF); revision: `0fec00104f754f4fbf795b9b4da5fa2459b32e76`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/RLinear.toml`](../../../../configs/models/RLinear.toml).

## Differences

Pinned source inspection: `models/RLinear.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from the paper's Section 2.3 baseline and affine equation `Y=XW+b`; the unlicensed reference repository was inspected at the pinned revision; no external source code was copied. The default path is parameter-free RevIN followed by one channel-independent affine history-to-horizon map. TSFLab additionally exposes channel-specific maps, affine/subtract-last RevIN variants, and optional input dropout as explicit ablations; the default preserves the paper baseline and no benchmark-training reference comparison is claimed.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `individual=False`, `affine=False`, `subtract_last=False`, `dropout=0.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-05)
- **arXiv**: https://arxiv.org/abs/2305.10721

## Abstract
Long-term time series forecasting (LTSF) has gained significant attention in recent years. While various specialized designs exist for capturing temporal dependency, recent studies have shown that even a single linear layer can achieve competitive performance. This paper investigates the intrinsic effectiveness of recent LTSF approaches and reveals the critical role of affine mapping. We conduct comprehensive experiments on both simulated and real-world datasets to analyze the components of state-of-the-art models. A theoretical analysis is provided to explain the working mechanisms of affine mapping in periodic signal forecasting. We evaluate the impact of reversible normalization and input horizon extension on model robustness. We find that (1) affine mapping dominates forecasting performance across commonly utilized benchmarks, with models learning similar transition matrices from input to output; (2) affine mapping effectively captures periodic patterns but struggles with non-periodic signals or time series with varying periods across channels; (3) reversible normalization significantly enhances trend forecasting by transforming non-periodic trends into periodic-like patterns; (4) increasing input horizon improves performance on multi-channel data with different periods. Code is available at: https://github.com/plumprc/RTSF. Our findings provide theoretical and experimental insights into the working mechanisms of LTSF models, highlighting both the strengths and limitations of linear approaches. The results suggest that future model development should focus on handling cross-channel period variations and non-periodic components.

## In TSFLab
Default config: `configs/models/RLinear.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Pinned source inspection: `models/RLinear.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from the paper's Section 2.3 baseline and affine equation `Y=XW+b`; the unlicensed reference repository was inspected at the pinned revision; no external source code was copied. The default path is parameter-free RevIN followed by one channel-independent affine history-to-horizon map. TSFLab additionally exposes channel-specific maps, affine/subtract-last RevIN variants, and optional input dropout as explicit ablations; the default preserves the paper baseline and no benchmark-training reference comparison is claimed.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2305-10721,
  author       = {Zhe Li and
                  Shiyi Qi and
                  Yiduo Li and
                  Zenglin Xu},
  title        = {Revisiting Long-term Time Series Forecasting: An Investigation on
                  Linear Mapping},
  journal      = {CoRR},
  volume       = {abs/2305.10721},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.10721},
  doi          = {10.48550/ARXIV.2305.10721},
  eprinttype   = {arXiv},
  eprint       = {2305.10721},
  timestamp    = {Thu, 25 May 2023 15:41:47 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-10721.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
