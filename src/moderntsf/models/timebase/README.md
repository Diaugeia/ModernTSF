---
name: "TimeBase"
summary: "TimeBase is an ultra-lightweight network for long-term time series forecasting that extracts core basis temporal components from the input window and transforms traditional point-level prediction into efficient segment-level forecasting, exploiting the temporal pattern similarity and low-rank structure inherent in long-horizon time series data."
paper: "https://proceedings.mlr.press/v267/huang25az.html"
paper_title: "TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting"
venue: "ICML 2025"
year: 2025
code: "https://github.com/hqh0728/TimeBase"
revision: "369b330f3d77371fcc7e8c75c808d01330c40899"
license: "MIT"
---
# TimeBase

TimeBase is an ultra-lightweight network for long-term time series forecasting that extracts core basis temporal components from the input window and transforms traditional point-level prediction into efficient segment-level forecasting, exploiting the temporal pattern similarity and low-rank structure inherent in long-horizon time series data.

<!-- model-card:canonical:start -->
## Method overview

TimeBase is an ultra-lightweight network for long-term time series forecasting that extracts core basis temporal components from the input window and transforms traditional point-level prediction into efficient segment-level forecasting, exploiting the temporal pattern similarity and low-rank structure inherent in long-horizon time series data.

## Core architecture

TimeBase is an ultra-lightweight network for long-term time series forecasting that extracts core basis temporal components from the input window and transforms traditional point-level prediction into efficient segment-level forecasting, exploiting the temporal pattern similarity and low-rank structure inherent in long-horizon time series data.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.mlr.press/v267/huang25az.html); title: TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting; venue/year: ICML 2025 / 2025
- [codebase](https://github.com/hqh0728/TimeBase); revision: `369b330f3d77371fcc7e8c75c808d01330c40899`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TimeBase.toml`](../../../../configs/models/TimeBase.toml).

## Differences

Pinned source inspection: `models/TimeBase.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed from the PMLR paper. The licensed author repository is pinned as `reference-only`; its source was inspected at the pinned revision; no external source code was copied for this independent implementation.
- Equations 1–4 are represented by segmenting `X`, applying `X_basis=BasisExtract(X_his)`, applying the segment-level forecast map, and flattening/trimming the result. Equations 5–7 are represented by `G=X_basis^T X_basis` and the off-diagonal Frobenius penalty.
- `orthogonal_weight = 0.08` is a runnable point from the paper's 0.00–0.20 sweep, not a universal paper setting; dataset-specific result reproduction is outside this structural validation.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `period_len=24`, `basis_num=6`, `individual=False`, `orthogonal_weight=0.08`, `use_period_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting
- **Venue**: ICML 2025
- **Published**: 2025
- **Paper**: https://proceedings.mlr.press/v267/huang25az.html

## Abstract
Long-term time series forecasting (LTSF) has traditionally relied on large parameters to capture extended temporal dependencies, resulting in substantial computational costs and inefficiencies in both memory usage and processing time. However, time series data, unlike high-dimensional images or text, often exhibit temporal pattern similarity and low-rank structures, especially in long-term horizons. By leveraging this structure, models can be guided to focus on more essential, concise temporal data, improving both accuracy and computational efficiency. In this paper, we introduce TimeBase, an ultra-lightweight network to harness the power of minimalism in LTSF. TimeBase 1) extracts core basis temporal components and 2) transforms traditional point-level forecasting into efficient segment-level forecasting, achieving optimal utilization of both data and parameters. Extensive experiments on diverse real-world datasets show that TimeBase achieves remarkable efficiency and secures competitive forecasting performance. Additionally, TimeBase can also serve as a very effective plug-and-play complexity reducer for any patch-based forecasting models.

## In ModernTSF
Default config: `configs/models/TimeBase.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Pinned source inspection: `models/TimeBase.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed from the PMLR paper. The licensed author repository is pinned as `reference-only`; its source was inspected at the pinned revision; no external source code was copied for this independent implementation.
- Equations 1–4 are represented by segmenting `X`, applying `X_basis=BasisExtract(X_his)`, applying the segment-level forecast map, and flattening/trimming the result. Equations 5–7 are represented by `G=X_basis^T X_basis` and the off-diagonal Frobenius penalty.
- `orthogonal_weight = 0.08` is a runnable point from the paper's 0.00–0.20 sweep, not a universal paper setting; dataset-specific result reproduction is outside this structural validation.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/HuangZYYW025,
  author       = {Qihe Huang and
                  Zhengyang Zhou and
                  Kuo Yang and
                  Zhongchao Yi and
                  Xu Wang and
                  Yang Wang},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {TimeBase: The Power of Minimalism in Efficient Long-term Time Series
                  Forecasting},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/huang25az.html},
  timestamp    = {Thu, 12 Feb 2026 07:51:25 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/HuangZYYW025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
