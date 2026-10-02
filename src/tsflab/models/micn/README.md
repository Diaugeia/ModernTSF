---
name: "MICN"
summary: "MICN (Multi-scale Isometric Convolution Network) is a long-term time-series forecasting model presented at ICLR 2023. It adopts a multi-scale branch structure where each branch extracts local temporal features via down-sampled convolution and captures global correlations via isometric convolution, achieving linear complexity with respect to sequence length while outperforming Transformer-based methods on standard benchmarks."
paper: "https://openreview.net/references/pdf?id=u64xKhWy-T"
paper_title: "MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting"
venue: "ICLR 2023"
year: 2023
code: "https://github.com/wanghq21/MICN"
revision: "370c69b841d72246556ca05dd23163c560c22b5a"
license: "NOASSERTION"
tagline: "Multi-scale branches of strided local conv, isometric global conv and transposed restore on the seasonal part."
tags: ["cnn", "decomposition", "multi-scale", "channel-mixing", "long-horizon"]
composition: ["normalization=none", "decomposition=component:series_decomposition", "temporal=local:multi-scale-isometric-convolution-branches", "channel=local:channel-mixing-linear-embedding", "head=local:linear-history-to-horizon-seasonal-plus-trend", "loss=loss:mse"]
---
# MICN

## Key ideas

- `MultiScaleDecomposition` averages several moving-average decompositions (`series_decomposition`) into one seasonal and one trend component.
- Each `IsometricConvolutionBranch` downsamples with a strided depthwise conv (local pattern), adds a global isometric convolution over the shortened sequence, and restores length with a transposed conv; `MICLayer` merges the scales.
- The seasonal part is refined with a residual and mapped by a linear history-to-horizon layer; the trend part gets its own linear map and the two forecasts are summed.
- Calendar embedding is omitted; the input embedding is a linear map over channels.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/references/pdf?id=u64xKhWy-T); title: MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting; venue/year: ICLR 2023 / 2023
- [codebase](https://github.com/wanghq21/MICN); revision: `370c69b841d72246556ca05dd23163c560c22b5a`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MICN.toml`](../../../../configs/models/MICN.toml).

## Differences

Clean-room implementation: confirmed. Multi-scale decomposition and downsample/isometric/restore branches were independently implemented from the paper; reference-only source was not copied. Calendar embedding is intentionally omitted.

## Shared components

- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `d_layers=1`, `dropout=0.05`, `conv_kernel=[12, 16]`
<!-- model-card:canonical:end -->

## Paper
- **Title**: MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting
- **Venue**: ICLR 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract
Recently, Transformer-based methods have achieved surprising performance in the field of long-term series forecasting, but the attention mechanism for computing global correlations entails high complexity. And they do not allow for targeted modeling of local features as CNN structures do. To solve the above problems, we propose to combine local features and global correlations to capture the overall view of time series (e.g., fluctuations, trends). To fully exploit the underlying information in the time series, a multi-scale branch structure is adopted to model different potential patterns separately. Each pattern is extracted with down-sampled convolution and isometric convolution for local features and global correlations, respectively. In addition to being more effective, our proposed method, termed as Multi-scale Isometric Convolution Network (MICN), is more efficient with linear complexity about the sequence length with suitable convolution kernels. Our experiments on six benchmark datasets show that compared with state-of-the-art methods, MICN yields 17.2% and 21.6% relative improvements for multivariate and univariate time series, respectively.

## In TSFLab
Default config: `configs/models/MICN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. Multi-scale decomposition and downsample/isometric/restore branches were independently implemented from the paper; reference-only source was not copied. Calendar embedding is intentionally omitted.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/Wang0HWCX23,
  author       = {Huiqiang Wang and
                  Jian Peng and
                  Feihu Huang and
                  Jince Wang and
                  Junhui Chen and
                  Yifei Xiao},
  title        = {{MICN:} Multi-scale Local and Global Context Modeling for Long-term
                  Series Forecasting},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023},
  url          = {https://openreview.net/forum?id=zt53IDUR1U},
  timestamp    = {Mon, 21 Oct 2024 15:07:23 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/Wang0HWCX23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
