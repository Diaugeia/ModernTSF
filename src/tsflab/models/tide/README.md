---
name: "TiDE"
summary: "TiDE (Time-series Dense Encoder) is an MLP-based encoder-decoder model for long-term time series forecasting, serving the standard time series prediction setting with optional covariate support. It encodes the historical time series together with past and future covariates using dense MLP layers, then decodes to produce future predictions — combining the simplicity and speed of linear models with the expressiveness needed for nonlinear dependencies. TiDE is 5-10x faster than comparable Transformer-based models on standard benchmarks."
paper: "https://arxiv.org/abs/2304.08424"
paper_title: "Long-term Forecasting with TiDE: Time-series Dense Encoder"
venue: "TMLR 2023"
year: 2023
code: "https://github.com/thuml/Time-Series-Library"
revision: "4e938a1767106324dd753b2a44832bf870a0252e"
license: "MIT"
---
# TiDE

TiDE (Time-series Dense Encoder) is an MLP-based encoder-decoder model for long-term time series forecasting, serving the standard time series prediction setting with optional covariate support. It encodes the historical time series together with past and future covariates using dense MLP layers, then decodes to produce future predictions — combining the simplicity and speed of linear models with the expressiveness needed for nonlinear dependencies. TiDE is 5-10x faster than comparable Transformer-based models on standard benchmarks.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2304.08424); title: Long-term Forecasting with TiDE: Time-series Dense Encoder; venue/year: TMLR 2023 / 2023
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `4e938a1767106324dd753b2a44832bf870a0252e`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TiDE.toml`](../../../../configs/models/TiDE.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independent clean-room implementation from the paper; the THUML repository
  is reference-only and no source was copied.
- The scalar temporal decoder deliberately omits LayerNorm: LayerNorm over one value makes the nonlinear branch identically zero. `decoder_output_dim` is an internal width and `time_feat_dim` describes runner markers.
- Static attributes, paper preprocessing, and numerical result reference comparison are not included.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `d_model=512`, `e_layers=2`, `d_layers=1`, `d_ff=2048`, `decoder_output_dim=7`, `time_feat_dim=6`, `dropout=0.1`, `bias=True`, `feature_encode_dim=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Long-term Forecasting with TiDE: Time-series Dense Encoder
- **Venue**: TMLR 2023
- **Published**: 2023 (arXiv: 2023-04)
- **arXiv**: https://arxiv.org/abs/2304.08424

## Abstract
Recent work has shown that simple linear models can outperform several Transformer based approaches in long term time-series forecasting. Motivated by this, we propose a Multi-layer Perceptron (MLP) based encoder-decoder model, Time-series Dense Encoder (TiDE), for long-term time-series forecasting that enjoys the simplicity and speed of linear models while also being able to handle covariates and non-linear dependencies. Theoretically, we prove that the simplest linear analogue of our model can achieve near optimal error rate for linear dynamical systems (LDS) under some assumptions. Empirically, we show that our method can match or outperform prior approaches on popular long-term time-series forecasting benchmarks while being 5-10x faster than the best Transformer based model.

## In TSFLab
Default config: `configs/models/TiDE.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independent clean-room implementation from the paper; the THUML repository
  is reference-only and no source was copied.
- The scalar temporal decoder deliberately omits LayerNorm: LayerNorm over one value makes the nonlinear branch identically zero. `decoder_output_dim` is an internal width and `time_feat_dim` describes runner markers.
- Static attributes, paper preprocessing, and numerical result reference comparison are not included.

## Citation

```bibtex
@article{DBLP:journals/tmlr/DasKLMSY23,
  author       = {Abhimanyu Das and
                  Weihao Kong and
                  Andrew Leach and
                  Shaan Mathur and
                  Rajat Sen and
                  Rose Yu},
  title        = {Long-term Forecasting with TiDE: Time-series Dense Encoder},
  journal      = {Trans. Mach. Learn. Res.},
  volume       = {2023},
  year         = {2023},
  url          = {https://arxiv.org/abs/2304.08424},
  eprinttype   = {arXiv},
  eprint       = {2304.08424},
  timestamp    = {Thu, 01 Aug 2024 15:37:25 +0200},
  biburl       = {https://dblp.org/rec/journals/tmlr/DasKLMSY23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
