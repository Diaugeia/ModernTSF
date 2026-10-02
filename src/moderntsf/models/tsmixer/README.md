---
name: "TSMixer"
summary: "TSMixer is an MLP-Mixer-style model for multivariate time-series forecasting that alternates mixing operations along the time dimension and the feature (channel) dimension. By stacking MLP blocks that operate on transposed views of the input, it efficiently extracts both temporal dynamics and cross-variate correlations without any attention mechanism, achieving competitive accuracy while remaining easy to implement."
paper: "https://arxiv.org/abs/2303.06053"
paper_title: "TSMixer: An All-MLP Architecture for Time Series Forecasting"
venue: "TMLR 2023"
year: 2023
code: "https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tsmixer"
revision: "5b09c22d73a9d35eb6c5d2a99b95677a45053466"
license: "Apache-2.0"
---
# TSMixer

TSMixer is an MLP-Mixer-style model for multivariate time-series forecasting that alternates mixing operations along the time dimension and the feature (channel) dimension. By stacking MLP blocks that operate on transposed views of the input, it efficiently extracts both temporal dynamics and cross-variate correlations without any attention mechanism, achieving competitive accuracy while remaining easy to implement.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2303.06053); title: TSMixer: An All-MLP Architecture for Time Series Forecasting; venue/year: TMLR 2023 / 2023
- [codebase](https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tsmixer); revision: `5b09c22d73a9d35eb6c5d2a99b95677a45053466`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TSMixer.toml`](../../../../configs/models/TSMixer.toml).

## Differences

Clean-room implementation: confirmed.

Clean-room implementation confirmed from paper Appendix B.3.1--B.3.2; the reference-only Google Research source was not copied. Each block applies pre-normalized residual time mixing and a two-layer residual feature mixer, followed by the shared temporal projection. ModernTSF implements only the basic historical-target model, uses sample-wise two-dimensional LayerNorm and GELU, and omits the auxiliary/static-feature extension, dataset-global preprocessing, benchmark-specific BatchNorm, and training stack.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`mixer_block`](../_components/mixer_block/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `e_layers=2`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TSMixer: An All-MLP Architecture for Time Series Forecasting
- **Venue**: TMLR 2023
- **Published**: 2023 (arXiv: 2023-03)
- **arXiv**: https://arxiv.org/abs/2303.06053

## Abstract
Real-world time-series datasets are often multivariate with complex dynamics. To capture this complexity, high capacity architectures like recurrent- or attention-based sequential deep learning models have become popular. However, recent work demonstrates that simple univariate linear models can outperform such deep learning models on several commonly used academic benchmarks. Extending them, in this paper, we investigate the capabilities of linear models for time-series forecasting and present Time-Series Mixer (TSMixer), a novel architecture designed by stacking multi-layer perceptrons (MLPs). TSMixer is based on mixing operations along both the time and feature dimensions to extract information efficiently. On popular academic benchmarks, the simple-to-implement TSMixer is comparable to specialized state-of-the-art models that leverage the inductive biases of specific benchmarks. On the challenging and large scale M5 benchmark, a real-world retail dataset, TSMixer demonstrates superior performance compared to the state-of-the-art alternatives. Our results underline the importance of efficiently utilizing cross-variate and auxiliary information for improving the performance of time series forecasting. We present various analyses to shed light into the capabilities of TSMixer. The design paradigms utilized in TSMixer are expected to open new horizons for deep learning-based time series forecasting.

## In ModernTSF
Default config: `configs/models/TSMixer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed.

Clean-room implementation confirmed from paper Appendix B.3.1--B.3.2; the reference-only Google Research source was not copied. Each block applies pre-normalized residual time mixing and a two-layer residual feature mixer, followed by the shared temporal projection. ModernTSF implements only the basic historical-target model, uses sample-wise two-dimensional LayerNorm and GELU, and omits the auxiliary/static-feature extension, dataset-global preprocessing, benchmark-specific BatchNorm, and training stack.

## Citation

```bibtex
@misc{chen2023tsmixer,
  author        = {Si-An Chen and
                  Chun-Liang Li and
                  Nate Yoder and
                  Sercan O. Arik and
                  Tomas Pfister},
  title         = {TSMixer: An All-MLP Architecture for Time Series Forecasting},
  year          = {2023},
  eprint        = {2303.06053},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2303.06053}
}
```
