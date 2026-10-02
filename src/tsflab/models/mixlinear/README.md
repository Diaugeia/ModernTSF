---
name: "MixLinear"
summary: "MixLinear is an ultra-lightweight multivariate time-series forecasting model for the standard time-series forecasting setting. It mixes time-domain linear projections (both intra-segment and inter-segment) with frequency-domain linear projections over a low-dimensional latent space, reducing the parameter scale of the core linear layers from O(n²) to O(n) while retaining competitive accuracy — making it well suited for resource-constrained deployment."
paper: "https://arxiv.org/abs/2410.02081"
paper_title: "MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/aitianma/MixLinear"
revision: "42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106"
license: "NOASSERTION"
---
# MixLinear

MixLinear is an ultra-lightweight multivariate time-series forecasting model for the standard time-series forecasting setting. It mixes time-domain linear projections (both intra-segment and inter-segment) with frequency-domain linear projections over a low-dimensional latent space, reducing the parameter scale of the core linear layers from O(n²) to O(n) while retaining competitive accuracy — making it well suited for resource-constrained deployment.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2410.02081); title: MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/aitianma/MixLinear); revision: `42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MixLinear.toml`](../../../../configs/models/MixLinear.toml).

## Differences

Pinned source inspection: `models/MixLinear.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from paper equations (1)--(5); the unlicensed reference repository was inspected at the pinned revision; no external source code was copied. The local model adds a segment-domain path (downsampling, local segment encoding, cross-segment mixing, and reconstruction) to the complex low-rank spectral operator `U(VF)`. TSFLab uses fixed average downsampling, a symmetric local encoder/decoder, linear interpolation to the requested horizon, and per-series centering; these disclosed choices replace unspecified adaptive/reconstruction details and do not claim the paper's exact 0.1K parameter count or benchmark reference comparison.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `downsample=4`, `segments=4`, `hidden_rank=2`, `spectral_rank=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2024-10)
- **arXiv**: https://arxiv.org/abs/2410.02081

## Abstract
Recently, there has been a growing interest in Long-term Time Series Forecasting (LTSF), which involves predicting long-term future values by analyzing a large amount of historical time-series data to identify patterns and trends. There exist significant challenges in LTSF due to its complex temporal dependencies and high computational demands. Although Transformer-based models offer high forecasting accuracy, they are often too compute-intensive to be deployed on devices with hardware constraints. On the other hand, the linear models aim to reduce the computational overhead by employing either decomposition methods in the time domain or compact representations in the frequency domain. In this paper, we propose MixLinear, an ultra-lightweight multivariate time series forecasting model specifically designed for resource-constrained devices. MixLinear effectively captures both temporal and frequency domain features by modeling intra-segment and inter-segment variations in the time domain and extracting frequency variations from a low-dimensional latent space in the frequency domain. By reducing the parameter scale of a downsampled n-length input/output one-layer linear model from O(n²) to O(n), MixLinear achieves efficient computation without sacrificing accuracy. Extensive evaluations with four benchmark datasets show that MixLinear attains forecasting performance comparable to, or surpassing, state-of-the-art models with significantly fewer parameters (0.1K), which makes it well-suited for deployment on devices with limited computational capacity.

## In TSFLab
Default config: `configs/models/MixLinear.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Pinned source inspection: `models/MixLinear.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from paper equations (1)--(5); the unlicensed reference repository was inspected at the pinned revision; no external source code was copied. The local model adds a segment-domain path (downsampling, local segment encoding, cross-segment mixing, and reconstruction) to the complex low-rank spectral operator `U(VF)`. TSFLab uses fixed average downsampling, a symmetric local encoder/decoder, linear interpolation to the requested horizon, and per-series centering; these disclosed choices replace unspecified adaptive/reconstruction details and do not claim the paper's exact 0.1K parameter count or benchmark reference comparison.

## Citation

```bibtex
@misc{ma2024mixlinear,
  author        = {Aitian Ma and
                  Dongsheng Luo and
                  Mo Sha},
  title         = {MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters},
  year          = {2024},
  eprint        = {2410.02081},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2410.02081}
}
```
