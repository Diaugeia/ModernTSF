---
name: "FITS"
summary: "FITS (Frequency Interpolation Time Series analysis) is a lightweight time series forecasting model that operates entirely in the complex frequency domain. Instead of processing raw time-domain sequences, FITS applies rFFT to compress the input, performs low-pass filtering to discard high-frequency noise, and uses frequency-domain interpolation to map the compressed representation to the target prediction length, enabling competitive forecasting performance with only approximately 10k parameters — small enough for edge-device deployment."
paper: "https://arxiv.org/abs/2307.03756"
paper_title: "FITS: Modeling Time Series with 10k Parameters"
venue: "ICLR 2024"
year: 2024
code: "https://github.com/VEWOXIC/FITS"
revision: "d040bb015b6299da26d879b90dd19c80fb72c160"
license: "Apache-2.0"
tagline: "Low-pass rFFT, learned complex frequency interpolation to the extended length, then irFFT; roughly 10k parameters."
tags: ["linear", "frequency", "lightweight", "channel-independent", "baseline"]
composition: ["normalization=local:instance-standardization", "decomposition=none", "temporal=local:complex-frequency-interpolation", "channel=local:channel-independent-shared-weights", "head=local:zero-padded-irfft-extension", "loss=loss:mse"]
---
# FITS

## Key ideas

- Keeps only the first `cut_freq` rFFT bins of the standardized window (low-pass filter).
- `ComplexFrequencyInterpolation` is one complex affine map from retained input bins to the bins of the `seq_len + pred_len` spectrum, shared across channels unless `individual=True`.
- Zero-pads the spectrum, applies irFFT over the full length, rescales by `(seq_len + pred_len) / seq_len`, and keeps the last `pred_len` steps.
- Normalization is a local detached mean/std centering restored on the output; there is no `revin` or decomposition.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2307.03756); title: FITS: Modeling Time Series with 10k Parameters; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/VEWOXIC/FITS); revision: `d040bb015b6299da26d879b90dd19c80fb72c160`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/FITS.toml`](../../../../configs/models/FITS.toml).

## Differences

**Paper-driven local implementation.** The local design performs reversible instance centering/scaling, rFFT projection, low-pass truncation, learned complex frequency interpolation, zero padding, irFFT reconstruction, length-ratio energy compensation, and inverse normalization. It returns only the forecast horizon; paper backcast/reconstruction supervision and anomaly detection are not claimed. The external repository is reference-only and no source file was copied or adapted.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `individual=False`, `cut_freq=24`
<!-- model-card:canonical:end -->

## Paper
- **Title**: FITS: Modeling Time Series with 10k Parameters
- **Venue**: ICLR 2024 (Spotlight)
- **Published**: 2024 (arXiv: 2023-07)
- **arXiv**: https://arxiv.org/abs/2307.03756

## Abstract
In this paper, we introduce FITS, a lightweight yet powerful model for time series analysis. Unlike existing models that directly process raw time-domain data, FITS operates on the principle that time series can be manipulated through interpolation in the complex frequency domain. By discarding high-frequency components with negligible impact on time series data, FITS achieves performance comparable to state-of-the-art models for time series forecasting and anomaly detection tasks, while having a remarkably compact size of only approximately $10k$ parameters. Such a lightweight model can be easily trained and deployed in edge devices, creating opportunities for various applications.

## In TSFLab
Default config: `configs/models/FITS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

**Paper-driven local implementation.** The local design performs reversible instance centering/scaling, rFFT projection, low-pass truncation, learned complex frequency interpolation, zero padding, irFFT reconstruction, length-ratio energy compensation, and inverse normalization. It returns only the forecast horizon; paper backcast/reconstruction supervision and anomaly detection are not claimed. The external repository is reference-only and no source file was copied or adapted.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/XuZ024,
  author       = {Zhijian Xu and
                  Ailing Zeng and
                  Qiang Xu},
  title        = {{FITS:} Modeling Time Series with 10k Parameters},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=bWcnvZ3qMb},
  timestamp    = {Mon, 29 Jul 2024 17:17:48 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/XuZ024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
