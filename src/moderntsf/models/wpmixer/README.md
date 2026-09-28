---
name: "WPMixer"
summary: "WPMixer (Wavelet Patch Mixer) is an MLP-based model for long-term time series forecasting in the standard time series setting. It combines three complementary techniques: multi-resolution wavelet decomposition to extract information in both frequency and time domains, patching to capture extended historical context and local patterns with an extended look-back window, and MLP mixing layers to incorporate global temporal information — significantly outperforming state-of-the-art MLP-based and Transformer-based models in a computationally efficient manner."
paper: "https://arxiv.org/abs/2412.17176"
paper_title: "WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
code: "https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer"
revision: "74104c9dddd54d279eb8323f48934b4fd75fcae7"
license: "MIT"
---
# WPMixer

WPMixer (Wavelet Patch Mixer) is an MLP-based model for long-term time series forecasting in the standard time series setting. It combines three complementary techniques: multi-resolution wavelet decomposition to extract information in both frequency and time domains, patching to capture extended historical context and local patterns with an extended look-back window, and MLP mixing layers to incorporate global temporal information — significantly outperforming state-of-the-art MLP-based and Transformer-based models in a computationally efficient manner.

<!-- model-card:canonical:start -->
## Method overview

WPMixer (Wavelet Patch Mixer) is an MLP-based model for long-term time series forecasting in the standard time series setting.

## Core architecture

It combines three complementary techniques: multi-resolution wavelet decomposition to extract information in both frequency and time domains, patching to capture extended historical context and local patterns with an extended look-back window, and MLP mixing layers to incorporate global temporal information — significantly outperforming state-of-the-art MLP-based and Transformer-based models in a computationally efficient manner.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2412.17176); title: WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting; venue/year: AAAI 2025 / 2025
- [codebase](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer); revision: `74104c9dddd54d279eb8323f48934b4fd75fcae7`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/WPMixer.toml`](../../../../configs/models/WPMixer.toml).

## Differences

Clean-room implementation: confirmed. The implementation was derived independently from the paper's orthogonal multi-resolution analysis, per-resolution patching, token/feature MLP mixing, and learned resolution fusion; reference source code was not copied or reused. Fixed mathematical Haar/db1/db2 analysis filters are local, and branch forecasts are fused directly rather than reconstructed by an external inverse-wavelet package.

Component audit (`mixer_block`): `ResolutionMixer`'s token/feature mixer normalizes the residual sum after each MLP (post-norm) instead of normalizing the input before it (pre-norm), and its patch-unfold-plus-embedding step has no counterpart in `mixer_block`; it stays model-local.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `dropout=0.1`, `tfactor=5`, `dfactor=5`, `wavelet='db2'`, `level=1`, `patch_len=16`, `stride=8`, `no_decomposition=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2412.17176

## Abstract
Time series forecasting is crucial for various applications, such as weather forecasting, power load forecasting, and financial analysis. In recent studies, MLP-mixer models for time series forecasting have been shown as a promising alternative to transformer-based models. However, the performance of these models is still yet to reach its potential. In this paper, we propose Wavelet Patch Mixer (WPMixer), a novel MLP-based model, for long-term time series forecasting, which leverages the benefits of patching, multi-resolution wavelet decomposition, and mixing. Our model is based on three key components: (i) multi-resolution wavelet decomposition, (ii) patching and embedding, and (iii) MLP mixing. Multi-resolution wavelet decomposition efficiently extracts information in both the frequency and time domains. Patching allows the model to capture an extended history with a look-back window and enhances capturing local information while MLP mixing incorporates global information. Our model significantly outperforms state-of-the-art MLP-based and transformer-based models for long-term time series forecasting in a computationally efficient way, demonstrating its efficacy and potential for practical applications.

## In ModernTSF
Default config: `configs/models/WPMixer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. The implementation was derived independently from the paper's orthogonal multi-resolution analysis, per-resolution patching, token/feature MLP mixing, and learned resolution fusion; reference source code was not copied or reused. Fixed mathematical Haar/db1/db2 analysis filters are local, and branch forecasts are fused directly rather than reconstructed by an external inverse-wavelet package.

Component audit (`mixer_block`): `ResolutionMixer`'s token/feature mixer normalizes the residual sum after each MLP (post-norm) instead of normalizing the input before it (pre-norm), and its patch-unfold-plus-embedding step has no counterpart in `mixer_block`; it stays model-local.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/MuradAY25,
  author       = {Md Mahmuddun Nabi Murad and
                  Mehmet Aktukmak and
                  Yasin Yilmaz},
  editor       = {Toby Walsh and
                  Julie Shah and
                  Zico Kolter},
  title        = {WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series
                  Forecasting},
  booktitle    = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
                  Conference on Innovative Applications of Artificial Intelligence,
                  Fifteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages        = {19581--19588},
  publisher    = {{AAAI} Press},
  year         = {2025},
  url          = {https://doi.org/10.1609/aaai.v39i18.34156},
  doi          = {10.1609/AAAI.V39I18.34156},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/MuradAY25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
