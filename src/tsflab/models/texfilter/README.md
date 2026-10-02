---
name: "TexFilter"
summary: "TexFilter is the contextual shaping filter variant proposed within the FilterNet framework, targeting the standard univariate and multivariate time-series forecasting setting. It applies a learnable frequency filter in the Fourier domain — first embedding the input, computing an FFT, multiplying by a complex-valued learned weight (the \"texture\" filter) that mixes real and imaginary parts via ReLU-activated bilinear interactions, then inverting back to the time domain — to selectively pass or attenuate frequency components while preserving full-spectrum information."
paper: "https://arxiv.org/abs/2411.01623"
paper_title: "FilterNet: Harnessing Frequency Filters for Time Series Forecasting"
venue: "NeurIPS 2024"
year: 2024
code: "https://github.com/aikunyi/FilterNet"
revision: "cdb321c4e338e0c07b45cee92f54b3c5bd5a809e"
license: "Apache-2.0"
---
# TexFilter

TexFilter is the contextual shaping filter variant proposed within the FilterNet framework, targeting the standard univariate and multivariate time-series forecasting setting. It applies a learnable frequency filter in the Fourier domain — first embedding the input, computing an FFT, multiplying by a complex-valued learned weight (the "texture" filter) that mixes real and imaginary parts via ReLU-activated bilinear interactions, then inverting back to the time domain — to selectively pass or attenuate frequency components while preserving full-spectrum information.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2411.01623); title: FilterNet: Harnessing Frequency Filters for Time Series Forecasting; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/aikunyi/FilterNet); revision: `cdb321c4e338e0c07b45cee92f54b3c5bd5a809e`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TexFilter.toml`](../../../../configs/models/TexFilter.toml).

## Differences

**Paper-driven local implementation.** The contextual kernel follows Equation
(9): a complex linear map embeds the input spectrum, complex learnable factors
generate an input-dependent response, and the response filters the embedded
spectrum before inverse FFT and forecasting. TSFLab reuses canonical RevIN.
The external repository is reference-only; no source file was copied or adapted.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `embed_size=128`, `hidden_size=256`, `dropout=0.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: FilterNet: Harnessing Frequency Filters for Time Series Forecasting
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-11)
- **arXiv**: https://arxiv.org/abs/2411.01623

## Abstract
Given the ubiquitous presence of time series data across various domains, precise forecasting of time series holds significant importance and finds widespread real-world applications such as energy, weather, healthcare, etc. While numerous forecasters have been proposed using different network architectures, the Transformer-based models have state-of-the-art performance in time series forecasting. However, forecasters based on Transformers are still suffering from vulnerability to high-frequency signals, efficiency in computation, and bottleneck in full-spectrum utilization, which essentially are the cornerstones for accurately predicting time series with thousands of points. In this paper, we explore a novel perspective of enlightening signal processing for deep time series forecasting. Inspired by the filtering process, we introduce one simple yet effective network, namely FilterNet, built upon our proposed learnable frequency filters to extract key informative temporal patterns by selectively passing or attenuating certain components of time series signals. Concretely, we propose two kinds of learnable filters in the FilterNet: (i) Plain shaping filter, that adopts a universal frequency kernel for signal filtering and temporal modeling; (ii) Contextual shaping filter, that utilizes filtered frequencies examined in terms of its compatibility with input signals for dependency learning. Equipped with the two filters, FilterNet can approximately surrogate the linear and attention mappings widely adopted in time series literature, while enjoying superb abilities in handling high-frequency noises and utilizing the whole frequency spectrum that is beneficial for forecasting. Finally, we conduct extensive experiments on eight time series forecasting benchmarks, and experimental results have demonstrated our superior performance in terms of both effectiveness and efficiency compared with state-of-the-art methods.

## In TSFLab
Default config: `configs/models/TexFilter.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

**Paper-driven local implementation.** The contextual kernel follows Equation
(9): a complex linear map embeds the input spectrum, complex learnable factors
generate an input-dependent response, and the response filters the embedded
spectrum before inverse FFT and forecasting. TSFLab reuses canonical RevIN.
The external repository is reference-only; no source file was copied or adapted.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/0001FZHHL024,
  author       = {Kun Yi and
                  Jingru Fei and
                  Qi Zhang and
                  Hui He and
                  Shufeng Hao and
                  Defu Lian and
                  Wei Fan},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {FilterNet: Harnessing Frequency Filters for Time Series Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/6323d96f79d5d49e0d3fc88835c082cd-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/0001FZHHL024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
