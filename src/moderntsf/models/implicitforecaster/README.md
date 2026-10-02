---
name: "ImplicitForecaster"
summary: "ImplicitForecaster (IF) is a time-series forecasting decoding module accepted at NeurIPS 2025. Rather than generating long-horizon forecasts by independently predicting each time point, it implicitly decomposes the target sequence into constituent waves parameterized by frequency, amplitude, and phase, capturing both long-term and short-term dynamics in a holistic manner and consistently boosting mainstream backbone models."
paper: "https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html"
paper_title: "Towards Accurate Time Series Forecasting via Implicit Decoding"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/rakuyorain/Implicit-Forecaster"
revision: "e3e7f77fb0489c2b5e58eac990096668b8b0aff4"
license: "Apache-2.0"
---
# ImplicitForecaster

ImplicitForecaster (IF) is a time-series forecasting decoding module accepted at NeurIPS 2025. Rather than generating long-horizon forecasts by independently predicting each time point, it implicitly decomposes the target sequence into constituent waves parameterized by frequency, amplitude, and phase, capturing both long-term and short-term dynamics in a holistic manner and consistently boosting mainstream backbone models.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html); title: Towards Accurate Time Series Forecasting via Implicit Decoding; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/rakuyorain/Implicit-Forecaster); revision: `e3e7f77fb0489c2b5e58eac990096668b8b0aff4`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ImplicitForecaster.toml`](../../../../configs/models/ImplicitForecaster.toml).

## Differences

Pinned source inspection: `models/IFT.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. `spectral_parameters()` follows equations
(2), (5), and (6): the history spectrum augments a channel-separated temporal
encoder, AHead predicts non-negative amplitudes, PHead predicts continuous
sine/cosine phase coordinates, and `irfft` composes the frequency pool. The linked
repository is reference-only; its source was inspected at the pinned revision; no external source code was copied.

The paper presents IF as a decoder that can replace the head of several large
backbones. This standalone rewrite uses a compact temporal MLP encoder, a pool of
`frequency_pool=192` samples, and one direct crop; it does not reproduce every
paper backbone, training recipe, or benchmark-specific frequency-pool search.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `frequency_pool=192`, `dropout=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Towards Accurate Time Series Forecasting via Implicit Decoding
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **Proceedings**: https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html

## Abstract
Recent booming time series models have demonstrated remarkable forecasting performance. However, these methods often place greater focus on more effectively modelling the historical series, largely neglecting the forecasting phase, which generates long-term forecasts by separately predicting multiple time points. Given that real-world time series typically consist of various long short-term dynamics, independent predictions over individual time points may fail to express complex underlying patterns and can lead to a lack of global views. To address these issues, this work explores new perspectives from the forecasting phase and proposes a novel Implicit Forecaster (IF) as an additional decoding module. Inspired by decomposition forecasting, IF adopts a more nuanced approach by implicitly predicting constituent waves represented by their frequency, amplitude, and phase, thereby accurately forming the time series. Extensive experimental results from multiple real-world datasets show that IF can consistently boost mainstream time series models, achieving state-of-the-art forecasting performance.

## Source and verification

Pinned source inspection: `models/IFT.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. `spectral_parameters()` follows equations
(2), (5), and (6): the history spectrum augments a channel-separated temporal
encoder, AHead predicts non-negative amplitudes, PHead predicts continuous
sine/cosine phase coordinates, and `irfft` composes the frequency pool. The linked
repository is reference-only; its source was inspected at the pinned revision; no external source code was copied.

The paper presents IF as a decoder that can replace the head of several large
backbones. This standalone rewrite uses a compact temporal MLP encoder, a pool of
`frequency_pool=192` samples, and one direct crop; it does not reproduce every
paper backbone, training recipe, or benchmark-specific frequency-pool search.

## In ModernTSF
Default config: `configs/models/ImplicitForecaster.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{li2025implicitforecasting,
  author       = {Xinyu Li and Yuchen Luo and Hao Wang and Haoxuan Li and Liuhua Peng and Feng Liu and Yandong Guo and Kun Zhang and Mingming Gong},
  title        = {Towards Accurate Time Series Forecasting via Implicit Decoding},
  booktitle    = {Advances in Neural Information Processing Systems},
  volume       = {38},
  year         = {2025},
  url          = {https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html}
}
```
