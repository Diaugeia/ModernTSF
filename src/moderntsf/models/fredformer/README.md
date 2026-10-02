---
name: "Fredformer"
summary: "Fredformer is a Transformer-based model for multivariate time series forecasting that addresses the frequency bias problem, where attention mechanisms tend to over-weight low-frequency (high-energy) components while ignoring high-frequency features. It applies frequency-equalised attention across patch-level frequency bands to produce more balanced and accurate forecasts."
paper: "https://arxiv.org/abs/2406.09009"
paper_title: "Fredformer: Frequency Debiased Transformer for Time Series Forecasting"
venue: "KDD 2024"
year: 2024
code: "https://github.com/chenzRG/Fredformer"
revision: "fa64775ea1012e313cbe30fe2c9b7e493a798aae"
license: "NOASSERTION"
---
# Fredformer

Fredformer is a Transformer-based model for multivariate time series forecasting that addresses the frequency bias problem, where attention mechanisms tend to over-weight low-frequency (high-energy) components while ignoring high-frequency features. It applies frequency-equalised attention across patch-level frequency bands to produce more balanced and accurate forecasts.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2406.09009); title: Fredformer: Frequency Debiased Transformer for Time Series Forecasting; venue/year: KDD 2024 / 2024
- [codebase](https://github.com/chenzRG/Fredformer); revision: `fa64775ea1012e313cbe30fe2c9b7e493a798aae`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Fredformer.toml`](../../../../configs/models/Fredformer.toml).

## Differences

Clean-room implementation: confirmed. It was derived from the paper's frequency-equalization and band-local attention design; source from the unlicensed reference repository was not copied or reused.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `band_width=16`, `model_width=48`, `depth=2`, `heads=6`, `feedforward=128`, `dropout=0.2`, `revin=True`, `affine=True`, `subtract_last=False`, `head_dropout=0.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Fredformer: Frequency Debiased Transformer for Time Series Forecasting
- **Venue**: KDD 2024
- **Published**: 2024 (arXiv: 2024-06)
- **arXiv**: https://arxiv.org/abs/2406.09009

## Abstract
The Transformer model has shown leading performance in time series forecasting. Nevertheless, in some complex scenarios, it tends to learn low-frequency features in the data and overlook high-frequency features, showing a frequency bias. This bias prevents the model from accurately capturing important high-frequency data features. In this paper, we undertook empirical analyses to understand this bias and discovered that frequency bias results from the model disproportionately focusing on frequency features with higher energy. Based on our analysis, we formulate this bias and propose Fredformer, a Transformer-based framework designed to mitigate frequency bias by learning features equally across different frequency bands. This approach prevents the model from overlooking lower amplitude features important for accurate forecasting. Extensive experiments show the effectiveness of our proposed approach, which can outperform other baselines in different real-world time-series datasets. Furthermore, we introduce a lightweight variant of the Fredformer with an attention matrix approximation, which achieves comparable performance but with much fewer parameters and lower computation costs.

## In ModernTSF
Default config: `configs/models/Fredformer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. It was derived from the paper's frequency-equalization and band-local attention design; source from the unlicensed reference repository was not copied or reused.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/Piao0MMS24,
  author       = {Xihao Piao and
                  Zheng Chen and
                  Taichi Murayama and
                  Yasuko Matsubara and
                  Yasushi Sakurai},
  editor       = {Ricardo Baeza{-}Yates and
                  Francesco Bonchi},
  title        = {Fredformer: Frequency Debiased Transformer for Time Series Forecasting},
  booktitle    = {Proceedings of the 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2024, Barcelona, Spain, August 25-29, 2024},
  pages        = {2400--2410},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3637528.3671928},
  doi          = {10.1145/3637528.3671928},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/Piao0MMS24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
