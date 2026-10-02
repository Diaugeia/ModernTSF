---
name: "Amplifier"
summary: "Amplifier is a multivariate/univariate time-series forecasting model that addresses the common failure mode of existing models that overlook low-energy frequency components. It introduces an energy amplification technique — comprising an amplification block and a restoration block — integrated with a seasonal-trend decomposition backbone, and further augments it with a semi-channel interaction temporal relationship enhancement block that exploits both commonality and specificity across channels."
paper: "https://arxiv.org/abs/2501.17216"
paper_title: "Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
code: "https://github.com/aikunyi/amplifier"
revision: "6cc089312254a0eeda7767342f690fd4536a1758"
license: "Apache-2.0"
---
# Amplifier

Amplifier is a multivariate/univariate time-series forecasting model that addresses the common failure mode of existing models that overlook low-energy frequency components. It introduces an energy amplification technique — comprising an amplification block and a restoration block — integrated with a seasonal-trend decomposition backbone, and further augments it with a semi-channel interaction temporal relationship enhancement block that exploits both commonality and specificity across channels.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2501.17216); title: Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting; venue/year: AAAI 2025 / 2025
- [codebase](https://github.com/aikunyi/amplifier); revision: `6cc089312254a0eeda7767342f690fd4536a1758`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Amplifier.toml`](../../../../configs/models/Amplifier.toml).

## Differences

**Clean-room implementation: confirmed.** The linked Apache-2.0 author
repository is `reference-only`; its source was not copied. The local design maps
paper Eqs. 5--9 to spectrum flip, amplification, complex horizon projection, and
restoration; Eqs. 10--11 to common/specific SCI paths; and Eqs. 12--13 to shared
seasonal-trend decomposition and two forecast MLPs. It uses one-sided real FFTs
and separate real/imaginary restoration maps, exposes forecasting only, and
makes no checkpoint, training-recipe, or published-metric reference comparison claim.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `hidden_size=128`, `sci=True`, `moving_average=25`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2025-01)
- **arXiv**: https://arxiv.org/abs/2501.17216

## Abstract
We propose an energy amplification technique to address the issue that existing models easily overlook low-energy components in time series forecasting. This technique comprises an energy amplification block and an energy restoration block. The energy amplification block enhances the energy of low-energy components to improve the model's learning efficiency for these components, while the energy restoration block returns the energy to its original level. Moreover, considering that the energy-amplified data typically displays two distinct energy peaks in the frequency spectrum, we integrate the energy amplification technique with a seasonal-trend forecaster to model the temporal relationships of these two peaks independently, serving as the backbone for our proposed model, Amplifier. Additionally, we propose a semi-channel interaction temporal relationship enhancement block for Amplifier, which enhances the model's ability to capture temporal relationships from the perspective of the commonality and specificity of each channel in the data. Extensive experiments on eight time series forecasting benchmarks consistently demonstrate our model's superiority in both effectiveness and efficiency compared to state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/Amplifier.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Source and verification

**Clean-room implementation: confirmed.** The linked Apache-2.0 author
repository is `reference-only`; its source was not copied. The local design maps
paper Eqs. 5--9 to spectrum flip, amplification, complex horizon projection, and
restoration; Eqs. 10--11 to common/specific SCI paths; and Eqs. 12--13 to shared
seasonal-trend decomposition and two forecast MLPs. It uses one-sided real FFTs
and separate real/imaginary restoration maps, exposes forecasting only, and
makes no checkpoint, training-recipe, or published-metric reference comparison claim.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/Fei000N25,
  author       = {Jingru Fei and
                  Kun Yi and
                  Wei Fan and
                  Qi Zhang and
                  Zhendong Niu},
  editor       = {Toby Walsh and
                  Julie Shah and
                  Zico Kolter},
  title        = {Amplifier: Bringing Attention to Neglected Low-Energy Components in
                  Time Series Forecasting},
  booktitle    = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
                  Conference on Innovative Applications of Artificial Intelligence,
                  Fifteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages        = {11645--11653},
  publisher    = {{AAAI} Press},
  year         = {2025},
  url          = {https://doi.org/10.1609/aaai.v39i11.33267},
  doi          = {10.1609/AAAI.V39I11.33267},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/Fei000N25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
