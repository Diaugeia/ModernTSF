---
name: "Koopa"
summary: "Koopa is a time series forecasting model for univariate and multivariate sequence prediction. It leverages modern Koopman theory to disentangle time-variant and time-invariant components of non-stationary time series, using a Fourier filter for decomposition and stackable Koopman Predictor blocks that advance each type of dynamics forward with learned linear operators."
paper: "https://arxiv.org/abs/2305.18803"
paper_title: "Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors"
venue: "NeurIPS 2023"
year: 2023
code: "https://github.com/thuml/Koopa"
revision: "a2e0bb77ec7c1a25e8e0579ba517ffb41358b844"
license: "MIT"
tagline: "Fourier split of invariant and variant dynamics, each advanced by a global or window-estimated Koopman operator."
tags: ["mlp", "koopman", "decomposition", "frequency", "normalization"]
composition: ["normalization=local:instance-standardization", "decomposition=local:fourier-dominant-mode-split", "temporal=local:global-and-local-koopman-operators", "channel=local:channel-mixing-measurement-function", "head=local:operator-rollout-decoder-summed-over-blocks", "loss=loss:mse"]
---
# Koopa

## Key ideas

- `FourierDynamicsSplit` keeps the top `alpha` fraction of rFFT bins by batch-average energy as the time-invariant part; the remainder is time-variant.
- `GlobalKoopmanPredictor` advances invariant latents with one learned shared transition; `LocalKoopmanPredictor` estimates a ridge-DMD operator (`estimate_operator`) from the latest `seg_len` latent states.
- `MeasurementFunction` encodes the channel vector at each step to a latent and decodes after rolling the operator forward `pred_len` steps.
- `KoopmanBlock`s are stacked residually: each subtracts its reconstructions and adds its forecast contribution.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 192, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2305.18803); title: Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors; venue/year: NeurIPS 2023 / 2023
- [codebase](https://github.com/thuml/Koopa); revision: `a2e0bb77ec7c1a25e8e0579ba517ffb41358b844`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Koopa.toml`](../../../../configs/models/Koopa.toml).

## Differences

Clean-room implementation: confirmed. Paper mapping: Fourier Filter → `FourierDynamicsSplit`; measurement function → `MeasurementFunction`; time-variant neighborhood operator → `LocalKoopmanPredictor`; invariant operator → `GlobalKoopmanPredictor`; residual hierarchy → `KoopmanBlock`. Reference-only source code was not copied. Dataset-global masks, rolling adaptation and numerical reference comparison remain outside this forecast-only rewrite.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=192` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `dynamic_dim=128`, `hidden_dim=64`, `hidden_layers=2`, `num_blocks=3`, `multistep=False`, `alpha=0.2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors
- **Venue**: NeurIPS 2023
- **Published**: 2023 (arXiv: 2023-05)
- **arXiv**: https://arxiv.org/abs/2305.18803

## Abstract
Real-world time series are characterized by intrinsic non-stationarity that poses a principal challenge for deep forecasting models. While previous models suffer from complicated series variations induced by changing temporal distribution, we tackle non-stationary time series with modern Koopman theory that fundamentally considers the underlying time-variant dynamics. Inspired by Koopman theory of portraying complex dynamical systems, we disentangle time-variant and time-invariant components from intricate non-stationary series by Fourier Filter and design Koopman Predictor to advance respective dynamics forward. Technically, we propose Koopa as a novel Koopman forecaster composed of stackable blocks that learn hierarchical dynamics. Koopa seeks measurement functions for Koopman embedding and utilizes Koopman operators as linear portraits of implicit transition. To cope with time-variant dynamics that exhibits strong locality, Koopa calculates context-aware operators in the temporal neighborhood and is able to utilize incoming ground truth to scale up forecast horizon. Besides, by integrating Koopman Predictors into deep residual structure, we ravel out the binding reconstruction loss in previous Koopman forecasters and achieve end-to-end forecasting objective optimization. Compared with the state-of-the-art model, Koopa achieves competitive performance while saving 77.3% training time and 76.0% memory.

## In TSFLab
Default config: `configs/models/Koopa.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. Paper mapping: Fourier Filter → `FourierDynamicsSplit`; measurement function → `MeasurementFunction`; time-variant neighborhood operator → `LocalKoopmanPredictor`; invariant operator → `GlobalKoopmanPredictor`; residual hierarchy → `KoopmanBlock`. Reference-only source code was not copied. Dataset-global masks, rolling adaptation and numerical reference comparison remain outside this forecast-only rewrite.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/LiuLWL23,
  author       = {Yong Liu and
                  Chenyu Li and
                  Jianmin Wang and
                  Mingsheng Long},
  editor       = {Alice Oh and
                  Tristan Naumann and
                  Amir Globerson and
                  Kate Saenko and
                  Moritz Hardt and
                  Sergey Levine},
  title        = {Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors},
  booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                  LA, USA, December 10 - 16, 2023},
  year         = {2023},
  url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html},
  timestamp    = {Sun, 29 Mar 2026 11:26:46 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/LiuLWL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
