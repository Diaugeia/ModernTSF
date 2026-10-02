---
name: "DeepAR"
summary: "DeepAR is an autoregressive recurrent neural network designed for probabilistic time-series forecasting. It trains a single global LSTM-based model over many related time series and outputs a learned probability distribution over the forecast horizon rather than a point prediction, making it well-suited to the standard univariate and multivariate time-series forecasting setting."
paper: "https://arxiv.org/abs/1704.04110"
paper_title: "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
venue: "International Journal of Forecasting 2020"
year: 2020
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "79641b1c75246ab2d8c53bb52f2ac72588be0cdc"
license: "Apache-2.0"
tagline: "Autoregressive LSTM emitting Gaussian parameters per step, fed back its own mean at inference; shared across series."
tags: ["rnn", "probabilistic", "autoregressive", "covariates", "channel-independent"]
composition: ["normalization=none", "decomposition=none", "temporal=local:autoregressive-lstm", "channel=local:channel-independent-shared-weights", "head=component:gaussian_parameter_head", "loss=loss:nll_gaussian"]
---
# DeepAR

## Key ideas

- A single global LSTM (`recurrent`) is shared by all channels, each treated as its own series, and embeds scalar values plus optional covariates from time marks.
- `gaussian_parameter_head` outputs location and positive scale at each step; the model returns `[batch, pred_len, channels, 2]` (output type `distribution`).
- Decoding is autoregressive: after encoding history, the likelihood mean is fed back as the next input (deterministic, no ancestral sampling).
- Trained with Gaussian negative log-likelihood (`nll_gaussian`).

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, parameters]` distribution parameters.

## Paper and code

- [paper](https://arxiv.org/abs/1704.04110); title: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks; venue/year: International Journal of Forecasting 2020 / 2020
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `79641b1c75246ab2d8c53bb52f2ac72588be0cdc`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DeepAR.toml`](../../../../configs/models/DeepAR.toml).

## Differences

**Clean-room implementation: confirmed.** The autoregressive likelihood,
recurrent transition, and Gaussian parameterization map directly from the
paper; reference-only code was not copied. Mean feedback replaces ancestral
sampling, and published-metric/checkpoint reference comparison is not claimed.

## Shared components

- [`gaussian_parameter_head`](../_components/gaussian_parameter_head/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `embedding_size=32`, `hidden_size=64`, `num_layers=2`, `cov_feat_size=0`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
- **Venue**: International Journal of Forecasting 2020
- **Published**: 2020 (arXiv: 2017-04)
- **arXiv**: https://arxiv.org/abs/1704.04110

## Abstract
Probabilistic forecasting, i.e. estimating the probability distribution of a time series' future given its past, is a key enabler for optimizing business processes. In retail businesses, for example, forecasting demand is crucial for having the right inventory available at the right time at the right place. In this paper we propose DeepAR, a methodology for producing accurate probabilistic forecasts, based on training an auto regressive recurrent network model on a large number of related time series. We demonstrate how by applying deep learning techniques to forecasting, one can overcome many of the challenges faced by widely-used classical approaches to the problem. We show through extensive empirical evaluation on several real-world forecasting data sets accuracy improvements of around 15% compared to state-of-the-art methods.

## In TSFLab
Default config: `configs/models/DeepAR.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** The autoregressive likelihood,
recurrent transition, and Gaussian parameterization map directly from the
paper; reference-only code was not copied. Mean feedback replaces ancestral
sampling, and published-metric/checkpoint reference comparison is not claimed.

## Citation

```bibtex
@article{DBLP:journals/corr/FlunkertSG17,
  author       = {Valentin Flunkert and
                  David Salinas and
                  Jan Gasthaus},
  title        = {DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks},
  journal      = {CoRR},
  volume       = {abs/1704.04110},
  year         = {2017},
  url          = {http://arxiv.org/abs/1704.04110},
  eprinttype   = {arXiv},
  eprint       = {1704.04110},
  timestamp    = {Mon, 13 Aug 2018 16:46:25 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/FlunkertSG17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
