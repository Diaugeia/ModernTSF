---
name: "GRUForecasterTS"
summary: "GRUForecasterTS is a standard Gated Recurrent Unit (GRU) sequence-to-sequence forecaster registered for the time-series forecasting setting. It accepts a fixed-length historical window of univariate or multivariate values and produces a fixed-length forecast horizon by unrolling the GRU recurrence over the input and decoding the final hidden state."
paper: "https://arxiv.org/abs/1412.3555"
paper_title: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
venue: "arXiv preprint"
year: 2014
tagline: "GRU encoder over the normalized window with a direct linear decode of its final hidden state to the horizon."
tags: ["rnn", "baseline", "channel-mixing", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:gru-final-state-encoder", "channel=local:channel-mixing-recurrent-input", "head=local:linear-direct-multi-horizon-head", "loss=loss:mse"]
---
# GRUForecasterTS

## Key ideas

- An `nn.GRU` reads all channels jointly at each step, so channels are mixed in the recurrent state.
- The last layer's final state is mapped by one linear `head` to `pred_len * enc_in` values (direct multi-horizon, no autoregressive decoding).
- `revin` normalizes the input and denormalizes the forecast (`use_revin`).

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1412.3555); title: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling; venue/year: arXiv preprint / 2014
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GRUForecasterTS.toml`](../../../../configs/models/GRUForecasterTS.toml).

## Differences

Clean-room implementation: confirmed. The local code was independently designed from the cited GRU equations and the repository tensor contract; no external implementation source was copied. The paper studies sequence modeling rather than this final-state direct multi-horizon forecast head or optional RevIN, so no paper-result reference comparison is claimed. Formula and full runtime-contract evidence are recorded in `../../../../verification/evidence/GRUForecasterTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.0`, `num_layers=1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- **Venue**: arXiv preprint
- **Published**: 2014
- **arXiv**: https://arxiv.org/abs/1412.3555

## Abstract
In this paper we compare different types of recurrent units in recurrent neural networks (RNNs). Especially, we focus on more sophisticated units that implement a gating mechanism, such as a long short-term memory (LSTM) unit and a recently proposed gated recurrent unit (GRU). We evaluate these recurrent units on the tasks of polyphonic music modeling and speech signal modeling. Our experiments revealed that these advanced recurrent units are indeed better than more traditional recurrent units such as tanh units. Also, we found GRU to be comparable to LSTM.

## Source and verification

Clean-room implementation: confirmed. The local code was independently designed from the cited GRU equations and the repository tensor contract; no external implementation source was copied. The paper studies sequence modeling rather than this final-state direct multi-horizon forecast head or optional RevIN, so no paper-result reference comparison is claimed. Formula and full runtime-contract evidence are recorded in `../../../../verification/evidence/GRUForecasterTS.json`.

## In TSFLab
Default config: `configs/models/GRUForecasterTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@misc{chung2014empirical,
  author        = {Junyoung Chung and
                  Caglar Gulcehre and
                  KyungHyun Cho and
                  Yoshua Bengio},
  title         = {Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling},
  year          = {2014},
  eprint        = {1412.3555},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1412.3555}
}
```
