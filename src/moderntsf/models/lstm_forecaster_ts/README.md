---
name: "LSTMForecasterTS"
summary: "LSTMForecasterTS is a clean-room LSTM baseline that encodes a fixed history and directly projects the final hidden state to a multistep forecast."
paper: "https://doi.org/10.1162/neco.1997.9.8.1735"
paper_title: "Long Short-Term Memory"
venue: "Neural Computation"
year: 1997
---
# LSTMForecasterTS

LSTMForecasterTS is a clean-room LSTM baseline that encodes a fixed history and directly projects the final hidden state to a multistep forecast.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1162/neco.1997.9.8.1735); title: Long Short-Term Memory; venue/year: Neural Computation / 1997
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/LSTMForecasterTS.toml`](../../../../configs/models/LSTMForecasterTS.toml).

## Differences

Clean-room implementation: confirmed. The local code was independently designed from the published LSTM gate equations and the repository tensor contract; no external implementation source was copied. The 1997 paper does not define the direct multi-horizon head, joint-channel setup, or optional RevIN, so no experimental reference comparison is claimed. Formula and full runtime-contract evidence are recorded in `../../../verification/evidence/LSTMForecasterTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.0`, `num_layers=1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Long Short-Term Memory
- **Venue**: Neural Computation
- **Published**: 1997
- **DOI**: https://doi.org/10.1162/neco.1997.9.8.1735

## Abstract
Long Short-Term Memory (LSTM) is a gated recurrent neural network architecture introduced by Hochreiter and Schmidhuber (1997) to address the vanishing-gradient problem in standard RNNs. An LSTM cell maintains a cell state and three learned gates — input, forget, and output — that regulate how information flows across time steps, allowing the network to selectively remember or discard information over long sequences. In the forecasting setting used here, the encoder processes the historical window token-by-token and the final hidden state seeds a linear projection head that produces the full prediction horizon in one shot. No single canonical paper defines this forecasting variant; the classical LSTM architecture is the sole methodological contribution.

## Source and verification

Clean-room implementation: confirmed. The local code was independently designed from the published LSTM gate equations and the repository tensor contract; no external implementation source was copied. The 1997 paper does not define the direct multi-horizon head, joint-channel setup, or optional RevIN, so no experimental reference comparison is claimed. Formula and full runtime-contract evidence are recorded in `../../../verification/evidence/LSTMForecasterTS.json`.

## In ModernTSF
Default config: `configs/models/LSTMForecasterTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@article{hochreiter1997long,
  author  = {Sepp Hochreiter and J{\"u}rgen Schmidhuber},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997},
  doi     = {10.1162/neco.1997.9.8.1735}
}
```
