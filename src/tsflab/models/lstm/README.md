---
name: "LSTM"
summary: "LSTM is a per-node vanilla Long Short-Term Memory sequence predictor applied in the spatiotemporal forecasting setting. Each spatial node is modeled independently as a univariate sequence, with the LSTM gates learning to selectively retain or forget information across timesteps — providing a simple but effective recurrent baseline for node-structured time series data."
paper: "https://doi.org/10.1162/neco.1997.9.8.1735"
paper_title: "Long Short-Term Memory"
venue: "Neural Computation 1997"
year: 1997
code: "https://github.com/PoorOtterBob/CauAir"
revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
license: "NOASSERTION"
tagline: "Per-node LSTM with shared weights over value plus calendar covariates and a two-layer MLP horizon head."
tags: ["rnn", "spatiotemporal", "covariates", "baseline", "channel-independent"]
composition: ["normalization=none", "decomposition=none", "temporal=local:lstm-per-node-encoder", "channel=local:channel-independent-shared-weights", "head=local:two-layer-mlp-horizon-head", "loss=loss:mse"]
---
# LSTM

## Key ideas

- Each node is treated as its own sequence (nodes folded into the batch), so no graph or cross-node interaction is used.
- `to_spatiotemporal` supplies the value plus `cov_dim` calendar covariates, which are linearly projected (`input_projection`) before a multi-layer `nn.LSTM`.
- The last hidden state feeds a `Linear` - GELU - `Linear` head that emits all `pred_len` steps directly.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1162/neco.1997.9.8.1735); title: Long Short-Term Memory; venue/year: Neural Computation 1997 / 1997
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/LSTM.toml`](../../../../configs/models/LSTM.toml).

## Differences

**Clean-room implementation: confirmed.** Gate-based recurrence, shared
per-node encoding, optional covariates, and the direct horizon decoder have
focused structure/runtime evidence. No reference implementation was copied and
paper-task or checkpoint reference comparison is not claimed.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `init_dim=32`, `hid_dim=64`, `end_dim=128`, `layer=2`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Long Short-Term Memory
- **Venue**: Neural Computation 1997
- **Published**: 1997
- **arXiv**: N/A

## Abstract
Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through constant error carousels within special units. Multiplicative gate units learn to open and close access to the constant error flow. Local in space and time; their computational complexity per time step and weight is O(1). Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms.

## In TSFLab
Default config: `configs/models/LSTM.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`. CauAir remains reference-only.

## Verification

**Clean-room implementation: confirmed.** Gate-based recurrence, shared
per-node encoding, optional covariates, and the direct horizon decoder have
focused structure/runtime evidence. No reference implementation was copied and
paper-task or checkpoint reference comparison is not claimed.

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
  doi     = {10.1162/neco.1997.9.8.1735},
  url     = {https://doi.org/10.1162/neco.1997.9.8.1735}
}
```
