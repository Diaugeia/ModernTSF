---
name: "MLPForecasterTS"
summary: "MLPForecasterTS is a clean-room channel-wise multilayer perceptron that maps each fixed lag window directly to a multistep forecast."
paper: "https://doi.org/10.1038/323533a0"
paper_title: "Learning Representations by Back-Propagating Errors"
venue: "Nature"
year: 1986
tagline: "Per-channel MLP that maps the lag window straight to the horizon over the time axis, wrapped in RevIN."
tags: ["mlp", "channel-independent", "normalization", "baseline", "lightweight"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:lag-window-mlp", "channel=local:channel-independent-shared-weights", "head=local:linear-horizon-projection", "loss=loss:mse"]
---
# MLPForecasterTS

## Key ideas

- Transposes to `[batch, channels, seq_len]` so one `nn.Sequential` MLP (Linear, GELU, Dropout blocks) acts over time and is shared by every channel.
- Ends in a single `nn.Linear(d_model, pred_len)` that emits the whole multistep forecast directly (no recursion).
- Reversible instance normalization (`revin`) normalizes inputs and de-normalizes the forecast; `use_revin=False` disables it.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1038/323533a0); title: Learning Representations by Back-Propagating Errors; venue/year: Nature / 1986
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MLPForecasterTS.toml`](../../../../configs/models/MLPForecasterTS.toml).

## Differences

Clean-room implementation: confirmed. The local code was independently designed from published feed-forward/back-propagation concepts and the repository tensor contract; no external implementation source was copied. The citation does not prescribe a time-series architecture; the shared channel-wise lag mapping, GELU, direct horizon head, and optional RevIN are disclosed local choices. Formula and full runtime-contract evidence are recorded in `../../../verification/evidence/MLPForecasterTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `dropout=0.1`, `num_layers=1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Learning Representations by Back-Propagating Errors
- **Venue**: Nature
- **Published**: 1986
- **DOI**: https://doi.org/10.1038/323533a0

## Abstract
MLPForecasterTS is a foundational feedforward neural network baseline for time series forecasting. A Multi-Layer Perceptron (MLP) stacks fully connected layers with nonlinear activations to learn a direct mapping from a fixed-length historical window to a future prediction window. This implementation shares the lag-to-horizon network across channels, does not mix channels, and optionally applies Reversible Instance Normalization (RevIN) to stabilize scale changes.

## Source and verification

Clean-room implementation: confirmed. The local code was independently designed from published feed-forward/back-propagation concepts and the repository tensor contract; no external implementation source was copied. The citation does not prescribe a time-series architecture; the shared channel-wise lag mapping, GELU, direct horizon head, and optional RevIN are disclosed local choices. Formula and full runtime-contract evidence are recorded in `../../../verification/evidence/MLPForecasterTS.json`.

## In TSFLab
Default config: `configs/models/MLPForecasterTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@article{rumelhart1986learning,
  author  = {David E. Rumelhart and Geoffrey E. Hinton and Ronald J. Williams},
  title   = {Learning Representations by Back-Propagating Errors},
  journal = {Nature},
  volume  = {323},
  number  = {6088},
  pages   = {533--536},
  year    = {1986},
  doi     = {10.1038/323533a0}
}
```
