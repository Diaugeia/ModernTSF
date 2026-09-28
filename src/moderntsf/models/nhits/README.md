---
name: "NHiTS"
summary: "NHiTS (Neural Hierarchical Interpolation for Time Series) is a time-series forecasting model that addresses long-horizon prediction by stacking MLP blocks with multi-rate data sampling and hierarchical interpolation. Each block in the stack emphasises a different frequency band of the signal, and the blocks' outputs are combined to synthesise the final forecast."
paper: "https://arxiv.org/abs/2201.12886"
paper_title: "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting"
venue: "AAAI 2023"
year: 2023
code: "https://github.com/Nixtla/neuralforecast"
revision: "6c4f3e557d0ed672314323edba972eb550cb3550"
license: "Apache-2.0"
---
# NHiTS

NHiTS (Neural Hierarchical Interpolation for Time Series) is a time-series forecasting model that addresses long-horizon prediction by stacking MLP blocks with multi-rate data sampling and hierarchical interpolation. Each block in the stack emphasises a different frequency band of the signal, and the blocks' outputs are combined to synthesise the final forecast.

<!-- model-card:canonical:start -->
## Method overview

NHiTS (Neural Hierarchical Interpolation for Time Series) is a time-series forecasting model that addresses long-horizon prediction by stacking MLP blocks with multi-rate data sampling and hierarchical interpolation.

## Core architecture

Each block in the stack emphasises a different frequency band of the signal, and the blocks' outputs are combined to synthesise the final forecast.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2201.12886); title: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting; venue/year: AAAI 2023 / 2023
- [codebase](https://github.com/Nixtla/neuralforecast); revision: `6c4f3e557d0ed672314323edba972eb550cb3550`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/NHiTS.toml`](../../../../configs/models/NHiTS.toml).

## Differences

This implementation was rewritten for ModernTSF after checking both the paper
and the pinned official open-source implementation. Blocks use multi-rate
pooling, regress a full-window backcast and coarse forecast knots, interpolate
only the forecast knots, subtract backcasts, and add forecasts across the
hierarchy from the last observed level. Channels share the same univariate
blocks and canonical RevIN is optional. No external source file was copied or
mechanically adapted. The runnable default is not a paper benchmark reproduction.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `stack_types=['identity', 'identity', 'identity']`, `n_blocks=[1, 1, 1]`, `mlp_units=[[256, 256]]`, `n_pool_kernel_size=[2, 2, 1]`, `n_freq_downsample=[4, 2, 1]`, `pooling_mode='MaxPool1d'`, `interpolation_mode='linear'`, `dropout=0.0`, `activation='ReLU'`, `use_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-01)
- **arXiv**: https://arxiv.org/abs/2201.12886

## Abstract
Recent progress in neural forecasting accelerated improvements in the performance of large-scale forecasting systems. Yet, long-horizon forecasting remains a very difficult task. Two common challenges afflicting the task are the volatility of the predictions and their computational complexity. We introduce N-HiTS, a model which addresses both challenges by incorporating novel hierarchical interpolation and multi-rate data sampling techniques. These techniques enable the proposed method to assemble its predictions sequentially, emphasizing components with different frequencies and scales while decomposing the input signal and synthesizing the forecast. We prove that the hierarchical interpolation technique can efficiently approximate arbitrarily long horizons in the presence of smoothness. Additionally, we conduct extensive large-scale dataset experiments from the long-horizon forecasting literature, demonstrating the advantages of our method over the state-of-the-art methods, where N-HiTS provides an average accuracy improvement of almost 20% over the latest Transformer architectures while reducing the computation time by an order of magnitude (50 times).

## In ModernTSF
Default config: `configs/models/NHiTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This implementation was rewritten for ModernTSF after checking both the paper
and the pinned official open-source implementation. Blocks use multi-rate
pooling, regress a full-window backcast and coarse forecast knots, interpolate
only the forecast knots, subtract backcasts, and add forecasts across the
hierarchy from the last observed level. Channels share the same univariate
blocks and canonical RevIN is optional. No external source file was copied or
mechanically adapted. The runnable default is not a paper benchmark reproduction.

## Citation

```bibtex
@misc{challu2022nhits,
  author        = {Cristian Challu and
                  Kin G. Olivares and
                  Boris N. Oreshkin and
                  Federico Garza and
                  Max Mergenthaler-Canseco and
                  Artur Dubrawski},
  title         = {N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting},
  year          = {2022},
  eprint        = {2201.12886},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2201.12886}
}
```
