---
name: "LassoRegressionTS"
summary: "LassoRegressionTS applies a shared channel-wise lag projection to the forecast horizon and exposes the Lasso L1 weight penalty through `aux_loss` for the standard trainer."
paper: "https://doi.org/10.1111/j.2517-6161.1996.tb02080.x"
paper_title: "Regression Shrinkage and Selection via the Lasso"
venue: "Journal of the Royal Statistical Society: Series B, 1996"
year: 1996
---
# LassoRegressionTS

LassoRegressionTS applies a shared channel-wise lag projection to the forecast horizon and exposes the Lasso L1 weight penalty through `aux_loss` for the standard trainer.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x); title: Regression Shrinkage and Selection via the Lasso; venue/year: Journal of the Royal Statistical Society: Series B, 1996 / 1996
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/LassoRegressionTS.toml`](../../../../configs/models/LassoRegressionTS.toml).

## Differences

This is an independent implementation from the cited Lasso objective; no
external source implementation was inspected or copied. It optimizes a direct
multi-horizon lag projection with gradient descent rather than a coordinate
descent solver. Coefficients are shared across channels, and the L1 weight term
is exposed to the trainer as `aux_loss`.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `l1_penalty=1e-05`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Regression Shrinkage and Selection via the Lasso
- **Venue**: Journal of the Royal Statistical Society: Series B, 1996
- **Published**: 1996
- **arXiv**: N/A

## Abstract
Lasso (Least Absolute Shrinkage and Selection Operator) is a classical penalised regression method introduced by Tibshirani (1996). It minimises the residual sum of squares subject to the sum of the absolute values of the regression coefficients being less than a constant. This L1 constraint has the effect of shrinking some coefficients exactly to zero, producing sparse and interpretable models while avoiding the instability of ordinary subset selection. The method combines the variable-selection capability of subset regression with the continuous shrinkage of ridge regression, making it effective when only a small subset of predictors is truly informative. In the time-series forecasting setting, Lasso regression is applied channel-by-channel over lag features derived from the historical input window, using L1 regularisation to identify the most predictive lags for each output channel.

## In ModernTSF
Default config: `configs/models/LassoRegressionTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is an independent implementation from the cited Lasso objective; no
external source implementation was inspected or copied. It optimizes a direct
multi-horizon lag projection with gradient descent rather than a coordinate
descent solver. Coefficients are shared across channels, and the L1 weight term
is exposed to the trainer as `aux_loss`.

## Citation

```bibtex
@article{tibshirani1996regression,
  author  = {Robert Tibshirani},
  title   = {Regression Shrinkage and Selection via the Lasso},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume  = {58},
  number  = {1},
  pages   = {267--288},
  year    = {1996},
  doi     = {10.1111/j.2517-6161.1996.tb02080.x}
}
```
