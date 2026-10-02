---
name: "ElasticNetTS"
summary: "ElasticNetTS is a direct channel-wise lag-regression forecast with the standard convex combination of L1 and L2 weight penalties exposed through `aux_loss`."
paper: "https://doi.org/10.1111/j.1467-9868.2005.00503.x"
paper_title: "Regularization and Variable Selection via the Elastic Net"
venue: "Journal of the Royal Statistical Society, Series B"
year: 2005
---
# ElasticNetTS

ElasticNetTS is a direct channel-wise lag-regression forecast with the standard convex combination of L1 and L2 weight penalties exposed through `aux_loss`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1111/j.1467-9868.2005.00503.x); title: Regularization and Variable Selection via the Elastic Net; venue/year: Journal of the Royal Statistical Society, Series B / 2005
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ElasticNetTS.toml`](../../../../configs/models/ElasticNetTS.toml).

## Differences

This is a clean-room, gradient-optimized direct multi-horizon adaptation. It shares coefficients across channels and does not reproduce the paper's least-squares solution path or variable-selection experiments. No third-party implementation was inspected or copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `penalty=0.0001`, `l1_ratio=0.5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Regularization and Variable Selection via the Elastic Net
- **Venue**: Journal of the Royal Statistical Society, Series B
- **Published**: 2005
- **Link**: https://doi.org/10.1111/j.1467-9868.2005.00503.x

## Abstract
Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalty terms of the Lasso and Ridge methods. It was introduced by Zou and Hastie (2005) to address the limitations of Lasso — in particular its instability when features are correlated and its inability to select more variables than observations. The Elastic Net penalty encourages a grouping effect in which strongly correlated predictors tend to be selected or dropped together. This combination achieves the sparsity of Lasso and the stability of Ridge, making it well suited to high-dimensional regression and variable selection problems where predictors exhibit correlation structure.

## In TSFLab
Default config: `configs/models/ElasticNetTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is a clean-room, gradient-optimized direct multi-horizon adaptation. It shares coefficients across channels and does not reproduce the paper's least-squares solution path or variable-selection experiments. No third-party implementation was inspected or copied.

## Citation

```bibtex
@article{zou2005elasticnet,
  author  = {Hui Zou and Trevor Hastie},
  title   = {Regularization and Variable Selection via the Elastic Net},
  journal = {Journal of the Royal Statistical Society: Series {B} (Statistical Methodology)},
  volume  = {67},
  number  = {2},
  pages   = {301--320},
  year    = {2005},
  doi     = {10.1111/j.1467-9868.2005.00503.x}
}
```
