---
name: "GradientBoostingTS"
summary: "GradientBoostingTS is an independent differentiable additive-tree baseline with sequential learned residual-state updates."
paper: "https://doi.org/10.1214/aos/1013203451"
paper_title: "Greedy function approximation: A gradient boosting machine"
venue: "Annals of Statistics, 2001"
year: 2001
---
# GradientBoostingTS

GradientBoostingTS is an independent differentiable additive-tree baseline with sequential learned residual-state updates.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1214/aos/1013203451); title: Greedy function approximation: A gradient boosting machine; venue/year: Annals of Statistics, 2001 / 2001
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GradientBoostingTS.toml`](../../../../configs/models/GradientBoostingTS.toml).

## Differences

This clean-room baseline applies all soft-tree stages end-to-end and updates an input-space residual through learned backcasts. It does not fit each tree to frozen loss pseudo-residuals or reproduce scikit-learn. The cited work supplies the stage-wise additive principle only; no external source code was inspected or copied. Evidence is in `../../../verification/evidence/GradientBoostingTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`soft_tree`](../_components/soft_tree/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_estimators=12`, `tree_depth=3`, `learning_rate=0.1`, `temperature=1.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Greedy function approximation: A gradient boosting machine
- **Venue**: Annals of Statistics, 2001
- **Published**: 2001
- **arXiv**: N/A

## Abstract
Function estimation/approximation is viewed from the perspective of numerical optimization in function space, rather than parameter space. A connection between stagewise additive expansions and steepest-descent minimization is identified. A general gradient descent "boosting" paradigm is developed for additive expansions based on any fitting criterion. Special enhancements are derived for regression with squared error loss, absolute error loss, and huberized M-loss, with applications to least-squares, least absolute deviation, and Huber-M loss functions for regression, and multiclass logistic likelihood for classification. Regression trees are shown to be especially amenable to this approach, giving rise to the Gradient Tree Boosting procedure. Competitive statistical performance of the resulting procedures is demonstrated on several datasets, producing highly robust, interpretable nonparametric regression and classification models appropriate for data mining applications.

## In ModernTSF
Default config: `configs/models/GradientBoostingTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Verification

This clean-room baseline applies all soft-tree stages end-to-end and updates an input-space residual through learned backcasts. It does not fit each tree to frozen loss pseudo-residuals or reproduce scikit-learn. The cited work supplies the stage-wise additive principle only; no external source code was inspected or copied. Evidence is in `../../../verification/evidence/GradientBoostingTS.json`.

## Citation

```bibtex
@article{friedman2001greedy,
  author  = {Jerome H. Friedman},
  title   = {Greedy Function Approximation: A Gradient Boosting Machine},
  journal = {The Annals of Statistics},
  volume  = {29},
  number  = {5},
  pages   = {1189--1232},
  year    = {2001},
  doi     = {10.1214/aos/1013203451}
}
```
