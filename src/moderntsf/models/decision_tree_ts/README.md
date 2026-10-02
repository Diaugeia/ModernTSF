---
name: "DecisionTreeTS"
summary: "DecisionTreeTS is an independent differentiable single-tree baseline over flattened lag windows."
paper: "https://search.worldcat.org/title/1422106714"
paper_title: "Classification and Regression Trees"
venue: "Wadsworth, 1984"
year: 1984
---
# DecisionTreeTS

DecisionTreeTS is an independent differentiable single-tree baseline over flattened lag windows.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://search.worldcat.org/title/1422106714); title: Classification and Regression Trees; venue/year: Wadsworth, 1984 / 1984
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DecisionTreeTS.toml`](../../../../configs/models/DecisionTreeTS.toml).

## Differences

This is a clean-room, end-to-end differentiable soft tree, not a CART training implementation. It does not greedily select impurity-reducing hard splits, prune a fitted tree, or reproduce scikit-learn. The book is conceptual background only; no external source code was inspected or copied. The verified formula map and runtime observations are in `../../../verification/evidence/DecisionTreeTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`soft_tree`](../_components/soft_tree/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `tree_depth=4`, `temperature=1.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Decision trees are classical non-parametric supervised learning models that recursively partition the input feature space using axis-aligned splits, selecting the split at each node by minimising an impurity criterion (e.g., mean squared error for regression). For time series forecasting, the model is applied by constructing a feature matrix of lagged input values and training a separate tree (or a single multi-output tree) to predict each future step. Although decision trees are highly interpretable and require no gradient-based optimisation, they can overfit without regularisation (maximum depth, minimum samples per leaf) and do not naturally capture sequential structure. In ModernTSF they are wrapped as a differentiable-style torch.nn.Module for uniform pipeline integration.

## In ModernTSF
Default config: `configs/models/DecisionTreeTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Verification

This is a clean-room, end-to-end differentiable soft tree, not a CART training implementation. It does not greedily select impurity-reducing hard splits, prune a fitted tree, or reproduce scikit-learn. The book is conceptual background only; no external source code was inspected or copied. The verified formula map and runtime observations are in `../../../verification/evidence/DecisionTreeTS.json`.

## Citation

```bibtex
@book{DBLP:books/wa/BreimanFOS84,
  author       = {Leo Breiman and
                  J. H. Friedman and
                  Richard A. Olshen and
                  C. J. Stone},
  title        = {Classification and Regression Trees},
  publisher    = {Wadsworth},
  year         = {1984},
  isbn         = {0-534-98053-8},
  timestamp    = {Mon, 10 Jul 2023 12:50:10 +0200},
  biburl       = {https://dblp.org/rec/books/wa/BreimanFOS84.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
