---
name: "ExtraTreesTS"
summary: "ExtraTreesTS is an independent differentiable ensemble with frozen random axis-aligned splits and learned leaf forecasts."
paper: "https://doi.org/10.1007/s10994-006-6226-1"
paper_title: "Extremely Randomized Trees"
venue: "Machine Learning 2006"
year: 2006
---
# ExtraTreesTS

ExtraTreesTS is an independent differentiable ensemble with frozen random axis-aligned splits and learned leaf forecasts.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1007/s10994-006-6226-1); title: Extremely Randomized Trees; venue/year: Machine Learning 2006 / 2006
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ExtraTreesTS.toml`](../../../../configs/models/ExtraTreesTS.toml).

## Differences

This clean-room baseline samples feature axes and normalized thresholds once, freezes that split geometry, and learns only leaf forecasts. It uses soft routing and gradient fitting; it is not the Extra-Trees training algorithm and does not reproduce scikit-learn. No external source code was inspected or copied. Evidence is in `../../../verification/evidence/ExtraTreesTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`soft_tree`](../_components/soft_tree/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_estimators=24`, `tree_depth=2`, `threshold_range=1.0`, `temperature=1.0`, `random_seed=1733`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Extremely Randomized Trees
- **Venue**: Machine Learning 2006
- **Published**: 2006
- **arXiv**: N/A

## Abstract
Extremely Randomized Trees (Extra-Trees) is a tree-based ensemble learning method introduced by Geurts, Ernst, and Wehenkel (2006). Like Random Forests, it builds an ensemble of unpruned decision or regression trees from the full training set, but with two key differences that increase randomisation: (1) split points are chosen uniformly at random within each feature's range rather than by optimising an impurity criterion, and (2) all training samples are used for building each tree (no bootstrap). These two choices trade a small increase in bias for a substantial reduction in variance and a significant speedup in training. The method consistently achieves competitive accuracy with Random Forests and gradient-boosted trees across regression and classification benchmarks, while being considerably faster to train.

## In ModernTSF
Default config: `configs/models/ExtraTreesTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Verification

This clean-room baseline samples feature axes and normalized thresholds once, freezes that split geometry, and learns only leaf forecasts. It uses soft routing and gradient fitting; it is not the Extra-Trees training algorithm and does not reproduce scikit-learn. No external source code was inspected or copied. Evidence is in `../../../verification/evidence/ExtraTreesTS.json`.

## Citation

```bibtex
@article{geurts2006extremely,
  author  = {Pierre Geurts and Damien Ernst and Louis Wehenkel},
  title   = {Extremely Randomized Trees},
  journal = {Machine Learning},
  volume  = {63},
  number  = {1},
  pages   = {3--42},
  year    = {2006},
  doi     = {10.1007/s10994-006-6226-1}
}
```
