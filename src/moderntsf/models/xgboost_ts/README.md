---
name: "XGBoostTS"
summary: "XGBoostTS is an independent differentiable additive-tree baseline with column masks, shrinkage, and leaf-value regularization."
paper: "https://arxiv.org/abs/1603.02754"
paper_title: "XGBoost: A Scalable Tree Boosting System"
venue: "KDD 2016"
year: 2016
---
# XGBoostTS

XGBoostTS is an independent differentiable additive-tree baseline with column masks, shrinkage, and leaf-value regularization.

<!-- model-card:canonical:start -->
## Method overview

XGBoostTS is an independent differentiable additive-tree baseline with column masks, shrinkage, and leaf-value regularization.

## Core architecture

XGBoostTS is an independent differentiable additive-tree baseline with column masks, shrinkage, and leaf-value regularization.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1603.02754); title: XGBoost: A Scalable Tree Boosting System; venue/year: KDD 2016 / 2016
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/XGBoostTS.toml`](../../../../configs/models/XGBoostTS.toml).

## Differences

This clean-room baseline borrows only additive trees, shrinkage, feature subsampling, and regularization as high-level ideas. It does not implement XGBoost's second-order objective, sparsity-aware hard split search, quantile sketch, systems optimizations, or external library API. No XGBoost source code was inspected or copied. Evidence is in `../../../verification/evidence/XGBoostTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`soft_tree`](../_components/soft_tree/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_estimators=16`, `tree_depth=3`, `learning_rate=0.1`, `column_fraction=0.8`, `l1_penalty=0.0`, `l2_penalty=0.0001`, `temperature=1.0`, `random_seed=1741`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: XGBoost: A Scalable Tree Boosting System
- **Venue**: KDD 2016
- **Published**: 2016 (arXiv: 2016-03)
- **arXiv**: https://arxiv.org/abs/1603.02754

## Abstract
Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.

## In ModernTSF
Default config: `configs/models/XGBoostTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Verification

This clean-room baseline borrows only additive trees, shrinkage, feature subsampling, and regularization as high-level ideas. It does not implement XGBoost's second-order objective, sparsity-aware hard split search, quantile sketch, systems optimizations, or external library API. No XGBoost source code was inspected or copied. Evidence is in `../../../verification/evidence/XGBoostTS.json`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/ChenG16,
  author       = {Tianqi Chen and
                  Carlos Guestrin},
  editor       = {Balaji Krishnapuram and
                  Mohak Shah and
                  Alexander J. Smola and
                  Charu C. Aggarwal and
                  Dou Shen and
                  Rajeev Rastogi},
  title        = {XGBoost: {A} Scalable Tree Boosting System},
  booktitle    = {Proceedings of the 22nd {ACM} {SIGKDD} International Conference on
                  Knowledge Discovery and Data Mining, San Francisco, CA, USA, August
                  13-17, 2016},
  pages        = {785--794},
  publisher    = {{ACM}},
  year         = {2016},
  url          = {https://doi.org/10.1145/2939672.2939785},
  doi          = {10.1145/2939672.2939785},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/ChenG16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
