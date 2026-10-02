---
name: "LightGBMTS"
summary: "LightGBMTS is an independent differentiable additive-tree baseline with learned lag-feature gating and compact varying-depth stages."
paper: "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html"
paper_title: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
venue: "NeurIPS 2017"
year: 2017
tagline: "Differentiable additive soft trees of varying depth over a learned soft lag-feature gate, with residual state updates."
tags: ["tree", "ensemble", "boosting", "baseline", "feature-selection", "normalization"]
composition: ["normalization=component:revin", "decomposition=local:stagewise-residual-backcast", "temporal=component:soft_tree+local:additive-stagewise-corrections", "channel=local:flattened-channel-mixing", "head=local:linear-base-plus-tree-corrections", "loss=loss:mse+local:feature-gate-sparsity-aux-loss"]
---
# LightGBMTS

## Key ideas

- A learned sigmoid gate (`feature_logits`) softly selects lag features of the flattened window before the base linear map and every tree.
- `num_estimators` `SoftDecisionTree`s (`soft_tree`) of cycling depth `1 + index % tree_depth` add scaled corrections to a linear base forecast; learned backcasts update the input state between stages.
- `aux_loss = 1e-4 * mean(gate)` encourages sparse feature use; `revin` wraps the model.
- It does not implement LightGBM's histogram split search, leaf-wise growth, GOSS or EFB.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html); title: LightGBM: A Highly Efficient Gradient Boosting Decision Tree; venue/year: NeurIPS 2017 / 2017
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/LightGBMTS.toml`](../../../../configs/models/LightGBMTS.toml).

## Differences

This clean-room baseline uses learned soft feature gates and compact additive trees. It does not implement LightGBM's histogram split search, leaf-wise growth, GOSS, EFB, distributed systems, or external library API. The paper is conceptual background only; no external source code was inspected or copied. Evidence is in `../../../verification/evidence/LightGBMTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`soft_tree`](../_components/soft_tree/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_estimators=20`, `tree_depth=3`, `learning_rate=0.1`, `temperature=1.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- **Venue**: NeurIPS 2017
- **Published**: 2017
- **arXiv**: N/A

## Abstract
Gradient Boosting Decision Tree (GBDT) is a popular machine learning algorithm, and has quite a few effective implementations such as XGBoost and pGBRT. Although many engineering optimizations have been adopted in these implementations, the efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large. A major reason is that for each feature, they need to scan all the data instances to estimate the information gain of all possible split points, which is very time consuming. To tackle this problem, we propose two novel techniques: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). With GOSS, we exclude a significant proportion of data instances with small gradients, and only use the rest to estimate the information gain. We prove that, since the data instances with larger gradients play a more important role in the computation of information gain, GOSS can obtain quite accurate estimation of the information gain with a much smaller data size. With EFB, we bundle mutually exclusive features (i.e., they rarely take nonzero values simultaneously), to reduce the number of features. We prove that finding the optimal bundling of exclusive features is NP-hard, but a greedy algorithm can achieve quite good approximation ratio (and thus can effectively reduce the number of features without hurting the accuracy of split point determination by much). We call our new GBDT implementation with GOSS and EFB LightGBM. Our experiments on multiple public datasets show that, LightGBM speeds up the training process of conventional GBDT by up to over 20 times while achieving almost the same accuracy.

## In TSFLab
Default config: `configs/models/LightGBMTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Verification

This clean-room baseline uses learned soft feature gates and compact additive trees. It does not implement LightGBM's histogram split search, leaf-wise growth, GOSS, EFB, distributed systems, or external library API. The paper is conceptual background only; no external source code was inspected or copied. Evidence is in `../../../verification/evidence/LightGBMTS.json`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/KeMFWCMYL17,
  author       = {Guolin Ke and
                  Qi Meng and
                  Thomas Finley and
                  Taifeng Wang and
                  Wei Chen and
                  Weidong Ma and
                  Qiwei Ye and
                  Tie{-}Yan Liu},
  editor       = {Isabelle Guyon and
                  Ulrike von Luxburg and
                  Samy Bengio and
                  Hanna M. Wallach and
                  Rob Fergus and
                  S. V. N. Vishwanathan and
                  Roman Garnett},
  title        = {LightGBM: {A} Highly Efficient Gradient Boosting Decision Tree},
  booktitle    = {Advances in Neural Information Processing Systems 30: Annual Conference
                  on Neural Information Processing Systems 2017, December 4-9, 2017,
                  Long Beach, CA, {USA}},
  pages        = {3146--3154},
  year         = {2017},
  url          = {https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html},
  timestamp    = {Thu, 21 Jan 2021 15:15:21 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/KeMFWCMYL17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
