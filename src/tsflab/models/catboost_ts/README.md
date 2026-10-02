---
name: "CatBoostTS"
summary: "CatBoostTS is an independent differentiable baseline using symmetric soft trees and prior-stage forecast context."
paper: "https://arxiv.org/abs/1706.09516"
paper_title: "CatBoost: unbiased boosting with categorical features"
venue: "NeurIPS 2018"
year: 2018
tagline: "Stack of soft symmetric (oblivious) trees where each stage sees the input minus a tanh context of the running forecast."
tags: ["tree", "boosting", "normalization", "channel-mixing", "baseline"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:soft_tree+local:ordered-context-boosting-stages", "channel=local:flattened-channel-mixing-input", "head=local:linear-base-plus-tree-residuals", "loss=loss:mse"]
---
# CatBoostTS

## Key ideas

- `soft_tree` provides differentiable oblivious trees (`SoftObliviousTree`) of fixed depth, trained by gradient descent rather than greedy splits.
- A linear `base` forecast is refined in `num_estimators` stages, each adding `learning_rate` times a tree output.
- Each stage's tree input is the flattened series minus `tanh(context(forecast / stage))`, a differentiable stand-in for ordered-boosting prior context.
- Channels and time are flattened into one feature vector; `revin` normalizes the input. CatBoost's permutation-based ordered boosting and target statistics are not implemented.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1706.09516); title: CatBoost: unbiased boosting with categorical features; venue/year: NeurIPS 2018 / 2018
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CatBoostTS.toml`](../../../../configs/models/CatBoostTS.toml).

## Differences

This clean-room baseline uses oblivious soft trees and conditions each stage on prior forecast context. It does not implement CatBoost's permutation-based ordered boosting, ordered target statistics, categorical-feature processing, or external library API. No CatBoost source code was inspected or copied. Evidence is in `../../../../verification/evidence/CatBoostTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`soft_tree`](../_components/soft_tree/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_estimators=16`, `tree_depth=3`, `learning_rate=0.1`, `temperature=1.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: CatBoost: unbiased boosting with categorical features
- **Venue**: NeurIPS 2018
- **Published**: 2018 (arXiv: 2017-06)
- **arXiv**: https://arxiv.org/abs/1706.09516

## Abstract
This paper presents the key algorithmic techniques behind CatBoost, a new gradient boosting toolkit. Their combination leads to CatBoost outperforming other publicly available boosting implementations in terms of quality on a variety of datasets. Two critical algorithmic advances introduced in CatBoost are the implementation of ordered boosting, a permutation-driven alternative to the classic algorithm, and an innovative algorithm for processing categorical features. Both techniques were created to fight a prediction shift caused by a special kind of target leakage present in all currently existing implementations of gradient boosting algorithms. In this paper, we provide a detailed analysis of this problem and demonstrate that proposed algorithms solve it effectively, leading to excellent empirical results.

## In TSFLab
Default config: `configs/models/CatBoostTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Verification

This clean-room baseline uses oblivious soft trees and conditions each stage on prior forecast context. It does not implement CatBoost's permutation-based ordered boosting, ordered target statistics, categorical-feature processing, or external library API. No CatBoost source code was inspected or copied. Evidence is in `../../../../verification/evidence/CatBoostTS.json`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/ProkhorenkovaGV18,
  author       = {Liudmila Ostroumova Prokhorenkova and
                  Gleb Gusev and
                  Aleksandr Vorobev and
                  Anna Veronika Dorogush and
                  Andrey Gulin},
  editor       = {Samy Bengio and
                  Hanna M. Wallach and
                  Hugo Larochelle and
                  Kristen Grauman and
                  Nicol{\`{o}} Cesa{-}Bianchi and
                  Roman Garnett},
  title        = {CatBoost: unbiased boosting with categorical features},
  booktitle    = {Advances in Neural Information Processing Systems 31: Annual Conference
                  on Neural Information Processing Systems 2018, NeurIPS 2018, December
                  3-8, 2018, Montr{\'{e}}al, Canada},
  pages        = {6639--6649},
  year         = {2018},
  url          = {https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html},
  timestamp    = {Mon, 16 May 2022 15:41:51 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/ProkhorenkovaGV18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
