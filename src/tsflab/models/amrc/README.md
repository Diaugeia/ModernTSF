---
name: "AMRC"
summary: "AMRC is a clean-room realization of Adaptive Masking Loss and Embedding Similarity Penalty over a compact channel-independent forecasting carrier."
paper: "https://arxiv.org/abs/2510.19980"
paper_title: "Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/MazelTovy/AMRC"
revision: "c0d742c6dad73c2fa5ed1c40ae57affc6740f40e"
license: "NOASSERTION"
tagline: "Training objective adding adaptive prefix-masking loss and embedding-similarity penalty to a compact linear forecaster."
tags: ["mlp", "normalization", "channel-independent", "masking", "loss-function", "representation-learning"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:linear-embedding-encoder", "channel=local:channel-independent-shared-weights", "head=local:linear-predictor", "loss=loss:mse+local:adaptive-masking-loss+local:embedding-similarity-penalty"]
---
# AMRC

## Key ideas

- AMRC is an optimization method; the forecaster is a compact carrier (linear encoder with `GELU` and `LayerNorm`, then a linear predictor) applied per channel, wrapped by `revin`.
- `adaptive_masking_loss` masks the oldest prefix of the lookback at several lengths, keeps the best-performing mask per sample, and pulls the unmasked embedding toward its embedding, weighted by the relative improvement.
- `embedding_similarity_penalty` matches pairwise embedding distances to pairwise target distances within a batch.
- `training_objective` sums MSE with these two terms (`lambda_aml`, `lambda_esp`); plain `forward` is point forecasting only.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.19980); title: Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/MazelTovy/AMRC); revision: `c0d742c6dad73c2fa5ed1c40ae57affc6740f40e`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AMRC.toml`](../../../../configs/models/AMRC.toml).

## Differences

Clean-room implementation: confirmed. The reference-only repository was not
inspected or copied. The local carrier implements paper equations (6)--(14):
sampled prefix selection, improvement-weighted embedding alignment, pairwise
embedding/output geometry matching, and the combined AMRC training objective.

The paper is backbone-agnostic, so this entry supplies a compact
channel-independent carrier. Default mask candidates are evenly spaced unless
the caller supplies stochastic lengths. Generic point-forecast training uses
`forward`; experiments must call `training_loss` to activate AML and ESP.
Executable evidence is in `../../../verification/evidence/AMRC.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `mask_samples=4`, `lambda_aml=0.1`, `lambda_esp=0.1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.19980

## Abstract
Time series forecasting plays a pivotal role in critical domains such as energy management and financial markets. Although deep learning-based approaches (e.g., MLP, RNN, Transformer) have achieved remarkable progress, the prevailing "long-sequence information gain hypothesis" exhibits inherent limitations. Through systematic experimentation, this study reveals a counterintuitive phenomenon: appropriately truncating historical data can paradoxically enhance prediction accuracy, indicating that existing models learn substantial redundant features (e.g., noise or irrelevant fluctuations) during training, thereby compromising effective signal extraction. Building upon information bottleneck theory, we propose an innovative solution termed Adaptive Masking Loss with Representation Consistency (AMRC), which features two core components: 1) Dynamic masking loss, which adaptively identified highly discriminative temporal segments to guide gradient descent during model training; 2) Representation consistency constraint, which stabilized the mapping relationships among inputs, labels, and predictions. Experimental results demonstrate that AMRC effectively suppresses redundant feature learning while significantly improving model performance. This work not only challenges conventional assumptions in temporal modeling but also provides novel theoretical insights and methodological breakthroughs for developing efficient and robust forecasting models.

## Source and verification

Clean-room implementation: confirmed. The reference-only repository was not
inspected or copied. The local carrier implements paper equations (6)--(14):
sampled prefix selection, improvement-weighted embedding alignment, pairwise
embedding/output geometry matching, and the combined AMRC training objective.

The paper is backbone-agnostic, so this entry supplies a compact
channel-independent carrier. Default mask candidates are evenly spaced unless
the caller supplies stochastic lengths. Generic point-forecast training uses
`forward`; experiments must call `training_loss` to activate AML and ESP.
Executable evidence is in `../../../verification/evidence/AMRC.json`.

## In TSFLab
Default config: `configs/models/AMRC.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2510-19980,
  author       = {Renzhao Liang and
                  Sizhe Xu and
                  Chenggang Xie and
                  Jingru Chen and
                  Feiyang Ren and
                  Shu Yang and
                  Takahiro Yabe},
  title        = {Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking
                  Loss with Representation Consistency},
  journal      = {CoRR},
  volume       = {abs/2510.19980},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2510.19980},
  doi          = {10.48550/ARXIV.2510.19980},
  eprinttype   = {arXiv},
  eprint       = {2510.19980},
  timestamp    = {Wed, 04 Mar 2026 19:44:06 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2510-19980.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
