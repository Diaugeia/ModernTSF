---
name: "KNNForecasterTS"
summary: "KNNForecasterTS is a differentiable nearest-reference forecaster. It compares each input window with learned reference windows and uses soft distance-kernel weights to combine their learned future continuations."
paper: "https://doi.org/10.1109/TIT.1967.1053964"
paper_title: "Nearest Neighbor Pattern Classification"
venue: "IEEE Transactions on Information Theory 1967"
year: 1967
tagline: "Soft nearest-reference forecaster: distance-kernel weights over learned reference windows blend learned continuations."
tags: ["statistical", "retrieval", "nearest-neighbor", "baseline", "channel-mixing"]
composition: ["normalization=none", "decomposition=none", "temporal=local:soft-kernel-nearest-reference-weights", "channel=local:flattened-channel-mixing", "head=local:kernel-weighted-learned-futures", "loss=loss:mse"]
---
# KNNForecasterTS

## Key ideas

- Holds `num_prototypes` learned reference windows (`reference_windows`) and matching learned future continuations (`reference_futures`) as parameters.
- `neighbor_weights` is a softmax of negative scaled mean squared distance between the whole input window (all channels) and every reference (`kernel_gamma`).
- The forecast is the weighted sum of reference futures; it is not a hard KNN over a stored training set.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1109/TIT.1967.1053964); title: Nearest Neighbor Pattern Classification; venue/year: IEEE Transactions on Information Theory 1967 / 1967
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/KNNForecasterTS.toml`](../../../../configs/models/KNNForecasterTS.toml).

## Differences

This is an independent differentiable adaptation of the nearest-neighbor idea;
no external source implementation was inspected or copied. It is not a hard
KNN estimator over a stored training set: the reference windows and future
continuations are learned parameters, and all references contribute through a
soft distance kernel. The cited paper is conceptual background, not an
equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_prototypes=32`, `kernel_gamma=0.08`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
K-nearest neighbours (KNN) regression is a non-parametric method that predicts an output by averaging the target values of the k training samples closest (in feature space) to the query point, using a distance metric such as Euclidean distance. Applied to time-series forecasting, KNN finds the k historical windows most similar to the current input window and uses their corresponding future segments as the forecast. The method has no single defining paper; it originates from the general KNN algorithm described by Fix & Hodges (1951) and Cover & Hart (1967). In TSFLab, KNNForecasterTS replaces the hard discrete lookup with differentiable RBF-weighted prototypes so the model can be trained end-to-end with the standard gradient-based trainer and can run on CUDA/MPS devices.

## In TSFLab
Default config: `configs/models/KNNForecasterTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is an independent differentiable adaptation of the nearest-neighbor idea;
no external source implementation was inspected or copied. It is not a hard
KNN estimator over a stored training set: the reference windows and future
continuations are learned parameters, and all references contribute through a
soft distance kernel. The cited paper is conceptual background, not an
equivalence claim.

## Citation

```bibtex
@article{cover1967nearest,
  author  = {Thomas M. Cover and Peter E. Hart},
  title   = {Nearest Neighbor Pattern Classification},
  journal = {IEEE Transactions on Information Theory},
  volume  = {13},
  number  = {1},
  pages   = {21--27},
  year    = {1967},
  doi     = {10.1109/TIT.1967.1053964}
}
```
