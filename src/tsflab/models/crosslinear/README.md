---
name: "CrossLinear"
summary: "CrossLinear is a linear-based time-series forecasting model designed for settings that include exogenous (external) variables. It incorporates a lightweight plug-and-play cross-correlation embedding module that captures time-invariant, direct variable dependencies between endogenous and exogenous channels while avoiding overfitting to time-varying or indirect dependencies. Patch-wise processing and a global linear head handle both short- and long-range temporal structure, serving the standard multivariate forecasting setting."
paper: "https://arxiv.org/abs/2505.23116"
paper_title: "CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables"
venue: "KDD 2025"
year: 2025
code: "https://github.com/mumiao2000/CrossLinear"
revision: "d22366e2f59ced560a02b2b1c7cc673e3c02a13f"
license: "MIT"
tagline: "Linear forecaster with a time-invariant cross-variate conv embedding and patch projection for exogenous variables."
tags: ["linear", "patching", "channel-mixing", "normalization", "lightweight", "covariates"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:patch-projection-position-blend", "channel=local:cross-correlation-conv-embedding", "head=local:global-flatten-linear-head", "loss=loss:mse"]
---
# CrossLinear

## Key ideas

- `CrossCorrelationEmbedding` blends the input with one direct Conv1d across variables using a learned alpha, capturing time-invariant endogenous-exogenous dependencies without a deep mixer.
- `PatchForecastHead` projects patches with a small MLP and blends them with positional embeddings by a learned beta.
- A single global linear layer over all patch embeddings gives the forecast; `revin` (non-affine) normalizes and denormalizes.
- Weights are shared across variables, the many-to-many extension of the paper.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.23116); title: CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables; venue/year: KDD 2025 / 2025
- [codebase](https://github.com/mumiao2000/CrossLinear); revision: `d22366e2f59ced560a02b2b1c7cc673e3c02a13f`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CrossLinear.toml`](../../../../configs/models/CrossLinear.toml).

## Differences

Clean-room implementation: confirmed.

Clean-room implementation confirmed against paper equations (3)--(11); the reference-only repository was not copied. The parameter-free reversible normalization, direct one-layer cross-correlation embedding, learned alpha/beta residual blends, patch projection, positional embedding, and global linear forecast head are mapped to local modules. TSFLab implements the paper's weight-shared many-to-many extension, not its target-channel many-to-one/MS data path, and does not reproduce the publication's data pipeline or optimization protocol.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `d_model=32`, `d_ff=128`, `alpha=0.5`, `beta=0.5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables
- **Venue**: KDD 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.23116

## Abstract
Time series forecasting with exogenous variables is a critical emerging paradigm that presents unique challenges in modeling dependencies between variables. Traditional models often struggle to differentiate between endogenous and exogenous variables, leading to inefficiencies and overfitting. In this paper, we introduce CrossLinear, a novel Linear-based forecasting model that addresses these challenges by incorporating a plug-and-play cross-correlation embedding module. This lightweight module captures the dependencies between variables with minimal computational cost and seamlessly integrates into existing neural networks. Specifically, it captures time-invariant and direct variable dependencies while disregarding time-varying or indirect dependencies, thereby mitigating the risk of overfitting in dependency modeling and contributing to consistent performance improvements. Furthermore, CrossLinear employs patch-wise processing and a global linear head to effectively capture both short-term and long-term temporal dependencies, further improving its forecasting precision. Extensive experiments on 12 real-world datasets demonstrate that CrossLinear achieves superior performance in both short-term and long-term forecasting tasks. The ablation study underscores the effectiveness of the cross-correlation embedding module. Additionally, the generalizability of this module makes it a valuable plug-in for various forecasting tasks across different domains.

## In TSFLab
Default config: `configs/models/CrossLinear.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed.

Clean-room implementation confirmed against paper equations (3)--(11); the reference-only repository was not copied. The parameter-free reversible normalization, direct one-layer cross-correlation embedding, learned alpha/beta residual blends, patch projection, positional embedding, and global linear forecast head are mapped to local modules. TSFLab implements the paper's weight-shared many-to-many extension, not its target-channel many-to-one/MS data path, and does not reproduce the publication's data pipeline or optimization protocol.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/ZhouLL0025,
  author       = {Pengfei Zhou and
                  Yunlong Liu and
                  Junli Liang and
                  Qi Song and
                  Xiangyang Li},
  editor       = {Luiza Antonie and
                  Jian Pei and
                  Xiaohui Yu and
                  Flavio Chierichetti and
                  Hady W. Lauw and
                  Yizhou Sun and
                  Srinivasan Parthasarathy},
  title        = {CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series
                  Forecasting with Exogenous Variables},
  booktitle    = {Proceedings of the 31st {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, V.2, {KDD} 2025, Toronto ON, Canada, August 3-7,
                  2025},
  pages        = {4120--4131},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3711896.3736899},
  doi          = {10.1145/3711896.3736899},
  timestamp    = {Wed, 24 Dec 2025 10:44:06 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/ZhouLL0025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
