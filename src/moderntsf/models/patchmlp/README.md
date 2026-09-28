---
name: "PatchMLP"
summary: "PatchMLP is a patch-based MLP model for long-term time series forecasting that attributes the effectiveness of recent Transformer models to their patch mechanism rather than to attention. It applies moving-average decomposition to separate smooth trend components from noise residuals, then processes the smooth branch with cross-variable channel mixing for semantic information exchange and handles the residual branch with channel-independent linear layers, achieving competitive accuracy without any attention operations."
paper: "https://arxiv.org/abs/2405.13575"
paper_title: "Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
code: "https://github.com/TangPeiwang/PatchMLP"
revision: "b36bbc92ecfc4732acaabb6d5e8c4ff487876f5d"
license: "NOASSERTION"
---
# PatchMLP

PatchMLP is a patch-based MLP model for long-term time series forecasting that attributes the effectiveness of recent Transformer models to their patch mechanism rather than to attention. It applies moving-average decomposition to separate smooth trend components from noise residuals, then processes the smooth branch with cross-variable channel mixing for semantic information exchange and handles the residual branch with channel-independent linear layers, achieving competitive accuracy without any attention operations.

<!-- model-card:canonical:start -->
## Method overview

PatchMLP is a patch-based MLP model for long-term time series forecasting that attributes the effectiveness of recent Transformer models to their patch mechanism rather than to attention.

## Core architecture

It applies moving-average decomposition to separate smooth trend components from noise residuals, then processes the smooth branch with cross-variable channel mixing for semantic information exchange and handles the residual branch with channel-independent linear layers, achieving competitive accuracy without any attention operations.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2405.13575); title: Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting; venue/year: AAAI 2025 / 2025
- [codebase](https://github.com/TangPeiwang/PatchMLP); revision: `b36bbc92ecfc4732acaabb6d5e8c4ff487876f5d`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PatchMLP.toml`](../../../../configs/models/PatchMLP.toml).

## Differences

Pinned source inspection: `model/PatchMLP.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. This rewrite follows the paper's architecture and equations. The authors' repository is recorded as `reference-only` because it does not declare a repository license; its source was inspected at the pinned revision; no external source code was copied.
- The implementation covers four-scale patch embedding and latent decomposition `X_s=AvgPool(X), X_r=X-X_s`; the noisy residual branch remains channel-independent while the smooth branch adds dot-product-style inter-variable mixing after its intra-variable temporal MLP. Both branches use residual normalization before horizon projection.
- The paper does not fully specify every initialization and dataset-specific setting; no numerical-reference comparison or reported-result claim is made.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=1024`, `e_layers=1`, `use_norm=True`, `moving_avg=13`, `patch_len=[48, 24, 12, 6]`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.13575

## Abstract
Recent studies have attempted to refine the Transformer architecture to demonstrate its effectiveness in Long-Term Time Series Forecasting (LTSF) tasks. Despite surpassing many linear forecasting models with ever-improving performance, we remain skeptical of Transformers as a solution for LTSF. We attribute the effectiveness of these models largely to the adopted Patch mechanism, which enhances sequence locality to an extent yet fails to fully address the loss of temporal information inherent to the permutation-invariant self-attention mechanism. Further investigation suggests that simple linear layers augmented with the Patch mechanism may outperform complex Transformer-based LTSF models. Moreover, diverging from models that use channel independence, our research underscores the importance of cross-variable interactions in enhancing the performance of multivariate time series forecasting. The interaction information between variables is highly valuable but has been misapplied in past studies, leading to suboptimal cross-variable models. Based on these insights, we propose a novel and simple Patch-based MLP (PatchMLP) for LTSF tasks. Specifically, we employ simple moving averages to extract smooth components and noise-containing residuals from time series data, engaging in semantic information interchange through channel mixing and specializing in random noise with channel independence processing. The PatchMLP model consistently achieves state-of-the-art results on several real-world datasets. We hope this surprising finding will spur new research directions in the LTSF field and pave the way for more efficient and concise solutions.

## In ModernTSF
Default config: `configs/models/PatchMLP.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Pinned source inspection: `model/PatchMLP.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. This rewrite follows the paper's architecture and equations. The authors' repository is recorded as `reference-only` because it does not declare a repository license; its source was inspected at the pinned revision; no external source code was copied.
- The implementation covers four-scale patch embedding and latent decomposition `X_s=AvgPool(X), X_r=X-X_s`; the noisy residual branch remains channel-independent while the smooth branch adds dot-product-style inter-variable mixing after its intra-variable temporal MLP. Both branches use residual normalization before horizon projection.
- The paper does not fully specify every initialization and dataset-specific setting; no numerical-reference comparison or reported-result claim is made.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/TangZ25,
  author       = {Peiwang Tang and
                  Weitai Zhang},
  editor       = {Toby Walsh and
                  Julie Shah and
                  Zico Kolter},
  title        = {Unlocking the Power of Patch: Patch-Based {MLP} for Long-Term Time
                  Series Forecasting},
  booktitle    = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
                  Conference on Innovative Applications of Artificial Intelligence,
                  Fifteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages        = {12640--12648},
  publisher    = {{AAAI} Press},
  year         = {2025},
  url          = {https://doi.org/10.1609/aaai.v39i12.33378},
  doi          = {10.1609/AAAI.V39I12.33378},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/TangZ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
