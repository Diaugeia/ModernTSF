---
name: "PHAT"
summary: "PHAT (Period Heterogeneity-Aware Transformer) is a Transformer-based model for multivariate time series forecasting that explicitly models periodic heterogeneity — the fact that different variables exhibit distinct and dynamically changing periods. It organises inputs into a three-dimensional periodic bucket tensor and applies a positive-negative attention mechanism to capture both periodic alignment and periodic deviation."
paper: "https://arxiv.org/abs/2602.00654"
paper_title: "PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/PoorOtterBob/PHAT"
revision: "313987b52b5fc8184efba7fb9c8b5707c6f03448"
license: "MIT"
---
# PHAT

PHAT (Period Heterogeneity-Aware Transformer) is a Transformer-based model for multivariate time series forecasting that explicitly models periodic heterogeneity — the fact that different variables exhibit distinct and dynamically changing periods. It organises inputs into a three-dimensional periodic bucket tensor and applies a positive-negative attention mechanism to capture both periodic alignment and periodic deviation.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2602.00654); title: PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/PoorOtterBob/PHAT); revision: `313987b52b5fc8184efba7fb9c8b5707c6f03448`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PHAT.toml`](../../../../configs/models/PHAT.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independent clean-room implementation from paper equations (4)-(13); the
  repository is retained only as a pinned reference and no source was copied.
- Special zero-period bucket, dataset-specific periods, and published-result
  reference comparison are not included.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=6`, `d_model=64`, `n_heads=8`, `d_layers=1`, `attn_dropout=0.1`, `ffn_dropout=0.1`, `ffn_expand_ratio=2.66667`, `period_topk=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2026-02)
- **arXiv**: https://arxiv.org/abs/2602.00654

## Abstract
While existing multivariate time series forecasting models have advanced significantly in modeling periodicity, they largely neglect the periodic heterogeneity common in real-world data, where variables exhibit distinct and dynamically changing periods. To effectively capture this periodic heterogeneity, we propose PHAT (Period Heterogeneity-Aware Transformer). Specifically, PHAT arranges multivariate inputs into a three-dimensional "periodic bucket" tensor, where the dimensions correspond to variable group characteristics with similar periodicity, time steps aligned by phase, and offsets within the period. By restricting interactions within buckets and masking cross-bucket connections, PHAT effectively avoids interference from inconsistent periods. We also propose a positive-negative attention mechanism, which captures periodic dependencies from two perspectives: periodic alignment and periodic deviation. Additionally, the periodic alignment attention scores are decomposed into positive and negative components, with a modulation term encoding periodic priors. This modulation constrains the attention mechanism to more faithfully reflect the underlying periodic trends. A mathematical explanation is provided to support this property. We evaluate PHAT comprehensively on 14 real-world datasets against 18 baselines, and the results show that it significantly outperforms existing methods, achieving highly competitive forecasting performance.

## In TSFLab
Default config: `configs/models/PHAT.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independent clean-room implementation from paper equations (4)-(13); the
  repository is retained only as a pinned reference and no source was copied.
- Special zero-period bucket, dataset-specific periods, and published-result
  reference comparison are not included.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2602-00654,
  author       = {Jiaming Ma and
                  Qihe Huang and
                  Haofeng Ma and
                  Guanjun Wang and
                  Sheng Huang and
                  Zhengyang Zhou and
                  Pengkun Wang and
                  Binwu Wang and
                  Yang Wang},
  title        = {{PHAT:} Modeling Period Heterogeneity for Multivariate Time Series
                  Forecasting},
  journal      = {CoRR},
  volume       = {abs/2602.00654},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2602.00654},
  doi          = {10.48550/ARXIV.2602.00654},
  eprinttype   = {arXiv},
  eprint       = {2602.00654},
  timestamp    = {Sat, 14 Mar 2026 17:13:45 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2602-00654.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
