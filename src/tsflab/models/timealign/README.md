---
name: "TimeAlign"
summary: "TimeAlign is a lightweight, plug-and-play framework for time series forecasting that aligns past and future representations to bridge the distributional gap between historical inputs and future targets. It establishes a new representation paradigm by aligning auxiliary features via a reconstruction task and feeding them back into any base forecaster, with gains arising primarily from correcting frequency mismatches between historical inputs and future outputs."
paper: "https://arxiv.org/abs/2509.14181"
paper_title: "Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/TROUBADOUR000/TimeAlign"
revision: "ab2dff5bde250f82e29d8755f87a494921857d71"
license: "NOASSERTION"
---
# TimeAlign

TimeAlign is a lightweight, plug-and-play framework for time series forecasting that aligns past and future representations to bridge the distributional gap between historical inputs and future targets. It establishes a new representation paradigm by aligning auxiliary features via a reconstruction task and feeding them back into any base forecaster, with gains arising primarily from correcting frequency mismatches between historical inputs and future outputs.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.14181); title: Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/TROUBADOUR000/TimeAlign); revision: `ab2dff5bde250f82e29d8755f87a494921857d71`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeAlign.toml`](../../../../configs/models/TimeAlign.toml).

## Differences

Clean-room implementation: confirmed. Paper mapping: simple patched encoders → `PatchMLPBranch`; local/global alignment → `DistributionAlignment`; prediction, reconstruction and alignment objective → `ModelSpec.training_objective`. The unlicensed repository is link-only and its source code was not copied. Alternative backbones, dataset recipes and numerical reference comparison are not reproduced.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_num=4`, `d_model=32`, `d_ff=32`, `e_layers=2`, `dropout=0.1`, `pos=True`, `layer_norm=True`, `loc=True`, `glo=True`, `local_margin=0.0`, `global_margin=0.0`, `w_recon=1.0`, `w_align=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.14181

## Abstract
Although contrastive and other representation-learning methods have long been explored in vision and NLP, their adoption in modern time series forecasters remains limited. We believe they hold strong promise for this domain. To unlock this potential, we explicitly align past and future representations, thereby bridging the distributional gap between input histories and future targets. To this end, we introduce TimeAlign, a lightweight, plug-and-play framework that establishes a new representation paradigm, distinct from contrastive learning, by aligning auxiliary features via a simple reconstruction task and feeding them back into any base forecaster. Extensive experiments across eight benchmarks verify its superior performance. Further studies indicate that the gains arise primarily from correcting frequency mismatches between historical inputs and future outputs. Additionally, we provide two theoretical justifications for how reconstruction improves forecasting generalization and how alignment increases the mutual information between learned representations and predicted targets. The code is available at https://github.com/TROUBADOUR000/TimeAlign.

## In TSFLab
Default config: `configs/models/TimeAlign.toml`; model specification: `spec.py`; implementation: `model.py`.

The independent implementation uses a patch-MLP predictor and a training-only
future reconstruction branch. The 3-term training objective
`L = L_pred + w_recon·L_recon + w_align·L_align` needs the future `Y`, so the
implementation registers its full paper loss through
`ModelSpec.training_objective`; validation and early stopping continue to use
the configured observation criterion.

Key params: `patch_num` (**must divide both `seq_len` and `pred_len`**),
`d_model`, `d_ff`, `e_layers`, `dropout`, `pos`, `layer_norm`, `loc`/`glo`
(local/global alignment toggles), `local_margin`/`global_margin`,
`w_recon`/`w_align`. Verify with `uv run tsf smoke --model TimeAlign`.

## Source and verification

Clean-room implementation: confirmed. Paper mapping: simple patched encoders → `PatchMLPBranch`; local/global alignment → `DistributionAlignment`; prediction, reconstruction and alignment objective → `ModelSpec.training_objective`. The unlicensed repository is link-only and its source code was not copied. Alternative backbones, dataset recipes and numerical reference comparison are not reproduced.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2509-14181,
  author       = {Yifan Hu and
                  Jie Yang and
                  Tian Zhou and
                  Peiyuan Liu and
                  Yujin Tang and
                  Rong Jin and
                  Liang Sun},
  title        = {Bridging Past and Future: Distribution-Aware Alignment for Time Series
                  Forecasting},
  journal      = {CoRR},
  volume       = {abs/2509.14181},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2509.14181},
  doi          = {10.48550/ARXIV.2509.14181},
  eprinttype   = {arXiv},
  eprint       = {2509.14181},
  timestamp    = {Wed, 25 Feb 2026 08:13:51 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2509-14181.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
