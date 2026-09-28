---
name: "TQNet"
summary: "TQNet turns CycleNet's per-variable learnable recurrent cycle into a cross-variable temporal query: a phase-aligned window of a learnable periodic table serves as the query of a single attention layer whose keys and values are the raw instance-normalized lookback window. This fuses a global periodic prior with local per-sample observations before a lightweight MLP produces the forecast."
paper: "https://arxiv.org/abs/2505.12917"
paper_title: "Temporal Query Network for Efficient Multivariate Time Series Forecasting"
venue: "ICML 2025"
year: 2025
code: "https://github.com/ACAT-SCUT/TQNet"
revision: "15e19cb23483ed52398566c4baa959168cfffa57"
license: "Apache-2.0"

---
# TQNet

<!-- model-card:canonical:start -->
## Method overview

TQNet turns CycleNet's per-variable learnable recurrent cycle into a cross-variable temporal query: a phase-aligned window of a learnable periodic table serves as the query of a single attention layer whose keys and values are the raw instance-normalized lookback window.

## Core architecture

This fuses a global periodic prior with local per-sample observations before a lightweight MLP produces the forecast.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.12917); title: Temporal Query Network for Efficient Multivariate Time Series Forecasting; venue/year: ICML 2025 / 2025
- [codebase](https://github.com/ACAT-SCUT/TQNet); revision: `15e19cb23483ed52398566c4baa959168cfffa57`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TQNet.toml`](../../../../configs/models/TQNet.toml).

## Differences

The official implementation (`models/TQNet.py`,
`data_provider/data_loader.py`) was inspected at the pinned revision to
resolve the exact phase convention: `cycle_index` is the phase of the
*first forecast step* (`cycle_index = (arange(len(data)) % cycle)[s_end]`),
and the query window is gathered by advancing that phase for `seq_len`
steps. This implementation derives an equivalent phase directly from the
four-input `x_mark_enc` contract, the same convention CycleNet already uses
in this catalog, instead of a bespoke data loader. The official
`nn.MultiheadAttention` dropout is a hardcoded `0.5` regardless of the
model's other dropout setting; this is exposed here as a separate
`attn_dropout` parameter (default `0.5`) rather than hidden. The official
ablation switches `use_tq` and `channel_aggre` (which can disable the
temporal query or the attention entirely) are not reproduced; this
implementation always runs the full TQNet configuration. No source file was
copied or adapted.

## Shared components

- [`periodic_query_bank`](../_components/periodic_query_bank/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `cycle=24`, `d_model=64`, `dropout=0.1`, `attn_dropout=0.5`, `channel_aggre_heads=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper

TQNet is the successor to CycleNet (arXiv:2409.18479). It reuses CycleNet's
periodically shifted learnable vectors as the *query* of a cross-variable
attention layer whose keys and values come from the raw lookback window, so
the model captures global inter-variable correlations while still grounding
the forecast in local, per-sample observations (paper Section 3, Figure 2).

## Source and verification

The official implementation (`models/TQNet.py`,
`data_provider/data_loader.py`) was inspected at the pinned revision to
resolve the exact phase convention: `cycle_index` is the phase of the
*first forecast step* (`cycle_index = (arange(len(data)) % cycle)[s_end]`),
and the query window is gathered by advancing that phase for `seq_len`
steps. This implementation derives an equivalent phase directly from the
four-input `x_mark_enc` contract, the same convention CycleNet already uses
in this catalog, instead of a bespoke data loader. The official
`nn.MultiheadAttention` dropout is a hardcoded `0.5` regardless of the
model's other dropout setting; this is exposed here as a separate
`attn_dropout` parameter (default `0.5`) rather than hidden. The official
ablation switches `use_tq` and `channel_aggre` (which can disable the
temporal query or the attention entirely) are not reproduced; this
implementation always runs the full TQNet configuration. No source file was
copied or adapted.

## Citation

```bibtex
@inproceedings{lin2025tqnet,
  title={Temporal Query Network for Efficient Multivariate Time Series Forecasting},
  author={Lin, Shengsheng and Chen, Haojun and Wu, Haijie and Qiu, Chunyun and Lin, Weiwei},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
