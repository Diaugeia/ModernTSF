---
name: "PGN"
summary: "PGN (as the TPGN forecaster) reshapes the instance-normalized history into period-length columns and rows, runs a Parallel Gated Network along the rows (a linear summary of all earlier rows gated against a candidate in one step) and a patch-then-global short-term branch, and maps the concatenated representations to the future rows with one linear layer per variate."
paper: "https://arxiv.org/abs/2409.17703"
paper_title: "PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting"
venue: "NeurIPS 2024"
year: 2024
tagline: "Period-folded series; a parallel gated network over rows plus a short-term global branch; per-variate linear head."
tags: ["rnn", "gating", "periodicity", "channel-independent", "normalization", "calendar-features", "lightweight"]
composition: ["normalization=component:revin", "decomposition=local:period-2d-reshape", "temporal=local:parallel-gated-network", "channel=local:per-variate-parameters", "head=local:long-short-branch-linear-head", "loss=loss:mse"]
---
# PGN

## Key ideas

- `ParallelGatedNetwork` is the paper's PGN: a linear Historical Information Extraction layer summarizes every strictly earlier row in parallel (zero padded in front), and one sigmoid gate mixes that summary with a tanh candidate, `Out = G*H + (1-G)*H^` (Eq. 1).
- `Model` is TPGN: the series is folded into `seq_len / period` rows of `period` columns with the value channel and four hourly calendar features per step; the long-term branch applies PGN along rows then pools the rows of each column, while the short-term branch pools each row's columns to patches and the patches to one global vector repeated over the period.
- One linear head per variate maps `[global, long]` (the official concatenation order) to the future rows, which are unfolded to `pred_len` (Eq. 5); every learned map has independent parameters per variate (`VariateLinear`).
- Instance mean/variance normalization is the shared `revin` component without affine terms; calendar marks are adapted by `marks.adapt_tslib_marks`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2409.17703); title: PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting; venue/year: NeurIPS 2024 / 2024
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PGN.toml`](../../../../configs/models/PGN.toml).

## Differences

- Implementation: independent rewrite from Sections 3.1 and 3.2 (Eqs. 1 to 5) of the paper. The official repository (`Water2sea/TPGN`, revision `4bbe2bee016094acd4299b2ad6fcd1db386bc99f`) has no license, so it is reference only: `models/TPGN.py` was read to resolve tensor layouts and omissions and nothing was copied or imported; its code, license and revision are therefore not recorded as source facts of this entry.
- Resolved from the reference code: the HIE kernel spans `rows - 1` earlier rows and excludes the current row; the gate and candidate come from one linear layer over `[value, calendar, H]` and are split in half; all linear maps are grouped per variate; the short-term branch feeds each row's columns through the row map and then a column map before repeating over the period; the instance normalization uses the biased variance plus `1e-5` inside the square root (the same statistics as `revin` with `affine=False`; the official code lets the gradient flow through the standard deviation while `revin` detaches both statistics).
- Differences from the paper and official code: `seq_len` and `pred_len` must be multiples of `period` (the official code rounds the number of forecast rows up and its experiment loop keeps the trailing `pred_len` outputs; that case is rejected here instead of misaligning the horizon). Calendar features are the four hourly Time-Series-Library features built from the runner's raw marks (`freq="h"` only; other frequencies need preprocessed four-wide marks); absent marks are replaced by zeros. The official code also accepts minutely (`t`, five features) and daily (`d`, three features) calendars. `d_model=64` and `period=24` are preset values, not the per-dataset searched hyperparameters of the paper, and the TPGN-GRU/LSTM/MLP variants and the PGN-only (no short branch) result are limited to the `use_short_branch` switch. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure and equation tests in `tests/test_pgn.py` (strictly causal HIE, gate equation, row independence, branch widths, gradients).

## Shared components

- [`marks`](../_components/marks/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `period=24`, `d_model=64`, `norm=True`, `use_short_branch=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-09)
- **arXiv**: https://arxiv.org/abs/2409.17703

## Source and verification

- Implementation: independent rewrite from Sections 3.1 and 3.2 (Eqs. 1 to 5) of the paper. The official repository (`Water2sea/TPGN`, revision `4bbe2bee016094acd4299b2ad6fcd1db386bc99f`) has no license, so it is reference only: `models/TPGN.py` was read to resolve tensor layouts and omissions and nothing was copied or imported; its code, license and revision are therefore not recorded as source facts of this entry.
- Resolved from the reference code: the HIE kernel spans `rows - 1` earlier rows and excludes the current row; the gate and candidate come from one linear layer over `[value, calendar, H]` and are split in half; all linear maps are grouped per variate; the short-term branch feeds each row's columns through the row map and then a column map before repeating over the period; the instance normalization uses the biased variance plus `1e-5` inside the square root (the same statistics as `revin` with `affine=False`; the official code lets the gradient flow through the standard deviation while `revin` detaches both statistics).
- Differences from the paper and official code: `seq_len` and `pred_len` must be multiples of `period` (the official code rounds the number of forecast rows up and its experiment loop keeps the trailing `pred_len` outputs; that case is rejected here instead of misaligning the horizon). Calendar features are the four hourly Time-Series-Library features built from the runner's raw marks (`freq="h"` only; other frequencies need preprocessed four-wide marks); absent marks are replaced by zeros. The official code also accepts minutely (`t`, five features) and daily (`d`, three features) calendars. `d_model=64` and `period=24` are preset values, not the per-dataset searched hyperparameters of the paper, and the TPGN-GRU/LSTM/MLP variants and the PGN-only (no short branch) result are limited to the `use_short_branch` switch. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure and equation tests in `tests/test_pgn.py` (strictly causal HIE, gate equation, row independence, branch widths, gradients).

## Citation

```bibtex
@inproceedings{jia2024pgn,
  title     = {PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting},
  author    = {Jia, Yuxin and Lin, Youfang and Yu, Jing and Wang, Shuo and Liu, Tianhao and Wan, Huaiyu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
