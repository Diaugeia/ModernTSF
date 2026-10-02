---
name: "GTR"
summary: "GTR (Global Temporal Retriever) is a lightweight, plug-and-play module for multivariate time series forecasting that extends any host model's temporal receptive field beyond the immediate input window by maintaining an adaptive global temporal embedding of the full cycle and dynamically retrieving and aligning relevant long-range historical segments with the current input, fusing them via 2D convolution and residual connections."
paper: "https://arxiv.org/abs/2602.10847"
paper_title: "Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/macovaseas/GTR"
revision: "d94161906151bdc0c94f4d21c2b0ca356aeb3135"
license: "Apache-2.0"
---
# GTR

GTR (Global Temporal Retriever) is a lightweight, plug-and-play module for multivariate time series forecasting that extends any host model's temporal receptive field beyond the immediate input window by maintaining an adaptive global temporal embedding of the full cycle and dynamically retrieving and aligning relevant long-range historical segments with the current input, fusing them via 2D convolution and residual connections.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2602.10847); title: Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/macovaseas/GTR); revision: `d94161906151bdc0c94f4d21c2b0ca356aeb3135`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GTR.toml`](../../../../configs/models/GTR.toml).

## Differences

Pinned source inspection: `models/GTR.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This clean-room rewrite follows paper Eqs. (1)--(5): absolute indices retrieve
from a trainable full-cycle matrix, a temporal linear map aligns the reference,
a `[2, P+1]`-style 2D convolution mixes local and global rows, and a residual
returns the enhanced series. The following two-layer GELU MLP and RevIN match
the disclosed forecast path. The common interface uses a zero start because its
batch contract has no absolute sample index; callers with that information use
`forecast_at(..., start_index=...)`. The cycle memory is learned locally and is not an external
historical database. The reference-only project was inspected at the pinned revision; no external source code was copied.
Strict evidence is in `../../../verification/evidence/GTR.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `cycle_length=168`, `local_period=24`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2026-02)
- **arXiv**: https://arxiv.org/abs/2602.10847

## Abstract
Multivariate time series forecasting (MTSF) plays a vital role in numerous real-world applications, yet existing models remain constrained by their reliance on a limited historical context. This limitation prevents them from effectively capturing global periodic patterns that often span cycles significantly longer than the input horizon - despite such patterns carrying strong predictive signals. Naive solutions, such as extending the historical window, lead to severe drawbacks, including overfitting, prohibitive computational costs, and redundant information processing. To address these challenges, we introduce the Global Temporal Retriever (GTR), a lightweight and plug-and-play module designed to extend any forecasting model's temporal awareness beyond the immediate historical context. GTR maintains an adaptive global temporal embedding of the entire cycle and dynamically retrieves and aligns relevant global segments with the input sequence. By jointly modeling local and global dependencies through a 2D convolution and residual fusion, GTR effectively bridges short-term observations with long-term periodicity without altering the host model architecture. Extensive experiments on six real-world datasets demonstrate that GTR consistently delivers state-of-the-art performance across both short-term and long-term forecasting scenarios, while incurring minimal parameter and computational overhead. These results highlight GTR as an efficient and general solution for enhancing global periodicity modeling in MTSF tasks.

## Source and verification

Pinned source inspection: `models/GTR.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This clean-room rewrite follows paper Eqs. (1)--(5): absolute indices retrieve
from a trainable full-cycle matrix, a temporal linear map aligns the reference,
a `[2, P+1]`-style 2D convolution mixes local and global rows, and a residual
returns the enhanced series. The following two-layer GELU MLP and RevIN match
the disclosed forecast path. The common interface uses a zero start because its
batch contract has no absolute sample index; callers with that information use
`forecast_at(..., start_index=...)`. The cycle memory is learned locally and is not an external
historical database. The reference-only project was inspected at the pinned revision; no external source code was copied.
Strict evidence is in `../../../verification/evidence/GTR.json`.

## In ModernTSF
Default config: `configs/models/GTR.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2602-10847,
  author       = {Fanpu Cao and
                  Lu Dai and
                  Jindong Han and
                  Hui Xiong},
  title        = {Enhancing Multivariate Time Series Forecasting with Global Temporal
                  Retrieval},
  journal      = {CoRR},
  volume       = {abs/2602.10847},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2602.10847},
  doi          = {10.48550/ARXIV.2602.10847},
  eprinttype   = {arXiv},
  eprint       = {2602.10847},
  timestamp    = {Sun, 29 Mar 2026 14:37:55 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2602-10847.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
