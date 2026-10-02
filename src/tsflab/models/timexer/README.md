---
name: "TimeXer"
summary: "TimeXer is a Transformer-based time series forecasting model for the standard time series forecasting setting that extends canonical Transformers to handle exogenous variables. It introduces deftly designed embedding layers that separately represent endogenous (target) variables via patch-wise self-attention and exogenous (external) variables via variate-wise cross-attention, with learned global endogenous tokens bridging causal information from exogenous series into endogenous temporal patches."
paper: "https://arxiv.org/abs/2402.19072"
paper_title: "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
venue: "NeurIPS 2024"
year: 2024
code: "https://github.com/thuml/TimeXer"
revision: "76011909357972bd55a27adba2e1be994d81b327"
license: "NOASSERTION"
---
# TimeXer

TimeXer is a Transformer-based time series forecasting model for the standard time series forecasting setting that extends canonical Transformers to handle exogenous variables. It introduces deftly designed embedding layers that separately represent endogenous (target) variables via patch-wise self-attention and exogenous (external) variables via variate-wise cross-attention, with learned global endogenous tokens bridging causal information from exogenous series into endogenous temporal patches.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2402.19072); title: TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/thuml/TimeXer); revision: `76011909357972bd55a27adba2e1be994d81b327`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeXer.toml`](../../../../configs/models/TimeXer.toml).

## Differences

Clean-room implementation: confirmed. The implementation was derived independently from the paper's endogenous patch self-attention, global endogenous bridge token, and variate-wise exogenous cross-attention; source from the unlicensed reference repository was not copied or reused. `MS`/`S` designate the last channel as endogenous; `M` vectorizes each channel as an endogenous target against the shared external context.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `patch_len=16`, `dropout=0.1`, `activation='gelu'`, `use_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.19072

## Abstract
Deep models have demonstrated remarkable performance in time series forecasting. However, due to the partially-observed nature of real-world applications, solely focusing on the target of interest, so-called endogenous variables, is usually insufficient to guarantee accurate forecasting. Notably, a system is often recorded into multiple variables, where the exogenous variables can provide valuable external information for endogenous variables. Thus, unlike well-established multivariate or univariate forecasting paradigms that either treat all the variables equally or ignore exogenous information, this paper focuses on a more practical setting: time series forecasting with exogenous variables. We propose a novel approach, TimeXer, to ingest external information to enhance the forecasting of endogenous variables. With deftly designed embedding layers, TimeXer empowers the canonical Transformer with the ability to reconcile endogenous and exogenous information, where patch-wise self-attention and variate-wise cross-attention are used simultaneously. Moreover, global endogenous tokens are learned to effectively bridge the causal information underlying exogenous series into endogenous temporal patches. Experimentally, TimeXer achieves consistent state-of-the-art performance on twelve real-world forecasting benchmarks and exhibits notable generality and scalability. Code is available at this repository: https://github.com/thuml/TimeXer.

## In TSFLab
Default config: `configs/models/TimeXer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. The implementation was derived independently from the paper's endogenous patch self-attention, global endogenous bridge token, and variate-wise exogenous cross-attention; source from the unlicensed reference repository was not copied or reused. `MS`/`S` designate the last channel as endogenous; `M` vectorizes each channel as an endogenous target against the shared external context.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/WangWDQZLQWL24,
  author       = {Yuxuan Wang and
                  Haixu Wu and
                  Jiaxiang Dong and
                  Guo Qin and
                  Haoran Zhang and
                  Yong Liu and
                  Yunzhong Qiu and
                  Jianmin Wang and
                  Mingsheng Long},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {TimeXer: Empowering Transformers for Time Series Forecasting with
                  Exogenous Variables},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/WangWDQZLQWL24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
