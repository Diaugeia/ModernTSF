---
name: "TiRex"
summary: "TiRex is a decoder-only probabilistic forecaster based on xLSTM-style scalar recurrent memory. The local implementation uses value/missing-mask patches, missing future tokens, stacked scalar-memory blocks, and multi-patch quantile decoding."
paper: "https://arxiv.org/abs/2505.23719"
paper_title: "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/NX-AI/tirex"
revision: "2226da4c9fa298ff34ad5af05369851674d622e5"
license: "LicenseRef-NXAI-Community"
tagline: "Decoder-only xLSTM-style scalar-memory blocks over value/mask patch tokens with missing future tokens, quantile output."
tags: ["rnn", "probabilistic", "patching", "channel-independent", "normalization"]
composition: ["normalization=local:instance-standardization", "decomposition=none", "temporal=local:scalar-lstm-memory-blocks", "channel=local:channel-independent-shared-weights", "head=component:quantile_head+local:multi-patch-residual-projection", "loss=loss:quantile"]
---
# TiRex

## Key ideas

- Each channel's z-scored history becomes patch tokens of values plus an observation mask (`ResidualProjection`); zero-valued missing-mask tokens for future patches let the recurrent state roll forward across the horizon.
- `ScalarLSTMBlock` stacks RMSNorm, a stabilised exponential-gated `ScalarMemory` recurrence, and a feed-forward residual.
- Future tokens are projected to per-step features and `QuantileHead` outputs quantiles (default 0.1 to 0.9), so training must use the quantile loss.
- Pre-trained weights and optimized xLSTM kernels are not reproduced.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, quantiles]` quantile forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.23719); title: TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/NX-AI/tirex); revision: `2226da4c9fa298ff34ad5af05369851674d622e5`; license: `LicenseRef-NXAI-Community`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TiRex.toml`](../../../../configs/models/TiRex.toml).

## Differences

Pinned source inspection: `src/tirex/models/tirex.py`, `src/tirex/models/slstm/block.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This implementation is randomly initialized and does not reproduce the released pre-trained model, optimized xLSTM kernels, exact published scale, or training data/augmentations. It implements a stabilized scalar-memory recurrence from public xLSTM equations and uses the shared monotone quantile head to satisfy the repository's non-crossing output contract. CPM is exposed for training integration but is not applied during inference. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`quantile_head`](../_components/quantile_head/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `patch_len=16`, `num_layers=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.23719

## Abstract
In-context learning, the ability of large language models to perform tasks using only examples provided in the prompt, has recently been adapted for time series forecasting. This paradigm enables zero-shot prediction, where past values serve as context for forecasting future values, making powerful forecasting tools accessible to non-experts and increasing the performance when training data are scarce. Most existing zero-shot forecasting approaches rely on transformer architectures, which, despite their success in language, often fall short of expectations in time series forecasting, where recurrent models like LSTMs frequently have the edge. Conversely, while LSTMs are well-suited for time series modeling due to their state-tracking capabilities, they lack strong in-context learning abilities. We introduce TiRex that closes this gap by leveraging xLSTM, an enhanced LSTM with competitive in-context learning skills. Unlike transformers, state-space models, or parallelizable RNNs such as RWKV, TiRex retains state-tracking, a critical property for long-horizon forecasting. To further facilitate its state-tracking ability, we propose a training-time masking strategy called CPM. TiRex sets a new state of the art in zero-shot time series forecasting on the HuggingFace benchmarks GiftEval and Chronos-ZS, outperforming significantly larger models including TabPFN-TS (Prior Labs), Chronos Bolt (Amazon), TimesFM (Google), and Moirai (Salesforce) across both short- and long-term forecasts.

## Source and verification

Pinned source inspection: `src/tirex/models/tirex.py`, `src/tirex/models/slstm/block.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This implementation is randomly initialized and does not reproduce the released pre-trained model, optimized xLSTM kernels, exact published scale, or training data/augmentations. It implements a stabilized scalar-memory recurrence from public xLSTM equations and uses the shared monotone quantile head to satisfy the repository's non-crossing output contract. CPM is exposed for training integration but is not applied during inference. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## In TSFLab
Default config: `configs/models/TiRex.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2505-23719,
  author       = {Andreas Auer and
                  Patrick Podest and
                  Daniel Klotz and
                  Sebastian B{\"{o}}ck and
                  G{\"{u}}nter Klambauer and
                  Sepp Hochreiter},
  title        = {TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced
                  In-Context Learning},
  journal      = {CoRR},
  volume       = {abs/2505.23719},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2505.23719},
  doi          = {10.48550/ARXIV.2505.23719},
  eprinttype   = {arXiv},
  eprint       = {2505.23719},
  timestamp    = {Tue, 05 Aug 2025 22:46:04 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2505-23719.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
