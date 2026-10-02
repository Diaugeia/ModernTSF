---
name: "TSRAG"
summary: "TSRAG (TS-RAG) is a retrieval-augmented generation framework for zero-shot time-series forecasting built on top of pre-trained Time Series Foundation Models (TSFMs). It uses a pre-trained time-series encoder to retrieve semantically relevant segments from a dedicated knowledge base and then fuses them with the TSFM's internal representations via a learnable Adaptive Retrieval Mixer (ARM) module — enhancing generalisation and interpretability without requiring task-specific fine-tuning. The model targets the standard multivariate time-series forecasting setting."
paper: "https://arxiv.org/abs/2503.07649"
paper_title: "TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/UConn-DSIS/TS-RAG"
revision: "73ac807789d2e61b8a3dfc8514e3fc947fe185cc"
license: "MIT"
tagline: "Top-k retrieval of future segments from a knowledge base, fused with the query by an adaptive retrieval mixer."
tags: ["mlp", "retrieval", "attention-variant", "channel-mixing", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:flatten-mlp-query-backbone", "channel=local:flattened-all-channel-embedding", "head=local:linear-forecast-projection", "loss=loss:mse"]
---
# TSRAG

## Key ideas

- `retrieve` embeds the query and candidate contexts by parameter-free adaptive pooling and takes the top-k nearest by Euclidean distance; `retrieved_projector` embeds their future windows.
- `AdaptiveRetrievalMixer` runs self-attention plus an FFN over the query and retrieved items, then softmax-weights them and adds the result to the query as a skip.
- `forecast_with_retrieval` accepts an external knowledge base (`retrieval_contexts`, `retrieval_futures`); `forward` uses a history-derived fallback of shifted windows, and there is no TSFM backbone or pre-built database in this implementation.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2503.07649); title: TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/UConn-DSIS/TS-RAG); revision: `73ac807789d2e61b8a3dfc8514e3fc947fe185cc`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TSRAG.toml`](../../../../configs/models/TSRAG.toml).

## Differences

Pinned source inspection: `TS-RAG/retrieve.py`, `TS-RAG/models/ChronosBolt.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

The local model has no third-party TSFM checkpoint, Chronos retrieval encoder, FAISS index, or pre-built multi-domain knowledge base. `forecast_with_retrieval` accepts explicit `retrieval_contexts`/`retrieval_futures`; common `forward` uses a documented deterministic history-derived fallback so the standalone repository contract remains runnable. Retrieval descriptors use parameter-free adaptive pooling, while ARM is learned. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `top_k=4`, `memory_size=8`, `num_heads=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-03)
- **arXiv**: https://arxiv.org/abs/2503.07649

## Abstract
Large Language Models (LLMs) and Foundation Models (FMs) have recently become prevalent for time series forecasting tasks. While fine-tuning LLMs enables domain adaptation, they often struggle to generalize across diverse and unseen datasets. Moreover, existing Time Series Foundation Models (TSFMs) still face challenges in handling non-stationary dynamics and distribution shifts, largely due to the lack of effective mechanisms for adaptation. To this end, we present TS-RAG, a retrieval-augmented generation framework for time series forecasting that enhances the generalization and interpretability of TSFMs. Specifically, TS-RAG leverages pre-trained time series encoders to retrieve semantically relevant segments from a dedicated knowledge base, enriching the contextual representation of the input query. Furthermore, we propose an Adaptive Retrieval Mixer (ARM) module that dynamically fuses the retrieved patterns with the TSFM's internal representation, improving forecasting accuracy without requiring task-specific fine-tuning. Thorough empirical studies on seven public benchmark datasets demonstrate that TS-RAG achieves state-of-the-art zero-shot forecasting performance, outperforming the existing TSFMs by up to 6.84% across diverse domains while also providing desirable interpretability.

## Source and verification

Pinned source inspection: `TS-RAG/retrieve.py`, `TS-RAG/models/ChronosBolt.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

The local model has no third-party TSFM checkpoint, Chronos retrieval encoder, FAISS index, or pre-built multi-domain knowledge base. `forecast_with_retrieval` accepts explicit `retrieval_contexts`/`retrieval_futures`; common `forward` uses a documented deterministic history-derived fallback so the standalone repository contract remains runnable. Retrieval descriptors use parameter-free adaptive pooling, while ARM is learned. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## In TSFLab
Default config: `configs/models/TSRAG.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2503-07649,
  author       = {Kanghui Ning and
                  Zijie Pan and
                  Yu Liu and
                  Yushan Jiang and
                  James Y. Zhang and
                  Kashif Rasul and
                  Anderson Schneider and
                  Lintao Ma and
                  Yuriy Nevmyvaka and
                  Dongjin Song},
  title        = {{TS-RAG:} Retrieval-Augmented Generation based Time Series Foundation
                  Models are Stronger Zero-Shot Forecaster},
  journal      = {CoRR},
  volume       = {abs/2503.07649},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2503.07649},
  doi          = {10.48550/ARXIV.2503.07649},
  eprinttype   = {arXiv},
  eprint       = {2503.07649},
  timestamp    = {Sun, 13 Apr 2025 20:58:54 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2503-07649.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
