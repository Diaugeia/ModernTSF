---
name: "CATS"
summary: "CATS (Cross-Attention-only Time Series transformer) is a multivariate time series forecasting model that eliminates self-attention entirely from the Transformer architecture and relies solely on cross-attention mechanisms, using future horizon-dependent parameters as queries with enhanced parameter sharing to improve long-term forecasting accuracy while reducing parameter count and memory usage."
paper: "https://openreview.net/forum?id=iN43sJoib7"
paper_title: "Are Self-Attentions Effective for Time Series Forecasting?"
venue: "NeurIPS 2024"
year: 2024
code: "https://github.com/dongbeank/CATS"
revision: "58854fc759d608ce400f378be83f4513960e505d"
license: "MIT"
tagline: "Cross-attention-only Transformer: learned future-patch queries attend history patches, with query-adaptive masking."
tags: ["transformer", "patching", "attention-variant", "channel-independent", "parameter-sharing"]
composition: ["normalization=local:last-value-centering", "decomposition=none", "temporal=local:cross-attention-only-decoder", "channel=local:channel-independent-shared-weights", "head=local:linear-patch-projection", "loss=loss:mse"]
---
# CATS

## Key ideas

- No self-attention at all: `CrossAttentionLayer` has learned `future_queries` (one per output patch) attend the embedded history patches.
- The same embedding, attention and projection weights serve every horizon patch and every channel (`query_independence=False` shares queries across channels).
- Query-adaptive masking randomly drops attention outputs in training with probability rising linearly from `QAM_start` to `QAM_end` across layers.
- The series is centered on its last value (not detached, so gradients flow through the level) and the level is added back after the linear patch projection.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=iN43sJoib7); title: Are Self-Attentions Effective for Time Series Forecasting?; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/dongbeank/CATS); revision: `58854fc759d608ce400f378be83f4513960e505d`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CATS.toml`](../../../../configs/models/CATS.toml).

## Differences

**Paper-driven local implementation.** Historical patches provide keys and
values; learned future-patch parameters provide the only queries. Every layer
uses cross-attention without self-attention, shares embedding/attention/output
parameters across horizons, and applies the paper's query-adaptive stochastic
mask to the attention residual during training. The external repository is
reference-only; no source file was copied or adapted.

**Component-extraction note.** CATS centers on `x_enc[:, -1:, :]` without
`.detach()`, unlike the `last_value_center` component (used by NLinear,
SegRNN, CrossGNN), so gradients flow through the additive level term back
into the encoder; this differs from the component's fixed detach contract
and the centering therefore stays model-local.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=8`, `patch_len=24`, `d_model=128`, `n_heads=16`, `d_ff=256`, `n_layers=3`, `dropout=0.1`, `stride=24`, `attn_dropout=0.0`, `query_independence=False`, `store_attn=False`, `QAM_start=0.1`, `QAM_end=0.5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Are Self-Attentions Effective for Time Series Forecasting?
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.16877

## Abstract
Time series forecasting is crucial for applications across multiple domains and various scenarios. Although Transformer models have dramatically advanced the landscape of forecasting, their effectiveness remains debated. Recent findings have indicated that simpler linear models might outperform complex Transformer-based approaches, highlighting the potential for more streamlined architectures. In this paper, we shift the focus from evaluating the overall Transformer architecture to specifically examining the effectiveness of self-attention for time series forecasting. To this end, we introduce a new architecture, Cross-Attention-only Time Series transformer (CATS), that rethinks the traditional Transformer framework by eliminating self-attention and leveraging cross-attention mechanisms instead. By establishing future horizon-dependent parameters as queries and enhanced parameter sharing, our model not only improves long-term forecasting accuracy but also reduces the number of parameters and memory usage. Extensive experiment across various datasets demonstrates that our model achieves superior performance with the lowest mean squared error and uses fewer parameters compared to existing models.

## In TSFLab
Default config: `configs/models/CATS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Paper-driven local implementation.** Historical patches provide keys and
values; learned future-patch parameters provide the only queries. Every layer
uses cross-attention without self-attention, shares embedding/attention/output
parameters across horizons, and applies the paper's query-adaptive stochastic
mask to the attention residual during training. The external repository is
reference-only; no source file was copied or adapted.

**Component-extraction note.** CATS centers on `x_enc[:, -1:, :]` without
`.detach()`, unlike the `last_value_center` component (used by NLinear,
SegRNN, CrossGNN), so gradients flow through the additive level term back
into the encoder; this differs from the component's fixed detach contract
and the centering therefore stays model-local.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/Kim00K24,
  author       = {Dongbin Kim and
                  Jinseong Park and
                  Jaewook Lee and
                  Hoki Kim},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {Are Self-Attentions Effective for Time Series Forecasting?},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/Kim00K24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
