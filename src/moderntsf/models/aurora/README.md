---
name: "Aurora"
summary: "Aurora is a compact clean-room multimodal-guided forecaster with patch tokens, modality distillation, future conditions, prototype retrieval, and deterministic flow integration."
paper: "https://arxiv.org/abs/2509.22295"
paper_title: "Aurora: Towards Universal Generative Multimodal Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/decisionintelligence/Aurora"
revision: "a247760abbc9d17a861bc365c032368d317815f2"
license: "NOASSERTION"
---
# Aurora

The paper's Aurora is a pretrained generative multimodal foundation model. This repository provides a compact paper-structured rewrite for ordinary forecasting, not the released pretrained system.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.22295); title: Aurora: Towards Universal Generative Multimodal Time Series Forecasting; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/decisionintelligence/Aurora); revision: `a247760abbc9d17a861bc365c032368d317815f2`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Aurora.toml`](../../../../configs/models/Aurora.toml).

## Differences

Pinned source inspection: `aurora/modeling_aurora.py`, `aurora/prototype_retriever.py`, `aurora/flow_loss.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. The rewrite maps paper equations (1)--(25) to temporal patching,
spectral guidance, learnable-query modality distillation, guided temporal
attention, future-condition decoding, prototype retrieval, and velocity flow.

It does not bundle BERT, ViT, the pretraining corpus, pretrained weights, raw
text/image tokenizers, stochastic sampling, or zero-shot claims. Optional dense
modality embeddings replace raw encoders; the registered point output follows a
deterministic mean flow. Evidence is in `../../../verification/evidence/Aurora.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `patch_len=16`, `num_heads=4`, `num_distill_tokens=2`, `num_prototypes=8`, `flow_steps=2`, `dropout=0.1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Aurora: Towards Universal Generative Multimodal Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.22295

## Abstract
Cross-domain generalization is very important in Time Series Forecasting because similar historical information may lead to distinct future trends due to the domain-specific characteristics. Recent works focus on building unimodal time series foundation models and end-to-end multimodal supervised models. Since domain-specific knowledge is often contained in modalities like texts, the former lacks the explicit utilization of them, thus hindering the performance. The latter is tailored for end-to-end scenarios and does not support zero-shot inference for cross-domain scenarios. In this work, we introduce Aurora, a Multimodal Time Series Foundation Model, which supports multimodal inputs and zero-shot inference. Pretrained on Cross-domain Multimodal Time Series Corpus, Aurora can adaptively extract and focus on key domain knowledge contained in corresponding text or image modalities, thus possessing strong cross-domain generalization capability. Through tokenization, encoding, and distillation, Aurora can extract multimodal domain knowledge as guidance and then utilizes a Modality-Guided Multi-head Self-Attention to inject them into the modeling of temporal representations. In the decoding phase, the multimodal representations are used to generate the conditions and prototypes of future tokens, contributing to a novel Prototype-Guided Flow Matching for generative probabilistic forecasting. Comprehensive experiments on 5 well-recognized benchmarks, including TimeMMD, TSFM-Bench, ProbTS, TFB, and EPF, demonstrate the consistent state-of-the-art performance of Aurora on both unimodal and multimodal scenarios.

## Source and verification

Pinned source inspection: `aurora/modeling_aurora.py`, `aurora/prototype_retriever.py`, `aurora/flow_loss.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. The rewrite maps paper equations (1)--(25) to temporal patching,
spectral guidance, learnable-query modality distillation, guided temporal
attention, future-condition decoding, prototype retrieval, and velocity flow.

It does not bundle BERT, ViT, the pretraining corpus, pretrained weights, raw
text/image tokenizers, stochastic sampling, or zero-shot claims. Optional dense
modality embeddings replace raw encoders; the registered point output follows a
deterministic mean flow. Evidence is in `../../../verification/evidence/Aurora.json`.

## In ModernTSF
Default config: `configs/models/Aurora.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2509-22295,
  author       = {Xingjian Wu and
                  Jianxin Jin and
                  Wanghui Qiu and
                  Peng Chen and
                  Yang Shu and
                  Bin Yang and
                  Chenjuan Guo},
  title        = {Aurora: Towards Universal Generative Multimodal Time Series Forecasting},
  journal      = {CoRR},
  volume       = {abs/2509.22295},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2509.22295},
  doi          = {10.48550/ARXIV.2509.22295},
  eprinttype   = {arXiv},
  eprint       = {2509.22295},
  timestamp    = {Mon, 20 Oct 2025 21:29:14 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2509-22295.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
