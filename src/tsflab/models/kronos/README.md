---
name: "Kronos"
summary: "Kronos is a decoder-only foundation model pre-trained on over 12 billion financial candlestick (K-line) records from 45 global exchanges, covering tasks including price-series forecasting, volatility prediction, and synthetic market-data generation. Its defining design discretizes each multivariate record with Binary Spherical Quantization into coarse and fine subtokens, then predicts those subtokens sequentially with a causal Transformer."
paper: "https://arxiv.org/abs/2508.02739"
paper_title: "Kronos: A Foundation Model for the Language of Financial Markets"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/shiyu-coder/Kronos"
revision: "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
license: "MIT"
---
# Kronos

Kronos is a decoder-only foundation model pre-trained on over 12 billion financial candlestick (K-line) records from 45 global exchanges, covering tasks including price-series forecasting, volatility prediction, and synthetic market-data generation. Its defining design discretizes each multivariate record with Binary Spherical Quantization into coarse and fine subtokens, then predicts those subtokens sequentially with a causal Transformer.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2508.02739); title: Kronos: A Foundation Model for the Language of Financial Markets; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/shiyu-coder/Kronos); revision: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Kronos.toml`](../../../../configs/models/Kronos.toml).

## Differences

Pinned source inspection: `model/kronos.py`, `model/module.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This compact clean-room rewrite implements the paper's defining hierarchy:
straight-through BSQ, equal coarse/fine binary subtokens, fused subtoken
embeddings, causal decoding, coarse prediction, and fine prediction conditioned
on a differentiable expected coarse code (Eqs. (2)--(8)). It is trained from scratch;
it does not include the authors' 12-billion-record corpus or pretrained weights.
The local tokenizer is an affine encoder/decoder rather than the paper's large
Transformer autoencoder, and the default eight-bit vocabulary is smaller than
the reported twenty-bit setup. It therefore exposes the architecture for local
experiments but is not a zero-shot Kronos checkpoint. The reference-only source
was inspected at the pinned revision; no external source code was copied. Evidence is in `../../../verification/evidence/Kronos.json`.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `code_bits=8`, `num_layers=2`, `num_heads=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Kronos: A Foundation Model for the Language of Financial Markets
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-08)
- **arXiv**: https://arxiv.org/abs/2508.02739

## Abstract
The success of large-scale pre-training paradigm, exemplified by Large Language Models (LLMs), has inspired the development of Time Series Foundation Models (TSFMs). However, their application to financial candlestick (K-line) data remains limited, often underperforming non-pre-trained architectures. Moreover, existing TSFMs often overlook crucial downstream tasks such as volatility prediction and synthetic data generation. To address these limitations, we propose Kronos, a unified, scalable pre-training framework tailored to financial K-line modeling. Kronos introduces a specialized tokenizer that discretizes continuous market information into token sequences, preserving both price dynamics and trade activity patterns. We pre-train Kronos using an autoregressive objective on a massive, multi-market corpus of over 12 billion K-line records from 45 global exchanges, enabling it to learn nuanced temporal and cross-asset representations. Kronos excels in a zero-shot setting across a diverse set of financial tasks. On benchmark datasets, Kronos boosts price series forecasting RankIC by 93% over the leading TSFM and 87% over the best non-pre-trained baseline. It also achieves a 9% lower MAE in volatility forecasting and a 22% improvement in generative fidelity for synthetic K-line sequences. These results establish Kronos as a robust, versatile foundation model for end-to-end financial time series analysis.

## Source and verification

Pinned source inspection: `model/kronos.py`, `model/module.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This compact clean-room rewrite implements the paper's defining hierarchy:
straight-through BSQ, equal coarse/fine binary subtokens, fused subtoken
embeddings, causal decoding, coarse prediction, and fine prediction conditioned
on a differentiable expected coarse code (Eqs. (2)--(8)). It is trained from scratch;
it does not include the authors' 12-billion-record corpus or pretrained weights.
The local tokenizer is an affine encoder/decoder rather than the paper's large
Transformer autoencoder, and the default eight-bit vocabulary is smaller than
the reported twenty-bit setup. It therefore exposes the architecture for local
experiments but is not a zero-shot Kronos checkpoint. The reference-only source
was inspected at the pinned revision; no external source code was copied. Evidence is in `../../../verification/evidence/Kronos.json`.

## In TSFLab
Default config: `configs/models/Kronos.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ShiFCZXZL26,
  author       = {Yu Shi and
                  Zongliang Fu and
                  Shuo Chen and
                  Bohan Zhao and
                  Wei Xu and
                  Changshui Zhang and
                  Jian Li},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Kronos: {A} Foundation Model for the Language of Financial Markets},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25366--25373},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39730},
  doi          = {10.1609/AAAI.V40I30.39730},
  timestamp    = {Wed, 03 Jun 2026 10:10:49 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/ShiFCZXZL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
