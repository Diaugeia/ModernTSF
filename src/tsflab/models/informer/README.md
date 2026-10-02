---
name: "Informer"
summary: "Informer is a Transformer-based model for long-sequence time-series forecasting in the standard univariate and multivariate setting. It introduces ProbSparse self-attention to achieve O(L log L) time and memory complexity, a self-attention distilling mechanism that halves cascading layer inputs to handle extreme-length inputs, and a generative-style decoder that produces the entire output sequence in a single forward pass, dramatically reducing inference latency on long-horizon tasks."
paper: "https://doi.org/10.1609/aaai.v35i12.17325"
paper_title: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
venue: "AAAI 2021"
year: 2021
code: "https://github.com/thuml/Time-Series-Library"
revision: "2fb5b84ecef67c45a759f7cf82023d27afe27882"
license: "MIT"
tagline: "ProbSparse self-attention encoder with distilling convolutions and a generative one-shot decoder."
tags: ["transformer", "attention-variant", "sparse-attention", "long-horizon", "channel-mixing"]
composition: ["normalization=none", "decomposition=none", "temporal=component:self_attention_family+component:transformer_encdec", "channel=component:embed", "head=local:generative-one-shot-decoder-projection", "loss=loss:mse"]
---
# Informer

## Key ideas

- `ProbAttention` (`self_attention_family`) scores queries by sparsity and attends only with the top queries, giving O(L log L) attention.
- `ConvLayer` distilling between encoder layers halves the sequence length (`distil`).
- The decoder (`transformer_encdec`) takes the label window plus placeholder horizon in one pass and projects to `c_out`, with causal ProbSparse self-attention and full cross-attention.
- `DataEmbedding` (`embed`) embeds values (mixing channels) and time marks; there is no instance normalization or decomposition.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1609/aaai.v35i12.17325); title: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting; venue/year: AAAI 2021 / 2021
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `2fb5b84ecef67c45a759f7cf82023d27afe27882`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Informer.toml`](../../../../configs/models/Informer.toml).

## Differences

**Paper-driven local implementation.** The local model implements the paper's
ProbSparse query selection, encoder distilling operation, and generative
one-shot decoder using verified shared attention and Transformer primitives.
TSFLab uses its common decoder-input contract and does not claim the paper's
reported benchmark values. The external repository is reference-only; no
source file was copied or adapted.
`ProbAttention` returns its context time-first (`[B, L, heads, d_head]`) as in
the original Informer implementation; the pinned Time-Series-Library revision
keeps heads-first and views it without a transpose, which TSFLab does not
reproduce. Batch or head count 1 is supported (explicit `squeeze(-2)`).

## Shared components

- [`embed`](../_components/embed/README.md)
- [`self_attention_family`](../_components/self_attention_family/README.md)
- [`transformer_encdec`](../_components/transformer_encdec/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_layers=1`, `d_ff=256`, `dropout=0.1`, `factor=3`, `activation='gelu'`, `distil=True`, `embed='timeF'`, `freq='h'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- **Venue**: AAAI 2021
- **Published**: 2021 (arXiv: 2020-12)
- **arXiv**: https://arxiv.org/abs/2012.07436

## Abstract
Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves O(L log L) in time complexity and memory usage, and has comparable performance on sequences' dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.

## In TSFLab
Default config: `configs/models/Informer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Paper-driven local implementation.** The local model implements the paper's
ProbSparse query selection, encoder distilling operation, and generative
one-shot decoder using verified shared attention and Transformer primitives.
TSFLab uses its common decoder-input contract and does not claim the paper's
reported benchmark values. The external repository is reference-only; no
source file was copied or adapted.
`ProbAttention` returns its context time-first (`[B, L, heads, d_head]`) as in
the original Informer implementation; the pinned Time-Series-Library revision
keeps heads-first and views it without a transpose, which TSFLab does not
reproduce. Batch or head count 1 is supported (explicit `squeeze(-2)`).

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZhouZPZLXZ21,
  author       = {Haoyi Zhou and
                  Shanghang Zhang and
                  Jieqi Peng and
                  Shuai Zhang and
                  Jianxin Li and
                  Hui Xiong and
                  Wancai Zhang},
  title        = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series
                  Forecasting},
  booktitle    = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2021, Thirty-Third Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
                  2021},
  pages        = {11106--11115},
  publisher    = {{AAAI} Press},
  year         = {2021},
  url          = {https://doi.org/10.1609/aaai.v35i12.17325},
  doi          = {10.1609/AAAI.V35I12.17325},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ZhouZPZLXZ21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
