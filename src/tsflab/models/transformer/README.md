---
name: "Transformer"
summary: "Transformer is the standard encoder-decoder attention architecture applied to time-series forecasting, with full scaled dot-product self-attention, causal decoder attention, encoder-decoder cross-attention, positional/value embeddings, and a one-shot forecast projection."
paper: "https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need"
paper_title: "Attention Is All You Need"
venue: "NeurIPS 2017"
year: 2017
code: "https://github.com/thuml/Time-Series-Library"
revision: "2fb5b84ecef67c45a759f7cf82023d27afe27882"
license: "MIT"
---
# Transformer

Transformer is the standard encoder-decoder attention architecture applied to time-series forecasting, with full scaled dot-product self-attention, causal decoder attention, encoder-decoder cross-attention, positional/value embeddings, and a one-shot forecast projection.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need); title: Attention Is All You Need; venue/year: NeurIPS 2017 / 2017
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `2fb5b84ecef67c45a759f7cf82023d27afe27882`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Transformer.toml`](../../../../configs/models/Transformer.toml).

## Differences

**Paper-driven local implementation.** TSFLab assembles the paper's scaled
dot-product attention, encoder/decoder residual blocks, causal decoder mask,
cross-attention, and position-wise feed-forward layers from verified shared
components. The time-series embedding and one-shot forecast boundary are local
integration choices. The external repository is reference-only; no source file
was copied or adapted. Published benchmark reproduction remains separate from
the independent code validation.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`self_attention_family`](../_components/self_attention_family/README.md)
- [`transformer_encdec`](../_components/transformer_encdec/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_layers=1`, `d_ff=256`, `dropout=0.1`, `activation='gelu'`, `embed='timeF'`, `freq='h'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Attention Is All You Need
- **Venue**: NeurIPS 2017
- **Published**: 2017 (arXiv: 2017-06)
- **arXiv**: https://arxiv.org/abs/1706.03762

## Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## In TSFLab
Default config: `configs/models/Transformer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Paper-driven local implementation.** TSFLab assembles the paper's scaled
dot-product attention, encoder/decoder residual blocks, causal decoder mask,
cross-attention, and position-wise feed-forward layers from verified shared
components. The time-series embedding and one-shot forecast boundary are local
integration choices. The external repository is reference-only; no source file
was copied or adapted. Published benchmark reproduction remains separate from
the independent code validation.

## Citation

```bibtex
@misc{vaswani2017attention,
  author        = {Ashish Vaswani and
                  Noam Shazeer and
                  Niki Parmar and
                  Jakob Uszkoreit and
                  Llion Jones and
                  Aidan N. Gomez and
                  Lukasz Kaiser and
                  Illia Polosukhin},
  title         = {Attention Is All You Need},
  year          = {2017},
  eprint        = {1706.03762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1706.03762}
}
```
