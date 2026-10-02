---
name: "Reformer"
summary: "Reformer is a memory-efficient Transformer model adapted for the time-series forecasting setting. It replaces standard dot-product self-attention with locality-sensitive hashing (LSH) attention, reducing the attention complexity from O(L²) to O(L log L), and employs reversible residual layers to avoid storing all intermediate activations, making it practical for long input sequences."
paper: "https://openreview.net/forum?id=rkgNKkHtvB"
paper_title: "Reformer: The Efficient Transformer"
venue: "ICLR 2020"
year: 2020
code: "https://github.com/thuml/Time-Series-Library"
revision: "3a4819420d14095354aae96750ce8c499ef5f05e"
license: "MIT"
tagline: "Encoder over history plus future placeholders using shared-QK LSH bucket attention in reversible residual blocks."
tags: ["transformer", "attention-variant", "sparse-attention", "reversible", "channel-mixing", "covariates"]
composition: ["normalization=none", "decomposition=none", "temporal=local:lsh-attention-reversible-encoder", "channel=local:channel-mixing-value-embedding", "head=local:last-pred-len-linear-projection", "loss=loss:mse"]
---
# Reformer

## Key ideas

- `LSHSelfAttention` shares query and key projections, hashes them with random rotations over `n_hashes` rounds, sorts into buckets of `bucket_size`, and attends within each chunk and the previous one; duplicates across rounds get a log-multiplicity correction.
- `ReversibleBlock` splits the width in half (`y1 = x1 + f(x2)`, `y2 = x2 + g(y1)`) and provides an `inverse`, as in the paper.
- History and `pred_len` future placeholders (from `x_dec`) are embedded together by `TimeValueEmbedding` (value plus 6 calendar marks); the last `pred_len` positions are projected to `c_out`.
- The attention is implemented with Python loops over buckets, so it is exact to the method but slow.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=rkgNKkHtvB); title: Reformer: The Efficient Transformer; venue/year: ICLR 2020 / 2020
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `3a4819420d14095354aae96750ce8c499ef5f05e`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Reformer.toml`](../../../../configs/models/Reformer.toml).

## Differences

**Clean-room implementation: confirmed.** Sparse candidate width, causal masking, duplicate correction, shared-QK hashing, and reversible inversion are exercised by local structure tests. Inputs are history plus future placeholders and optional six-column marks; outputs are `[B, pred_len, c_out]`. Standard autograd does not claim the paper's custom reversible-memory reference comparison. The linked wrapper is reference-only and no source was copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `dropout=0.1`, `bucket_size=4`, `n_hashes=4`, `causal=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Reformer: The Efficient Transformer
- **Venue**: ICLR 2020
- **Published**: 2020
- **arXiv**: https://arxiv.org/abs/2001.04451

## Abstract
Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L²) to O(L log L), where L is the length of the sequence. Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training process instead of N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences.

## In TSFLab
Default config: `configs/models/Reformer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** Sparse candidate width, causal masking, duplicate correction, shared-QK hashing, and reversible inversion are exercised by local structure tests. Inputs are history plus future placeholders and optional six-column marks; outputs are `[B, pred_len, c_out]`. Standard autograd does not claim the paper's custom reversible-memory reference comparison. The linked wrapper is reference-only and no source was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/KitaevKL20,
  author       = {Nikita Kitaev and
                  Lukasz Kaiser and
                  Anselm Levskaya},
  title        = {Reformer: The Efficient Transformer},
  booktitle    = {8th International Conference on Learning Representations, {ICLR} 2020,
                  Addis Ababa, Ethiopia, April 26-30, 2020},
  publisher    = {OpenReview.net},
  year         = {2020},
  url          = {https://arxiv.org/abs/2001.04451},
  eprinttype   = {arXiv},
  eprint       = {2001.04451},
  timestamp    = {Thu, 07 May 2020 17:11:48 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/KitaevKL20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
