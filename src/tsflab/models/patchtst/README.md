---
name: "PatchTST"
summary: "PatchTST is a Transformer-based model for multivariate and univariate long-term time-series forecasting that segments each channel into subseries-level patches fed as input tokens, combined with a channel-independence strategy where each channel shares the same Transformer weights. This design retains local semantic information, drastically reduces attention-map memory, and allows the model to attend over a much longer historical context."
paper: "https://openreview.net/forum?id=Jbdc0vTOcol"
paper_title: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
venue: "ICLR 2023"
year: 2023
code: "https://github.com/yuqinie98/PatchTST"
revision: "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
license: "Apache-2.0"
---
# PatchTST

PatchTST is a Transformer-based model for multivariate and univariate long-term time-series forecasting that segments each channel into subseries-level patches fed as input tokens, combined with a channel-independence strategy where each channel shares the same Transformer weights. This design retains local semantic information, drastically reduces attention-map memory, and allows the model to attend over a much longer historical context.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=Jbdc0vTOcol); title: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers; venue/year: ICLR 2023 / 2023
- [codebase](https://github.com/yuqinie98/PatchTST); revision: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PatchTST.toml`](../../../../configs/models/PatchTST.toml).

## Differences

**Clean-room implementation: confirmed.** `patchify` maps the paper's
overlapping patch equation; folding channels into the batch axis enforces
channel independence and shared Transformer weights; the flatten head maps all
patch tokens to the horizon. Inputs are `[B, context_window, enc_in]`; outputs
are `[B, target_window, enc_in]`; marks/decoder inputs are accepted and ignored.
Self-supervised pretraining, transfer learning, residual-attention accumulation,
and checkpoint or published-metric reference comparison are not included.

**Component-extraction note.** This model does not consume the cataloged
`patchtst.PatchTSTBackbone` (used by `quantile_patchtst`). The two diverge
materially: the backbone's `"zeros"` positional-encoding mode draws
`uniform_(-0.02, 0.02)` noise while this model's `"zeros"` mode is a literal
zero tensor; the backbone's encoder is `tst_transformer.TSTEncoder`
(`nn.TransformerEncoder`/`nn.TransformerEncoderLayer`, fused in-projection
attention) while this model uses a bespoke `PatchEncoderLayer`; and the
backbone's `FlattenForecastHead` applies dropout after the linear head while
this model applies `head_dropout` before it. Migrating would change both
parameter initialization and `state_dict()` keys, so per the rule that keeps
checkpoint-loading attribute names, this model stays on its own local
backbone.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `padding_patch='end'`, `e_layers=3`, `d_model=512`, `n_heads=8`, `d_ff=2048`, `activation='gelu'`, `norm='BatchNorm'`, `attn_dropout=0.0`, `ffn_dropout=0.0`, `res_dropout=0.0`, `proj_dropout=0.0`, `head_dropout=0.0`, `pre_norm=False`, `pe='zeros'`, `learn_pe=False`, `individual=False`, `revin=True`, `affine=False`, `subtract_last=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **Venue**: ICLR 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.14730

## Abstract
We propose an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning. It is based on two key components: (i) segmentation of time series into subseries-level patches which are served as input tokens to Transformer; (ii) channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series. Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history. Our channel-independent patch time series Transformer (PatchTST) can improve the long-term forecasting accuracy significantly when compared with that of SOTA Transformer-based models. We also apply our model to self-supervised pre-training tasks and attain excellent fine-tuning performance, which outperforms supervised training on large datasets. Transferring of masked pre-trained representation on one dataset to others also produces SOTA forecasting accuracy.

## In TSFLab
Default config: `configs/models/PatchTST.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`. The Apache-2.0 repository is retained as
`reference-only`; its source was not copied for this rewrite. Structural and
runtime evidence is generated by `uv run tsf verify model PatchTST`.

## Verification

**Clean-room implementation: confirmed.** `patchify` maps the paper's
overlapping patch equation; folding channels into the batch axis enforces
channel independence and shared Transformer weights; the flatten head maps all
patch tokens to the horizon. Inputs are `[B, context_window, enc_in]`; outputs
are `[B, target_window, enc_in]`; marks/decoder inputs are accepted and ignored.
Self-supervised pretraining, transfer learning, residual-attention accumulation,
and checkpoint or published-metric reference comparison are not included.

**Component-extraction note.** This model does not consume the cataloged
`patchtst.PatchTSTBackbone` (used by `quantile_patchtst`). The two diverge
materially: the backbone's `"zeros"` positional-encoding mode draws
`uniform_(-0.02, 0.02)` noise while this model's `"zeros"` mode is a literal
zero tensor; the backbone's encoder is `tst_transformer.TSTEncoder`
(`nn.TransformerEncoder`/`nn.TransformerEncoderLayer`, fused in-projection
attention) while this model uses a bespoke `PatchEncoderLayer`; and the
backbone's `FlattenForecastHead` applies dropout after the linear head while
this model applies `head_dropout` before it. Migrating would change both
parameter initialization and `state_dict()` keys, so per the rule that keeps
checkpoint-loading attribute names, this model stays on its own local
backbone.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/NieNSK23,
  author       = {Yuqi Nie and
                  Nam H. Nguyen and
                  Phanwadee Sinthong and
                  Jayant Kalagnanam},
  title        = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023},
  url          = {https://openreview.net/forum?id=Jbdc0vTOcol},
  timestamp    = {Wed, 24 Jul 2024 16:50:33 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/NieNSK23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
