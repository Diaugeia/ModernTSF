---
name: "GPHT"
summary: "GPHT models tokens auto-regressively with a cascade of max-pooled causal patch transformers whose stage inputs are iterative residuals and whose outputs are summed."
paper: "https://arxiv.org/abs/2402.16516"
paper_title: "Generative Pretrained Hierarchical Transformer for Time Series Forecasting"
venue: "KDD 2024"
year: 2024
code: "https://github.com/icantnamemyself/GPHT"
revision: "cfdbbd2f8386603476d0b48705b6189cfae0ad90"
license: "MIT"
tagline: "Auto-regressive token forecaster: pooled causal patch-transformer stages chained by residuals, outputs summed."
tags: ["transformer", "patching", "hierarchical", "multi-scale", "autoregressive", "generative", "pretraining", "normalization", "channel-independent", "revin"]
composition: ["normalization=component:revin", "decomposition=local:iterative-residual-stages", "temporal=component:self_attention_family+component:transformer_encdec", "channel=local:channel-independent-shared-weights", "head=local:next-token-linear-head-rollout", "loss=local:next-token-teacher-forced-mse"]
---
# GPHT

## Key ideas

- GPHT is auto-regressive over tokens of `token_len` steps: a causally masked patch transformer maps every token to the next token (`HierarchicalStage`, paper eq. 3).
- Each stage max-pools its input (`pooling_rates`, default 8/4/2/1) so early stages see coarse patterns; stage inputs are iterative residuals `x - PadFirstToken(out)` and stage outputs are summed (eqs. 4-5).
- `revin` instance normalization wraps the stack; `forward` rolls the window one token at a time for horizons longer than `token_len`.
- `training_objective` (wired through `spec.py`; the configured criterion is not used) is the paper's teacher-forced next-token MSE over every position; the pretrained mixed-dataset checkpoint is not part of this entry.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 48, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2402.16516); title: Generative Pretrained Hierarchical Transformer for Time Series Forecasting; venue/year: KDD 2024 / 2024
- [codebase](https://github.com/icantnamemyself/GPHT); revision: `cfdbbd2f8386603476d0b48705b6189cfae0ad90`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GPHT.toml`](../../../../configs/models/GPHT.toml).

## Differences

- Implementation: local rewrite from Section 3 (Eqs. 3 to 6). The official code (`icantnamemyself/GPHT`, MIT) was inspected at revision `cfdbbd2f8386603476d0b48705b6189cfae0ad90` (`models/GPHT.py`, `exp/exp_long_term_forecasting_GPHT.py`, `layers/Embed.py`, `scripts/`) to resolve omissions; no source was copied.
- Resolved from the official code: each stage is `MaxPool1d(rate)`, a patch embedding with `patch = token_len / rate`, stride equal to the patch and no padding (bias-free linear value embedding plus sinusoidal positions, then dropout), a causally masked full-attention encoder with a final `LayerNorm`, and `Linear(d_model, token_len)` mapping every hidden state to the next token; instance normalization is the biased variance plus `1e-5` and is recomputed on every rolled window; the next stage receives `x - PadFirstToken(out)` (right shift by one token); the training target is `x[token_len:]` followed by the first `token_len` steps of the label, scored with MSE on every position.
- Differences from the official code: the official model reads `GT_d_model`, `GT_d_ff`, `GT_e_layers` as single integers shared by all stages, as here, and takes `depth` separately from `GT_pooling_rate` (here `depth` is `len(pooling_rates)`). The official training script asserts `pred_len == token_len`; here the objective accepts any `pred_len >= token_len` and uses only the first `token_len` label steps, and `forward` rolls the window token by token for any `pred_len` (the official evaluation does this with a separate `ar_pred_len`). Training uses all channels and a matching target; the official script's `MS` slicing is not reproduced.
- Not implemented: the mixed-dataset pretraining corpus and checkpoint, and the official fine-tuning mode that loads the pretrained model and freezes everything except `forecast_head`; this entry is the architecture only. The preset is `d_model=512`, `d_ff=2048`, `e_layers=3`, `n_heads=8`, `dropout=0.1` (the official pretraining defaults; its fine-tuning scripts use `dropout=0.0`), and `seq_len` comes from the task (official default 336) and must be a multiple of `token_len`. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure and equation tests in `tests/test_gpht_structure.py`; no training was run.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`revin`](../_components/revin/README.md)
- [`self_attention_family`](../_components/self_attention_family/README.md)
- [`transformer_encdec`](../_components/transformer_encdec/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=48`. Default
model parameters are: `enc_in=7`, `token_len=48`, `pooling_rates=[8, 4, 2, 1]`, `d_model=512`, `d_ff=2048`, `n_heads=8`, `e_layers=3`, `dropout=0.1`, `activation='gelu'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Generative Pretrained Hierarchical Transformer for Time Series Forecasting
- **Venue**: KDD 2024 (Research Track)
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.16516

## Source and verification

- Implementation: local rewrite from Section 3 (Eqs. 3 to 6). The official code (`icantnamemyself/GPHT`, MIT) was inspected at revision `cfdbbd2f8386603476d0b48705b6189cfae0ad90` (`models/GPHT.py`, `exp/exp_long_term_forecasting_GPHT.py`, `layers/Embed.py`, `scripts/`) to resolve omissions; no source was copied.
- Resolved from the official code: each stage is `MaxPool1d(rate)`, a patch embedding with `patch = token_len / rate`, stride equal to the patch and no padding (bias-free linear value embedding plus sinusoidal positions, then dropout), a causally masked full-attention encoder with a final `LayerNorm`, and `Linear(d_model, token_len)` mapping every hidden state to the next token; instance normalization is the biased variance plus `1e-5` and is recomputed on every rolled window; the next stage receives `x - PadFirstToken(out)` (right shift by one token); the training target is `x[token_len:]` followed by the first `token_len` steps of the label, scored with MSE on every position.
- Differences from the official code: the official model reads `GT_d_model`, `GT_d_ff`, `GT_e_layers` as single integers shared by all stages, as here, and takes `depth` separately from `GT_pooling_rate` (here `depth` is `len(pooling_rates)`). The official training script asserts `pred_len == token_len`; here the objective accepts any `pred_len >= token_len` and uses only the first `token_len` label steps, and `forward` rolls the window token by token for any `pred_len` (the official evaluation does this with a separate `ar_pred_len`). Training uses all channels and a matching target; the official script's `MS` slicing is not reproduced.
- Not implemented: the mixed-dataset pretraining corpus and checkpoint, and the official fine-tuning mode that loads the pretrained model and freezes everything except `forecast_head`; this entry is the architecture only. The preset is `d_model=512`, `d_ff=2048`, `e_layers=3`, `n_heads=8`, `dropout=0.1` (the official pretraining defaults; its fine-tuning scripts use `dropout=0.0`), and `seq_len` comes from the task (official default 336) and must be a multiple of `token_len`. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure and equation tests in `tests/test_gpht_structure.py`; no training was run.

## Citation

```bibtex
@inproceedings{liu2024gpht,
  title         = {Generative Pretrained Hierarchical Transformer for Time Series Forecasting},
  author        = {Liu, Zhiding and Yang, Jiqian and Cheng, Mingyue and Luo, Yucong and Li, Zhi},
  booktitle     = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year          = {2024},
  eprint        = {2402.16516},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2402.16516}
}
```
