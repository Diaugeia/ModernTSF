---
name: "MoU"
summary: "MoU (Mixture of Universals) is a channel-independent patch forecaster that replaces the patch embedding with a sparsely gated mixture of linear feature extractors (MoF) and encodes the tokens with a hierarchical Mamba, feed-forward, convolution and self-attention block (MoA) so that the receptive field grows from selected SSM dependencies to global attention."
paper: "https://arxiv.org/abs/2408.15997"
paper_title: "Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need"
venue: "KDD 2025"
year: 2025
code: "https://github.com/lunaaa95/mou"
revision: "5a73fd797fc177355fdae93bef7de95ea6d009aa"
license: "NOASSERTION"
tagline: "Noisy top-k mixture of linear patch extractors, then a Mamba, FFN, conv, attention hierarchy over patch tokens."
tags: ["hybrid", "mamba", "attention-variant", "patching", "mixture-of-experts", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:mamba+local:ffn-conv-attention-hierarchy", "channel=none", "head=component:flatten_forecast_head", "loss=loss:mse"]
---
# MoU

## Key ideas

- `MixtureOfFeatureExtractors` (MoF) gives each patch a noisy top-k mixture of linear `patch_len -> d_model` extractors, so patches with different local patterns use different encoders at near-constant active cost.
- `MixtureOfArchitecturesBlock` (MoA) stacks Mamba (`mamba` component), a GELU feed-forward residual, a kernel-3 convolution residual and one post-norm self-attention layer, widening the receptive field layer by layer.
- Tokens are channel-independent: RevIN, end-replication padding, unfolding into patches, a learned position table, then a flatten linear head (`flatten_forecast_head`).

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2408.15997); title: Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need; venue/year: KDD 2025 / 2025
- [codebase](https://github.com/lunaaa95/mou); revision: `5a73fd797fc177355fdae93bef7de95ea6d009aa`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MoU.toml`](../../../../configs/models/MoU.toml).

## Differences

Paper and official code (pinned revision `5a73fd7`, `layers/MoU_backbone.py`, `layers/MoE3.py`, `models/MoU.py`) were checked; the local module is an independent rewrite using the `entype=mof`, `ltencoder=mfca` configuration. The official repository has no license file, so the recorded license is `NOASSERTION`; no source was copied. The camera-ready KDD 2025 version is titled "Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting"; the arXiv title is the one recorded in the header. Recorded differences:

- The official `Encoder_MFCA.forward` assigns `output = block(x)` inside its loop, so only the last block's output is used and earlier blocks are dead for `e_layers > 1`. The local `blocks` are stacked sequentially.
- The Mamba mixer is the portable pure-PyTorch `mamba` component (no `mamba_ssm` CUDA kernel), with `d_conv=4`, `expand=2`, `dt_rank=ceil(d_model/16)` and `d_state=21`.
- The self-attention layer is `nn.MultiheadAttention` in a post-norm (BatchNorm) encoder layer. The official layer is the PatchTST `TSTEncoderLayer` with residual attention scores; with one attention layer per block the two are equivalent up to parameterization.
- The official sparse MoE (`MoE3.py`) dispatches tokens per expert and its load-balancing loss is commented out; the local gate computes all linear extractors densely and zeroes non-top-k weights (top-k of the softmax, renormalized with eps 1e-6, noise `softplus + 1e-2` in training). No auxiliary loss is used, matching the official training objective (MSE only).
- RevIN uses `affine=False` (official default `--affine 0`); end replication padding and patch count `(seq_len - patch_len) // stride + 2` match the official `padding_patch='end'`. The official series-decomposition option and the `se`/`dyconv`/`w` entype and alternative long-term encoder variants are not implemented.
- Dropouts that are hard-coded in the official block (Mamba 0.1, convolution 0.3) are exposed as `mamba_dropout` and `conv_dropout`; the rest follow the official `--dps` order (`ffn`, `block`, `attn`, `tf`). The preset sizes (`d_model=128`, `n_heads=16`, `d_ff=256`, `e_layers=1`) are TSFLab defaults; the `d_state=21`, `num_experts=4`, `top_k=2` values match the official defaults, while the official ETTh1 script uses `d_model=64`, `n_heads=4`, `d_ff=128`, `seq_len=336`.

## Shared components

- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`positional_encoding`](../_components/positional_encoding/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `d_model=128`, `n_heads=16`, `d_ff=256`, `e_layers=1`, `num_experts=4`, `top_k=2`, `d_state=21`, `ffn_expand=2`, `dropout=0.1`, `mamba_dropout=0.1`, `ffn_dropout=0.2`, `block_dropout=0.2`, `conv_dropout=0.3`, `attn_dropout=0.0`, `tf_dropout=0.2`, `head_dropout=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Source and verification

Paper and official code (pinned revision `5a73fd7`, `layers/MoU_backbone.py`, `layers/MoE3.py`, `models/MoU.py`) were checked; the local module is an independent rewrite using the `entype=mof`, `ltencoder=mfca` configuration. The official repository has no license file, so the recorded license is `NOASSERTION`; no source was copied. The camera-ready KDD 2025 version is titled "Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting"; the arXiv title is the one recorded in the header. Recorded differences:

- The official `Encoder_MFCA.forward` assigns `output = block(x)` inside its loop, so only the last block's output is used and earlier blocks are dead for `e_layers > 1`. The local `blocks` are stacked sequentially.
- The Mamba mixer is the portable pure-PyTorch `mamba` component (no `mamba_ssm` CUDA kernel), with `d_conv=4`, `expand=2`, `dt_rank=ceil(d_model/16)` and `d_state=21`.
- The self-attention layer is `nn.MultiheadAttention` in a post-norm (BatchNorm) encoder layer. The official layer is the PatchTST `TSTEncoderLayer` with residual attention scores; with one attention layer per block the two are equivalent up to parameterization.
- The official sparse MoE (`MoE3.py`) dispatches tokens per expert and its load-balancing loss is commented out; the local gate computes all linear extractors densely and zeroes non-top-k weights (top-k of the softmax, renormalized with eps 1e-6, noise `softplus + 1e-2` in training). No auxiliary loss is used, matching the official training objective (MSE only).
- RevIN uses `affine=False` (official default `--affine 0`); end replication padding and patch count `(seq_len - patch_len) // stride + 2` match the official `padding_patch='end'`. The official series-decomposition option and the `se`/`dyconv`/`w` entype and alternative long-term encoder variants are not implemented.
- Dropouts that are hard-coded in the official block (Mamba 0.1, convolution 0.3) are exposed as `mamba_dropout` and `conv_dropout`; the rest follow the official `--dps` order (`ffn`, `block`, `attn`, `tf`). The preset sizes (`d_model=128`, `n_heads=16`, `d_ff=256`, `e_layers=1`) are TSFLab defaults; the `d_state=21`, `num_experts=4`, `top_k=2` values match the official defaults, while the official ETTh1 script uses `d_model=64`, `n_heads=4`, `d_ff=128`, `seq_len=336`.

## Citation

```bibtex
@inproceedings{peng2025semantics,
  title     = {Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting},
  author    = {Peng, Sijia and Xiong, Yun and Zhu, Yangyong and Shen, Zhiqiang},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages     = {2269--2280},
  year      = {2025},
  doi       = {10.1145/3711896.3737123}
}
```
