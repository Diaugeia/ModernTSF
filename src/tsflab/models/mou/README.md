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
tags: ["transformer", "ssm", "mamba", "patching", "moe", "channel-independent", "normalization", "time-series"]
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

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`positional_encoding`](../_components/positional_encoding/README.md)
- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `d_model=128`, `n_heads=16`, `d_ff=256`, `e_layers=1`, `num_experts=4`, `top_k=2`, `d_state=21`, `ffn_expand=2`, `dropout=0.1`, `mamba_dropout=0.1`, `ffn_dropout=0.2`, `block_dropout=0.2`, `conv_dropout=0.3`, `attn_dropout=0.0`, `tf_dropout=0.2`, `head_dropout=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Citation

```bibtex
@inproceedings{peng2025semantics,
  title     = {Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need},
  author    = {Peng, Sijia and Xiong, Yun and Zhu, Yangyong and Shen, Zhiqiang},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2025}
}
```
