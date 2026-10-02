---
name: "SAMformer"
summary: "SAMformer is a deliberately shallow Transformer for multivariate forecasting: RevIN, one single-head channel-wise attention block with a residual connection, and a linear forecaster, trained with sharpness-aware minimization (SAM) to escape the sharp loss minima that make ordinary Transformers generalize poorly on long-horizon benchmarks."
paper: "https://arxiv.org/abs/2402.10198"
paper_title: "SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention"
venue: "ICML 2024"
year: 2024
code: "https://github.com/romilbert/samformer"
revision: "71f10eaa696f2a098798779ee14b6ecd6b69bcd9"
license: "Apache-2.0"
tagline: "One channel-wise attention block plus linear head under RevIN, trained with sharpness-aware minimization."
tags: ["transformer", "channel-mixing", "attention-variant", "normalization", "lightweight", "sam", "training-objective"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:channel_wise_linear", "channel=local:channel-wise-attention", "head=component:channel_wise_linear", "loss=loss:mse+component:sharpness_aware"]
---
# SAMformer

## Key ideas

- `ChannelWiseAttention` treats each variate as a token whose features are its `seq_len` time steps: `softmax(QK^T / sqrt(hid_dim)) V` with `Q, K` projected to `hid_dim` and `V` kept at width `seq_len`, added residually; a single head, a single layer.
- The attended window is mapped to the horizon by one shared linear layer (`channel_wise_linear`), and `revin` normalizes the input and restores scale on the forecast.
- Sharpness-aware minimization is the training procedure, not architecture: `spec.training_objective` evaluates the loss at `w + rho * g / ||g||` through the `sharpness_aware` component, so ordinary optimizers realize SAM; `forward` is the plain network.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2402.10198); title: SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention; venue/year: ICML 2024 / 2024
- [codebase](https://github.com/romilbert/samformer); revision: `71f10eaa696f2a098798779ee14b6ecd6b69bcd9`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SAMformer.toml`](../../../../configs/models/SAMformer.toml).

## Differences

Paper and official code (pinned revision `71f10ea`, `samformer_pytorch/samformer/`) were checked; the local module is an independent rewrite. The architecture matches the official `SAMFormerArchitecture` (RevIN, single-head channel-wise attention with `Q, K: seq_len -> hid_dim`, `V: seq_len -> seq_len`, scale `1/sqrt(hid_dim)`, residual add, one shared `seq_len -> pred_len` linear forecaster, RevIN denormalization). Recorded differences:

- SAM is not an optimizer wrapper here. The official `SAM` optimizer runs `first_step` (ascent to `w + rho g/||g||`) and `second_step` (restore `w`, then step the base optimizer). TSFLab instead exposes `spec.training_objective`, which evaluates the criterion at the perturbed weights through `torch.func.functional_call` (first-order, perturbation detached), so the run's ordinary optimizer applies the same update. This costs the same two forward and backward passes. Gradient clipping, schedulers, weight decay, mixed precision and the optimizer itself come from the run config, not from the official defaults (Adam, `lr=1e-3`, `weight_decay=1e-5`, `rho=0.5`).
- `training_objective` is used only in training; the validation and test loss and `forward` are the plain network. `rho` is validated non-negative and `rho=0` reduces to one ordinary pass.
- `hid_dim` is a configurable parameter (the official trainer hard-codes 16); `use_revin=False` skips normalization as in the official model.
- The official network returns a flattened `[batch, channels * pred_len]` tensor and its trainer fits on pre-windowed arrays; TSFLab returns `[batch, pred_len, channels]` and uses the framework data pipeline, scaling, and loss (MSE) instead of the official dataset utilities and 100-epoch loop.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`revin`](../_components/revin/README.md)
- [`sharpness_aware`](../_components/sharpness_aware/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `hid_dim=16`, `rho=0.5`, `use_revin=True`
<!-- model-card:canonical:end -->

## Source and verification

Paper and official code (pinned revision `71f10ea`, `samformer_pytorch/samformer/`) were checked; the local module is an independent rewrite. The architecture matches the official `SAMFormerArchitecture` (RevIN, single-head channel-wise attention with `Q, K: seq_len -> hid_dim`, `V: seq_len -> seq_len`, scale `1/sqrt(hid_dim)`, residual add, one shared `seq_len -> pred_len` linear forecaster, RevIN denormalization). Recorded differences:

- SAM is not an optimizer wrapper here. The official `SAM` optimizer runs `first_step` (ascent to `w + rho g/||g||`) and `second_step` (restore `w`, then step the base optimizer). TSFLab instead exposes `spec.training_objective`, which evaluates the criterion at the perturbed weights through `torch.func.functional_call` (first-order, perturbation detached), so the run's ordinary optimizer applies the same update. This costs the same two forward and backward passes. Gradient clipping, schedulers, weight decay, mixed precision and the optimizer itself come from the run config, not from the official defaults (Adam, `lr=1e-3`, `weight_decay=1e-5`, `rho=0.5`).
- `training_objective` is used only in training; the validation and test loss and `forward` are the plain network. `rho` is validated non-negative and `rho=0` reduces to one ordinary pass.
- `hid_dim` is a configurable parameter (the official trainer hard-codes 16); `use_revin=False` skips normalization as in the official model.
- The official network returns a flattened `[batch, channels * pred_len]` tensor and its trainer fits on pre-windowed arrays; TSFLab returns `[batch, pred_len, channels]` and uses the framework data pipeline, scaling, and loss (MSE) instead of the official dataset utilities and 100-epoch loop.

## Citation

```bibtex
@inproceedings{ilbert2024samformer,
  title     = {{SAMformer}: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention},
  author    = {Romain Ilbert and Ambroise Odonnat and Vasilii Feofanov and Aladin Virmaux and Giuseppe Paolo and Themis Palpanas and Ievgen Redko},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024}
}
```
