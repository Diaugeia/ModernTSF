---
name: "MambaTS"
summary: "MambaTS lays out every variable's patch tokens one variable after another (Variable-Aware Scan along Time) and runs a causal-convolution-free selective SSM with dropout on the selective parameters over the whole token sequence. During training the variable order is randomly permuted and each sample's loss is credited to the variable transitions it used; at inference the cheapest scan path of those costs, solved as an asymmetric travelling-salesman problem by simulated annealing, fixes the order. A flatten-linear head shared across variables produces the forecast."
paper: "https://arxiv.org/abs/2405.16440"
paper_title: "MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting"
venue: "Pattern Recognition 2026"
year: 2026
code: "https://github.com/XiudingCai/MambaTS-pytorch"
revision: "3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d"
license: "MIT"
tagline: "Variable-major patch-token scan by a conv-free Mamba with a learned variable order from permutation training and ATSP."
tags: ["ssm", "mamba", "patching", "channel-mixing", "normalization", "variable-ordering"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:mamba", "channel=local:variable-aware-scan-order", "head=component:flatten_forecast_head", "loss=loss:mse+local:scan-cost-update"]
---
# MambaTS

## Key ideas

- `Model` embeds non-overlapping patches per variable and concatenates tokens variable-major, so a single Mamba scan visits each variable's whole history before the next (VAST).
- `TemporalMambaStack` stacks `mamba` `MambaBlock` mixers with `use_conv=False` (no causal convolution) and `x_dropout` on the step-size/B/C projections, in add-then-LayerNorm order.
- Variable Permutation Training shuffles the variable order per sample each step; the model's `training_objective` credits each sample's loss to its K transitions through `update_scan_costs`.
- `scan_order` solves the asymmetric TSP over the mean transition cost (`anneal_scan_path`, or `greedy_scan_path` for `atsp_solver="GD"`) once per cost state; inference scans in that order.
- `revin` standardises each variable and `flatten_forecast_head` maps every variable's flattened patch tokens to the horizon with one shared linear layer.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2405.16440); title: MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting; venue/year: Pattern Recognition 2026 / 2026
- [codebase](https://github.com/XiudingCai/MambaTS-pytorch); revision: `3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MambaTS.toml`](../../../../configs/models/MambaTS.toml).

## Differences

Independent implementation; no official code was copied (official repository MIT, pinned revision above; checked `models/MambaTS.py`, `layers/mamba_ssm/mixer2_seq_simple.py`, `layers/mamba_ssm/mamba_simple.py`, `exp/exp_long_term_forecasting.py`, `run.py`, and the paper's Sec. 5 and Eqs. 3-7). Inputs are `[B, seq_len, enc_in]`, marks and decoder inputs are ignored, and the output is `[B, pred_len, enc_in]` (the runner slices the target for `MS`). The SSM mixer, instance normalisation and head are shared components; the VAST cost bookkeeping and ATSP decoder are model-local, and the runner `training_objective` returns the configured criterion while crediting per-sample criterion values to the shuffled transitions.

- **Kernels and dependencies.** The official model needs `mamba_ssm` kernels, Triton fused add-norm and `python_tsp`; here the scan is the `mamba` component's sequential PyTorch recurrence (equivalent, slower) and the add-then-LayerNorm residual stream is explicit.
- **Causal convolution.** Official experiments never pass `--use_casual_conv`, so no convolution (and no SiLU on the scan input) is used; `use_causal_conv=False` is the default, as in the paper's TMB.
- **Cost accumulation.** Official code keeps a running sum and visit count (mean of observed costs, as in Proposition 2); the paper's Eq. 6 describes a moving average with rate beta whose batch form is unspecified, so it is not implemented. Unvisited pairs (NaN in the official division) take the mean of visited pairs; the cheapest pair is shifted to about zero as in the official `get_adjacency_matrix`.
- **Per-sample attribution.** The official `scatter_add_` pairs flattened `(batch, transition)` indices with `cost.repeat(K)`, which credits sample `i % B`'s loss to the wrong samples' transitions; here each sample's loss goes to its own K transitions (the paper's intent). The batch-deviation scaling is skipped for a single sample or zero spread (NaN or infinity officially). The unused official `ending_points` is dropped and the statistics are persistent buffers, not frozen parameters.
- **Loss.** The official VPT mode switches to element-wise MSE to read per-sample losses; here the configured criterion (MSE in the paper) is applied per sample without gradients and the optimised loss is that criterion on the whole batch.
- **ATSP solver.** Official: `python_tsp` simulated annealing from the identity order (options `LS`, `LK`, `GD`, `Random`), cached at the first evaluation until `reset_ids_shuffle` is called before test. Here `SA` is a seeded annealing over swaps starting from the greedy path with an `atsp_steps` budget and `GD` is nearest-neighbour; `LS`, `LK`, `Random` are not provided. The order is recomputed whenever the cost state changes (including a loaded checkpoint), so validation and test use the latest costs without a manual reset.
- **Patching.** The paper writes ceil(L/s) patches; official code and this model use `unfold` without padding, and the official head width `d_model * seq_len // patch_len` only matches when `stride == patch_len`, whereas the head here uses the true patch count.
- **Initialisation.** Reference Mamba step-size initialisation and the `1/sqrt(layers)` residual-projection scaling are reproduced; patch embedding and head keep PyTorch defaults, as officially.
- **Runner.** Optimiser, schedule, early stopping and mixed precision are runner configuration; the official seed 3047 is not modelled. `training_objective` is unsupported under `DataParallel`.
- **Evidence.** Structure and reference-recursion tests are in `tests/test_mambats.py`; unified verification evidence has not been produced yet and is left stale for CI.

## Shared components

- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `e_layers=4`, `d_state=16`, `d_conv=4`, `expand=2`, `dropout=0.2`, `patch_len=48`, `stride=48`, `use_causal_conv=False`, `vpt_mode=1`, `atsp_solver='SA'`, `atsp_steps=20000`, `atsp_seed=0`
<!-- model-card:canonical:end -->

## Source and verification

Independent implementation; no official code was copied (official repository MIT, pinned revision above; checked `models/MambaTS.py`, `layers/mamba_ssm/mixer2_seq_simple.py`, `layers/mamba_ssm/mamba_simple.py`, `exp/exp_long_term_forecasting.py`, `run.py`, and the paper's Sec. 5 and Eqs. 3-7). Inputs are `[B, seq_len, enc_in]`, marks and decoder inputs are ignored, and the output is `[B, pred_len, enc_in]` (the runner slices the target for `MS`). The SSM mixer, instance normalisation and head are shared components; the VAST cost bookkeeping and ATSP decoder are model-local, and the runner `training_objective` returns the configured criterion while crediting per-sample criterion values to the shuffled transitions.

- **Kernels and dependencies.** The official model needs `mamba_ssm` kernels, Triton fused add-norm and `python_tsp`; here the scan is the `mamba` component's sequential PyTorch recurrence (equivalent, slower) and the add-then-LayerNorm residual stream is explicit.
- **Causal convolution.** Official experiments never pass `--use_casual_conv`, so no convolution (and no SiLU on the scan input) is used; `use_causal_conv=False` is the default, as in the paper's TMB.
- **Cost accumulation.** Official code keeps a running sum and visit count (mean of observed costs, as in Proposition 2); the paper's Eq. 6 describes a moving average with rate beta whose batch form is unspecified, so it is not implemented. Unvisited pairs (NaN in the official division) take the mean of visited pairs; the cheapest pair is shifted to about zero as in the official `get_adjacency_matrix`.
- **Per-sample attribution.** The official `scatter_add_` pairs flattened `(batch, transition)` indices with `cost.repeat(K)`, which credits sample `i % B`'s loss to the wrong samples' transitions; here each sample's loss goes to its own K transitions (the paper's intent). The batch-deviation scaling is skipped for a single sample or zero spread (NaN or infinity officially). The unused official `ending_points` is dropped and the statistics are persistent buffers, not frozen parameters.
- **Loss.** The official VPT mode switches to element-wise MSE to read per-sample losses; here the configured criterion (MSE in the paper) is applied per sample without gradients and the optimised loss is that criterion on the whole batch.
- **ATSP solver.** Official: `python_tsp` simulated annealing from the identity order (options `LS`, `LK`, `GD`, `Random`), cached at the first evaluation until `reset_ids_shuffle` is called before test. Here `SA` is a seeded annealing over swaps starting from the greedy path with an `atsp_steps` budget and `GD` is nearest-neighbour; `LS`, `LK`, `Random` are not provided. The order is recomputed whenever the cost state changes (including a loaded checkpoint), so validation and test use the latest costs without a manual reset.
- **Patching.** The paper writes ceil(L/s) patches; official code and this model use `unfold` without padding, and the official head width `d_model * seq_len // patch_len` only matches when `stride == patch_len`, whereas the head here uses the true patch count.
- **Initialisation.** Reference Mamba step-size initialisation and the `1/sqrt(layers)` residual-projection scaling are reproduced; patch embedding and head keep PyTorch defaults, as officially.
- **Runner.** Optimiser, schedule, early stopping and mixed precision are runner configuration; the official seed 3047 is not modelled. `training_objective` is unsupported under `DataParallel`.
- **Evidence.** Structure and reference-recursion tests are in `tests/test_mambats.py`; unified verification evidence has not been produced yet and is left stale for CI.

## Citation

```bibtex
@article{cai2026mambats,
  title   = {MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting},
  author  = {Xiuding Cai and Xueyao Wang and Yaoyao Zhu and Yu Yao},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {arXiv:2405.16440},
  url     = {https://arxiv.org/abs/2405.16440}
}
```
