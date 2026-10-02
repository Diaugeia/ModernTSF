---
name: "CoRe"
summary: "CoRe is a clean-room test-time-adaptation module that mixes variates in the correction space rather than the prediction space. A COSA-style base adapter yields per-variate corrections, a shared-anchor rank-r bottleneck refines them, and a tanh gate driven by input-window spectral entropy and band-energy ratios scales the refinement: Y = Y_base + Delta + g * delta."
paper: "https://arxiv.org/abs/2609.34638"
paper_title: "Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting"
venue: "arXiv"
year: 2026
code: "https://github.com/yyddou/CoReTTA"
revision: "7e3861f24e3aaece9fffc048fd585b3d283835d8"
license: "unspecified (no LICENSE file at the pinned revision)"
---
# CoRe
<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2609.34638); title: Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting; venue/year: arXiv / 2026
- [codebase](https://github.com/yyddou/CoReTTA); revision: `7e3861f24e3aaece9fffc048fd585b3d283835d8`; license: `unspecified (no LICENSE file at the pinned revision)`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CoRe.toml`](../../../../configs/models/CoRe.toml).

## Differences

No external source file is copied or imported; the official repository has no
license, so this is an independent rewrite from the paper.

- The streaming protocol is not run: delayed-label buffer, batch-wise adaptation loop, PAAS, adaptive learning-rate schedule, and weight-decay objective (Eq. 9) are outside the model. `adaptable_parameters()` exposes the adapted set; a frozen last-value base and a channel-mean history context are explicit self-contained fallbacks, whereas the official context is a buffer of past revealed target means.
- The official code computes one spectral descriptor per adaptation batch (power spectrum averaged over the batch); the paper defines it per input window, and this implementation computes it per sample (`[batch, 4]`).
- Band edges follow the paper's inclusive index ranges (Eqs. 13-15). The official code uses half-open thirds (`[0,K//3)`, `[K//3,2K//3)`, `[2K//3,K)`), so its bands differ by one bin at each edge.
- The official base adapter supports a multi-layer MLP option (`ADAPTER_LAYERS`, `HIDDEN_DIM`); only the single-layer linear adapter used in the paper's main configuration is implemented.
- Defaults follow the paper and official code: gate initialized to zero (adapted forecast equals the base at initialization), SCR Xavier gain 0.01 with zero biases, spectral gate weights zero and bias -1, `r = C`. For high-dimensional datasets the paper uses `r = 16`; set `scr_rank`.
- Because the gates start at zero, only the gate parameters and SCR/gate biases receive non-zero gradients on the first step.

Evidence is in `../../../verification/evidence/CoRe.json`; equation tests are in `tests/test_core_tta.py`.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`spectral_descriptor`](../_components/spectral_descriptor/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `context_len=5`, `var_wise_gating=True`, `scr_rank=0`, `gate_init=0.0`, `gate_bias_init=-1.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting
- **arXiv**: https://arxiv.org/abs/2609.34638 (v1, 2026-09-28)
- **Code**: https://github.com/yyddou/CoReTTA at `7e3861f24e3aaece9fffc048fd585b3d283835d8`; no LICENSE file, and the repository asserts no repository-wide license (it lists TAFAS and COSA third-party notices). Treated as read-only reference.

## Architecture map

1. Frozen base forecast `Y_base` (self-contained last-value fallback, or any external frozen forecast).
2. Base adapter: per-variate (or shared) linear map of `[Y_base_c || context]` scaled by `tanh(gate)` (variate-wise gate by default).
3. Shared correction anchor `alpha = mean_c Delta_c` and concatenation `[Delta_c || alpha]` (Eqs. 3-4).
4. Rank-`r` bottleneck `delta_c = W_up tanh(W_down z_c)`, shared across variates, `r = C` by default (Eq. 5).
5. Spectral descriptor `s = [SE, LBR, MBR, HBR]` of the input window and gate `g = tanh(W_g s + b_g)` per variate (Eqs. 7-8, 10-17).
6. Output `Y = Y_base + Delta + g * delta` (Eq. 6).

Sections 3.2-3.6 and Appendix B.1-B.3 of the paper were read, and `tta/core.py` (`SimpleOutputAdapter`, `_compute_spectral_context`) was inspected at the pinned revision to resolve initialization and tensor layout. Component decisions: `channel_wise_linear` reused (frozen last-value base and per-variate/shared base adapter); `spectral_descriptor` extracted as a new paper-neutral component (Eqs. 10-17); the SCR bottleneck, anchor, and gate wiring stay model-local.

## Source and verification

No external source file is copied or imported; the official repository has no
license, so this is an independent rewrite from the paper.

- The streaming protocol is not run: delayed-label buffer, batch-wise adaptation loop, PAAS, adaptive learning-rate schedule, and weight-decay objective (Eq. 9) are outside the model. `adaptable_parameters()` exposes the adapted set; a frozen last-value base and a channel-mean history context are explicit self-contained fallbacks, whereas the official context is a buffer of past revealed target means.
- The official code computes one spectral descriptor per adaptation batch (power spectrum averaged over the batch); the paper defines it per input window, and this implementation computes it per sample (`[batch, 4]`).
- Band edges follow the paper's inclusive index ranges (Eqs. 13-15). The official code uses half-open thirds (`[0,K//3)`, `[K//3,2K//3)`, `[2K//3,K)`), so its bands differ by one bin at each edge.
- The official base adapter supports a multi-layer MLP option (`ADAPTER_LAYERS`, `HIDDEN_DIM`); only the single-layer linear adapter used in the paper's main configuration is implemented.
- Defaults follow the paper and official code: gate initialized to zero (adapted forecast equals the base at initialization), SCR Xavier gain 0.01 with zero biases, spectral gate weights zero and bias -1, `r = C`. For high-dimensional datasets the paper uses `r = 16`; set `scr_rank`.
- Because the gates start at zero, only the gate parameters and SCR/gate biases receive non-zero gradients on the first step.

Evidence is in `../../../verification/evidence/CoRe.json`; equation tests are in `tests/test_core_tta.py`.


## Citation

```bibtex
@misc{deng2026correctionspacecrossvariateinteractiontesttime,
  title         = {Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting},
  author        = {Yuanyuan Deng and Mykola Pechenizkiy and Songgaojun Deng},
  year          = {2026},
  eprint        = {2609.34638},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.34638}
}
```
