---
name: "FeTS"
description: "Patch tokens scored by a Fourier-plus-polynomial basis into a binary feature mask that gates local feature aggregation, then local-conv and global fusion. Use for channel-independent long-term forecasting that benefits from selecting informative feature dimensions; not for cross-channel modelling."
---

# FeTS

## Idea

- `FourierPolyMask` scores each patch-token dimension with a cosine, sine and polynomial basis and thresholds at the mean to get a binary mask; the forward mask is exact and a sigmoid straight-through estimator carries gradients.
- `adaptive_features` (AdaFE) uses the mask to gate a learned local aggregation kernel over neighbouring feature dimensions, added residually to the patch tokens.
- A local `Conv1d` branch and a global mean branch (DSFFN) are concatenated and fused, then flattened and projected linearly to the horizon.
- Overlapping patches with channels folded into the batch; `revin` wraps the model.

## When to use

- Designed to amplify informative feature dimensions and suppress irrelevant ones in long-term forecasting, with a compact model.
- Channel-independent with shared weights: no cross-channel interaction.

## Configure

- `enc_in`: number of channels.
- `patch_len`, `stride` (clamped to `seq_len`): choose so `seq_len - patch_len` is divisible by `stride`; otherwise the latest steps are dropped by patching.

Other hyperparameters: preset defaults in `configs/models/FeTS.toml`; tune generically.

## Differences

Written for TSFLab from Eqs. (2)-(14) of the AAAI paper; `models/FeTS.py` of `lllucky111/FeTS` at `d908e434` (no license file, recorded `NOASSERTION`) was inspected as reference only, nothing copied.

- The paper's binary threshold mask is non-differentiable and the gradient path is unstated; here the forward mask stays exact with a sigmoid straight-through gradient in training.
- One compact AdaFE/DSFFN block; the paper's dataset-specific training schedule and hyperparameter sweep are not reproduced.

Citation: Wang, L., Chen, J., Liu, S. "FeTS: A Feature-Aware Framework for Time Series Forecasting." AAAI 2026, pp. 26328-26336. doi:10.1609/aaai.v40i31.39838.
