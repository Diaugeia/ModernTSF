---
name: "LightGBMTS"
description: "Differentiable boosting baseline: additive soft trees of varying depth over a learned soft lag-feature gate, with residual state updates. Use as a tree-ensemble baseline on few-channel data; not for many-channel data (window is flattened) or real LightGBM behaviour."
---

# LightGBMTS

## Idea

- A learned sigmoid gate (`feature_logits`) softly selects lag features of the flattened window before the base linear map and every tree.
- `num_estimators` `SoftDecisionTree`s (`soft_tree`) of cycling depth `1 + index % tree_depth` add scaled corrections to a linear base forecast; learned backcasts update the input state between stages.
- `aux_loss = 1e-4 * mean(gate)` encourages sparse feature use; `revin` wraps the model.

## When to use

- A boosting-style baseline that selects a sparse subset of lag features; useful to compare deep models against a tree ensemble trained end to end.
- RevIN normalization handles level shifts across windows.
- Avoid with many channels: the window is flattened to `seq_len * enc_in` inputs and outputs, so parameters grow with channels times lengths.
- Not a substitute for the LightGBM library (no split search or leaf-wise growth).

## Configure

- `enc_in`: must equal the channel count; the input is flattened to `seq_len * enc_in` features.

Other hyperparameters: preset defaults in `configs/models/LightGBMTS.toml`; tune generically.

## Differences

- Clean-room baseline using learned soft feature gates and compact additive trees; the paper is conceptual background only and no external source was inspected or copied.
- Does not implement LightGBM's histogram split search, leaf-wise growth, GOSS, EFB, distributed systems, or the external library API.
