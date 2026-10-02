---
name: "sharpness_aware"
kind: "component"
module: "tsflab.models._components.sharpness_aware"
summary: "First-order sharpness-aware minimization as a loss evaluated at adversarially perturbed weights, usable by any optimizer through ModelSpec.training_objective."
category: "utility"
input: "model nn.Module; loss_fn(run) -> (aux or None, scalar loss); rho float"
output: "(aux detached or None, scalar loss differentiating to the SAM gradient)"
origin: "SAM, Foret et al., ICLR 2021 (Sharpness-Aware Minimization for Efficiently Improving Generalization); adaptive form from ASAM, Kwon et al., ICML 2021"
origin_models: ["samformer"]
tags: ["sam", "sharpness", "training-objective", "adversarial-weights", "functional-call", "optimizer-agnostic"]
---

# sharpness_aware

## Purpose

SAM updates `w` with `grad L(w + e(w))`, where
`e(w) = rho * g / ||g||_2` and `g = grad L(w)` over all trainable parameters.
`sharpness_aware_loss(model, loss_fn, rho)` runs one ascent pass to obtain `g`,
detaches `e`, then evaluates the loss again with parameters `w + e` through
`torch.func.functional_call`. Backpropagating that loss gives, with respect to
the stored `w`, exactly the gradient the reference SAM optimizer applies after
its second step, so the usual `loss.backward(); optimizer.step()` loop (any base
optimizer, weight decay included) realizes SAM without a wrapper optimizer.

## Origin and granularity

Cut at the arithmetic only: the caller supplies `loss_fn`, which decides how a
batch becomes a loss, so the same component serves any model through the runner's
`training_objective` hook. SAMformer is the first consumer.

## Interface

`sharpness_aware_loss(model, loss_fn, rho, adaptive=False, eps=1e-12)`

- `model`: `nn.Module` in training mode; only parameters with
  `requires_grad` are perturbed.
- `loss_fn(run)`: called twice (once with `run = model`, once with a functional
  call using perturbed parameters). It must call `run(...)` where it would call
  the model, and return `(aux, loss)` with `loss` a scalar tensor; `aux` may be
  `None` or a tensor (for example the forecast).
- `rho` (float >= 0): neighborhood radius. `rho == 0` performs one ordinary pass.
- `adaptive`: ASAM-style element-wise scaling by `|w|` (and `|w|^2` in `e`).
- Returns `(aux, loss)`: `aux` from the unperturbed pass, detached; `loss` from
  the perturbed pass with autograd attached to the original parameters.
- Raises `ValueError` for negative `rho` or a model without trainable
  parameters. No parameters, buffers, or state-dict keys are added; no in-place
  modification of weights occurs.

## Invariants and equivalence evidence

- `tests/test_sharpness_aware.py` checks `rho = 0` equals the plain loss, the
  perturbation has norm `rho`, the gradient equals a manual two-step SAM
  (ascent, gradient at `w + e`), and the adaptive variant.
- Differs from the reference two-step optimizer only in floating-point order.

## Variants and options

`adaptive` only. Not provided: second-order terms, per-group `rho`, or
gradient-norm clipping.

## When to use and when not to use

Use for a paper that trains with SAM and whose runner already calls a custom
objective. Do not use with `torch.nn.DataParallel` (the runner forbids custom
objectives there), mixed-precision gradient scalers (the ascent gradient is not
unscaled), or modules that mutate parameters in `forward`. It costs two forward
and backward passes per batch.

## Related components

`revin`, `channel_wise_linear` (consumers' other blocks).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `sharpness_aware_loss(model: nn.Module, loss_fn: Callable[[Callable[..., torch.Tensor]], tuple[torch.Tensor | None, torch.Tensor]], rho: float, adaptive: bool=False, eps: float=1e-12)`
  Return ``(clean_aux, loss at w + e(w))`` for one SAM training step.

```python
from tsflab.models._components.sharpness_aware import sharpness_aware_loss
```

## Retrieval terms

`adversarial-weights`, `optimizer-agnostic`, `sam`, `sharpness`, `training-objective`, `functional-call`

## Current model consumers (1)

`samformer`
<!-- component-card:generated:end -->
