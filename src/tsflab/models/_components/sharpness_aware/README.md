---
name: "sharpness_aware"
kind: "component"
module: "tsflab.models._components.sharpness_aware"
summary: "First-order sharpness-aware minimization (SAM/ASAM) as a loss evaluated at adversarially perturbed weights, usable with any optimizer through ModelSpec.training_objective."
category: "utility"
input: "model nn.Module; loss_fn(run) -> (aux or None, scalar loss); rho float >= 0; adaptive bool; eps float"
output: "(aux detached or None, scalar loss that differentiates to the SAM gradient)"
origin: "SAM, Foret et al., ICLR 2021 (Sharpness-Aware Minimization for Efficiently Improving Generalization); adaptive form from ASAM, Kwon et al., ICML 2021"
origin_models: ["samformer"]
tags: ["sam", "sharpness", "training-objective", "adversarial-weights", "functional-call", "optimizer-agnostic", "asam", "loss-wrapper", "stateless"]
---

# sharpness_aware

## Purpose

SAM updates `w` with `grad L(w + e(w))`, where
`e(w) = rho * g / (||g||_2 + eps)` and `g = grad L(w)` over all trainable
parameters jointly. `sharpness_aware_loss(model, loss_fn, rho)` runs one ascent
pass to obtain `g`, detaches `e`, then evaluates the loss again with parameters
`w + e` through `torch.func.functional_call`. Backpropagating that loss gives,
with respect to the stored `w`, the gradient the reference SAM optimizer applies
after its second step, so the usual `loss.backward(); optimizer.step()` loop (any
base optimizer, weight decay included) realizes SAM without a wrapper optimizer.
With `adaptive=True` the ASAM form is used: the norm is `|| |w| * g ||_2` and
`e = rho * w^2 * g / (|| |w| * g ||_2 + eps)` (element-wise).

## Origin and granularity

Cut at the arithmetic only: the caller supplies `loss_fn`, which decides how a
batch becomes a loss, so the same component serves any model through the runner's
`ModelSpec.training_objective` hook. SAMformer is the only consumer today (its
`sharpness_aware_objective` method passes `rho`). The component was added with
SAMformer (commit `1a87e79b`). Choosing `rho`, reading the batch, and which
criterion is minimized stay in the model.

## Interface

`sharpness_aware_loss(model, loss_fn, rho, adaptive=False, eps=1e-12)`. The only
public symbol.

- `model`: `nn.Module`; only parameters with `requires_grad` are perturbed
  (parameters whose ascent gradient is `None` are left unperturbed). The
  function does not set train/eval mode; the runner calls it in training mode.
- `loss_fn(run)`: called twice (once with `run = model`, once with `run` a
  functional call that uses the perturbed parameters). It must call `run(...)`
  where it would call `model(...)` (only the forward call is redirected, not
  other methods) and return `(aux, loss)` with `loss` a scalar tensor; `aux` may
  be `None` or a tensor (for example the forecast).
- `rho` (float >= 0): neighborhood radius. `rho == 0` performs one ordinary pass
  (and skips the trainable-parameter check).
- `adaptive` (bool): ASAM-style scaling by `|w|` as in Purpose. `eps` (float):
  added to the gradient norm.
- Returns `(aux, loss)`: `aux` from the unperturbed pass, detached (`None` stays
  `None`); `loss` from the perturbed pass with autograd attached to the original
  parameters.
- Raises `ValueError` for negative `rho` and, when `rho > 0`, for a model without
  trainable parameters. No parameters, buffers, or state-dict keys are added; no
  in-place modification of parameters occurs. Buffers are not perturbed, but a
  module that updates buffers in `forward` (for example BatchNorm running
  statistics) updates them in both passes, and stochastic layers such as dropout
  draw independent masks in the two passes.

## Invariants and equivalence evidence

- `tests/test_sharpness_aware.py` checks that `rho = 0` equals the plain loss,
  that the loss and parameter gradients equal a manual two-step SAM (ascent,
  gradient at `w + e`, restore) for both `adaptive=False` and `adaptive=True`, that
  parameters are unchanged and `aux` carries no grad, that the non-adaptive
  perturbation has norm `rho`, that the perturbed loss is not smaller than the
  clean loss for a small `rho`, and the two `ValueError` cases.
- no fixture: the comparison reference is the manual two-step procedure written
  in the test, not the official SAM/ASAM optimizer; any difference from that
  optimizer is expected to be floating-point order only (not measured).
- `tests/test_samformer.py` (`test_training_objective_is_sam_over_configured_criterion`,
  `test_spec_declares_objective_and_components`) exercises the integration through
  `samformer`'s registered objective.

## Variants and options

`adaptive` only. Not provided: second-order terms, per-group `rho`, or
gradient-norm clipping.

## When to use and when not to use

Use for a paper that trains with SAM and whose runner already calls a custom
objective. Do not use with `torch.nn.DataParallel` (the runner raises for custom
objectives there), with modules that mutate parameters in `forward`, or when two
independent dropout masks per step are unacceptable. Under mixed precision the
runner scales only the returned loss; the ascent gradient comes from the
unscaled loss, so it can underflow in float16. It costs two forward passes and
two backward passes per batch.

## Related components

None overlaps in responsibility: no other component wraps a training loss. The
other custom-objective models (`dipelinear`'s SFALoss, `distdf`, `mmpd`) keep their
objectives model-local. The remaining `samformer` components, `revin` and
`channel_wise_linear`, are architecture blocks and unrelated to this wrapper.

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
