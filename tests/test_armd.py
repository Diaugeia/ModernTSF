"""Paper-structure and reference-formula checks for ARMD."""

from __future__ import annotations

import pytest
import torch

from tsflab.benchmark.runner.objective import TrainingBatch
from tsflab.models.armd.model import Model, cosine_beta_schedule, linear_beta_schedule
from tsflab.models.armd.spec import SPEC, training_objective


def tiny(**overrides) -> Model:
    options = dict(seq_len=12, pred_len=8, enc_in=3, sampling_steps=2)
    options.update(overrides)
    torch.manual_seed(0)
    return Model(**options)


def test_devolution_network_follows_equation_five() -> None:
    model = tiny()
    state = torch.randn(2, 8, 3)
    step = 5
    weight = model.step_weight[step]
    distance = model.temporal(state.transpose(1, 2)).transpose(1, 2)
    expected = (weight * state + (1 - 2 * weight) * distance) / torch.sqrt(1 - weight)
    torch.testing.assert_close(model.devolve(state, step), expected)


def test_step_weight_starts_at_linear_schedule_cumulative_product() -> None:
    model = tiny()
    expected = torch.cumprod(1 - linear_beta_schedule(8), dim=0).float()
    torch.testing.assert_close(model.step_weight.detach(), expected)
    assert model.step_weight.requires_grad
    assert not tiny(w_grad=False).step_weight.requires_grad


def test_trend_matches_equation_three() -> None:
    model = tiny()
    future, state = torch.randn(2, 8, 3), torch.randn(2, 8, 3)
    abar = model.alphas_cumprod[3]
    expected = (state / abar.sqrt() - future) / torch.sqrt(1 / abar - 1)
    torch.testing.assert_close(model._trend(state, future, 3), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        model._trend(state, state * model.sqrt_alphas_cumprod[3] ** -1, 3),
        torch.zeros_like(state), atol=1e-4, rtol=1e-4,
    )


def test_intermediate_state_slides_from_future_toward_history() -> None:
    model = tiny()
    series = torch.arange(16.0).view(1, 16, 1)
    assert torch.equal(model.intermediate_state(series, 7), series[:, :8])  # history
    assert model.intermediate_state(series, 0).flatten().tolist() == list(range(7, 15))
    assert model.intermediate_state(series, 3).flatten().tolist() == list(range(4, 12))


def test_training_loss_is_finite_scalar_with_gradients() -> None:
    model = tiny().train()
    history, future = torch.randn(4, 12, 3), torch.randn(4, 8, 3)
    for _ in range(5):  # different diffusion steps
        loss = model.training_loss(history, future)
        assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.temporal.weight.grad is not None
    assert model.step_weight.grad is not None and model.step_weight.grad.abs().sum() > 0


def test_training_loss_matches_manual_formula_at_fixed_step() -> None:
    model = tiny(loss_type="l2").train()
    history, future = torch.randn(2, 12, 3), torch.randn(2, 8, 3)
    torch.manual_seed(5)
    loss = model.training_loss(history, future)
    torch.manual_seed(5)
    step = int(torch.randint(0, 8, (1,)))
    series = torch.cat([history[:, -8:], future], dim=1)
    state = model.intermediate_state(series, step)
    predicted = model.devolve(state, step, perturb=True)
    a, b = model.sqrt_alphas_cumprod[step], model.sqrt_one_minus_alphas_cumprod[step]
    error = ((state - predicted * a) / b - (state - future * a) / b) ** 2
    expected = error.flatten(1).mean(1) * model.loss_weight[step]
    torch.testing.assert_close(loss, expected.mean())


def test_single_step_forecast_is_devolution_of_the_last_history_window() -> None:
    model = tiny(sampling_steps=1).eval()
    x = torch.randn(2, 12, 3)
    torch.testing.assert_close(model(x), model.devolve(x[:, -8:], 7))
    changed = x.clone()
    changed[:, :4] += torch.randn(2, 4, 3)  # older than the last pred_len steps
    torch.testing.assert_close(model(changed), model(x))


def test_two_step_sampling_matches_reference_ddim_recursion() -> None:
    model = tiny(sampling_steps=2).eval()
    x = torch.randn(2, 12, 3)
    state = x[:, -8:]
    # times = linspace(-1, 7, 3) -> [-1, 3, 7]: steps (7 -> 3) then (3 -> -1)
    start = model.devolve(state, 7)
    trend = (model.sqrt_recip_alphas_cumprod[7] * state - start) / model.sqrt_recipm1_alphas_cumprod[7]
    nxt = model.alphas_cumprod[3]
    state = start * nxt.sqrt() + (1 - nxt).sqrt() * trend
    torch.testing.assert_close(model(x), model.devolve(state, 3))


def test_forecast_is_deterministic_and_mode_independent() -> None:
    model = tiny()
    x = torch.randn(2, 12, 3)
    model.train()
    first = model(x)
    torch.testing.assert_close(model(x), first)
    torch.testing.assert_close(model.eval()(x), first)


def test_cosine_schedule_buffers_have_pred_len_steps() -> None:
    model = tiny()
    assert model.alphas_cumprod.shape == (8,)
    assert torch.all(model.alphas_cumprod[1:] <= model.alphas_cumprod[:-1])
    torch.testing.assert_close(
        model.sqrt_one_minus_alphas_cumprod**2 + model.sqrt_alphas_cumprod**2,
        torch.ones(8),
    )
    assert cosine_beta_schedule(8).max() <= 0.999


def test_training_objective_replaces_criterion_and_returns_no_forecast() -> None:
    model = tiny().train()
    x, y = torch.randn(3, 12, 3), torch.randn(3, 8, 3)
    batch = TrainingBatch(
        x=x, x_mark=None, dec_inp=torch.zeros(3, 8, 3), y_mark=None,
        y=y, pred_len=8, features="M",
    )
    forecast, loss = training_objective(model, batch, criterion=None)
    assert forecast is None and loss.ndim == 0 and torch.isfinite(loss)
    assert SPEC.training_objective is training_objective and SPEC.components == ()


def test_validation() -> None:
    with pytest.raises(ValueError):
        Model(seq_len=4, pred_len=8, enc_in=2)
    with pytest.raises(ValueError):
        Model(seq_len=8, pred_len=8, enc_in=2, sampling_steps=9)
    with pytest.raises(ValueError):
        tiny()(torch.randn(2, 12, 4))
    with pytest.raises(ValueError):
        tiny().training_loss(torch.randn(2, 12, 3), torch.randn(2, 7, 3))
