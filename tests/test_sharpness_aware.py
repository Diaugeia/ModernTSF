"""Contract tests for the sharpness_aware component."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tsflab.models._components.sharpness_aware import sharpness_aware_loss


def _setup():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    def loss_fn(run):
        out = run(x)
        return out, ((out - y) ** 2).mean()

    return model, x, y, loss_fn


def _manual_sam_grads(model, x, y, rho, adaptive=False):
    params = list(model.parameters())
    loss = ((model(x) - y) ** 2).mean()
    grads = torch.autograd.grad(loss, params)
    norm = torch.sqrt(sum(((p.abs() if adaptive else 1.0) * g).pow(2).sum() for p, g in zip(params, grads)))
    es = [(p.pow(2) if adaptive else 1.0) * g * rho / (norm + 1e-12) for p, g in zip(params, grads)]
    with torch.no_grad():
        for p, e in zip(params, es):
            p.add_(e)
    adv = ((model(x) - y) ** 2).mean()
    adv_grads = torch.autograd.grad(adv, params)
    with torch.no_grad():
        for p, e in zip(params, es):
            p.sub_(e)
    return adv, adv_grads, es


def test_rho_zero_is_plain_loss() -> None:
    model, x, y, loss_fn = _setup()
    aux, loss = sharpness_aware_loss(model, loss_fn, 0.0)
    assert torch.allclose(loss, ((model(x) - y) ** 2).mean())
    assert aux is not None


@pytest.mark.parametrize("adaptive", [False, True])
def test_gradient_matches_manual_two_step_sam(adaptive: bool) -> None:
    model, x, y, loss_fn = _setup()
    before = [p.detach().clone() for p in model.parameters()]
    aux, loss = sharpness_aware_loss(model, loss_fn, 0.3, adaptive=adaptive)
    loss.backward()
    assert not aux.requires_grad
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p.detach(), b)  # weights untouched
    adv, adv_grads, es = _manual_sam_grads(model, x, y, 0.3, adaptive)
    assert torch.allclose(loss, adv, atol=1e-6)
    for p, g in zip(model.parameters(), adv_grads):
        assert torch.allclose(p.grad, g, atol=1e-6)
    if not adaptive:
        total = torch.sqrt(sum(e.pow(2).sum() for e in es))
        assert torch.allclose(total, torch.tensor(0.3), atol=1e-6)


def test_perturbed_loss_is_not_smaller_than_clean_loss_for_small_rho() -> None:
    model, x, y, loss_fn = _setup()
    _, clean = sharpness_aware_loss(model, loss_fn, 0.0)
    _, adv = sharpness_aware_loss(model, loss_fn, 0.05)
    assert adv.item() >= clean.item() - 1e-7


def test_invalid_arguments() -> None:
    model, _, _, loss_fn = _setup()
    with pytest.raises(ValueError):
        sharpness_aware_loss(model, loss_fn, -1.0)
    frozen = nn.Linear(2, 2).requires_grad_(False)
    with pytest.raises(ValueError):
        sharpness_aware_loss(frozen, lambda run: (None, run(torch.zeros(1, 2)).sum()), 0.1)
