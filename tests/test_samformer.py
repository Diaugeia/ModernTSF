"""SAMformer: equations, official-architecture agreement and the SAM objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tsflab.benchmark.runner.objective import TrainingBatch
from tsflab.models.samformer.model import Model
from tsflab.models.samformer.spec import SPEC, training_objective


def _official_forward(m: Model, x: torch.Tensor) -> torch.Tensor:
    """Reference transcription of the paper's network (Eq. in section 4, official layout)."""
    w, b, eps = m.revin.affine_weight, m.revin.affine_bias, m.revin.eps
    mean = x.mean(1, keepdim=True)
    std = torch.sqrt(x.var(1, unbiased=False, keepdim=True) + eps)
    xn = ((x - mean) / std * w + b).transpose(1, 2)  # (n, D, L)
    q, k, v = m.attention.query(xn), m.attention.key(xn), m.attention.value(xn)
    att = F.scaled_dot_product_attention(q, k, v)
    out = m.forecaster.linear(xn + att).transpose(1, 2)
    return (out - b) / (w + eps * eps) * std + mean


def test_forward_matches_official_architecture() -> None:
    torch.manual_seed(0)
    m = Model(24, 6, 3, hid_dim=8).eval()
    with torch.no_grad():
        m.revin.affine_weight.copy_(torch.rand(3) + 0.5)
        m.revin.affine_bias.copy_(torch.randn(3))
    x = torch.randn(4, 24, 3) * 3 + 1
    assert torch.allclose(m(x), _official_forward(m, x), atol=1e-5)


def test_attention_is_across_channels_with_temporal_tokens() -> None:
    m = Model(16, 4, 5, hid_dim=6)
    assert m.attention.query.weight.shape == (6, 16)
    assert m.attention.key.weight.shape == (6, 16)
    assert m.attention.value.weight.shape == (16, 16)


def test_revin_off_is_plain_attention_plus_linear() -> None:
    torch.manual_seed(0)
    m = Model(12, 3, 2, use_revin=False).eval()
    x = torch.randn(3, 12, 2)
    xt = x.transpose(1, 2)
    ref = m.forecaster(xt + m.attention(xt)).transpose(1, 2)
    assert torch.allclose(m(x), ref, atol=1e-6)


def test_training_objective_is_sam_over_configured_criterion() -> None:
    torch.manual_seed(0)
    m = Model(12, 3, 2, rho=0.5)
    x, y = torch.randn(5, 12, 2), torch.randn(5, 3, 2)
    batch = TrainingBatch(x=x, x_mark=None, dec_inp=torch.zeros_like(y), y_mark=None, y=y, pred_len=3, features="M")
    mse = torch.nn.functional.mse_loss
    forecast, loss = training_objective(m, batch, mse)
    loss.backward()
    assert forecast.shape == y.shape
    # manual SAM step: ascent on w, gradient at w + e
    params = list(m.parameters())
    plain = mse(m(x), y)
    grads = torch.autograd.grad(plain, params)
    norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    adv_grads_ref = None
    with torch.no_grad():
        es = [g * 0.5 / (norm + 1e-12) for g in grads]
        for p, e in zip(params, es):
            p.add_(e)
    adv = mse(m(x), y)
    adv_grads_ref = torch.autograd.grad(adv, params)
    with torch.no_grad():
        for p, e in zip(params, es):
            p.sub_(e)
    for p, g in zip(params, adv_grads_ref):
        assert torch.allclose(p.grad, g, atol=1e-6)
    assert loss.item() >= plain.item() - 1e-6


def test_spec_declares_objective_and_components() -> None:
    assert SPEC.training_objective is training_objective
    assert {"revin", "channel_wise_linear", "sharpness_aware"} <= set(SPEC.components)
