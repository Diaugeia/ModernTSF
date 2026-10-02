"""AMD: MDM/DDI/AMS equations, gate behaviour and the auxiliary balance loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tsflab.models.amd.model import (
    AdaptiveMultiPredictorSynthesis,
    DualDependencyInteraction,
    Model,
    MultiScaleDecomposableMixing,
    NoisyTopKGate,
)


def test_mdm_mixes_pooled_scales_top_down() -> None:
    torch.manual_seed(0)
    mdm = MultiScaleDecomposableMixing(32, 3, k=2, c=2, batchnorm=False).eval()
    x = torch.randn(4, 3, 32)
    s4, s2 = F.avg_pool1d(x, 4), F.avg_pool1d(x, 2)
    s2 = s2 + mdm.mixers[0](s4)
    expected = x + mdm.mixers[1](s2)
    assert mdm.windows == [4, 2]
    assert torch.allclose(mdm(x), expected, atol=1e-6)


def test_ddi_is_sequential_over_previous_output_patch() -> None:
    torch.manual_seed(0)
    ddi = DualDependencyInteraction(12, 2, patch=4, alpha=0.0, dropout=0.0, batchnorm=False).eval()
    x = torch.randn(3, 2, 12)
    out = ddi(x)
    assert torch.equal(out[..., :4], x[..., :4])
    second = F.gelu(ddi.aggregate(ddi.patch_norm(out[..., :4]))) + x[..., 4:8]
    assert torch.allclose(out[..., 4:8], second, atol=1e-6)
    third = F.gelu(ddi.aggregate(ddi.patch_norm(out[..., 4:8]))) + x[..., 8:]
    assert torch.allclose(out[..., 8:], third, atol=1e-6)


def test_ddi_channel_mlp_branch_is_alpha_scaled_residual() -> None:
    torch.manual_seed(0)
    ddi = DualDependencyInteraction(8, 3, patch=4, alpha=0.5, dropout=0.0, batchnorm=False).eval()
    x = torch.randn(2, 3, 8)
    out = ddi(x)
    res = F.gelu(ddi.aggregate(ddi.patch_norm(x[..., :4]))) + x[..., 4:]
    branch = ddi.channel_mlp(ddi.channel_norm(res).transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(out[..., 4:], res + 0.5 * branch, atol=1e-6)


def test_gate_is_a_distribution_and_favors_topk() -> None:
    torch.manual_seed(0)
    gate = NoisyTopKGate(10, 6, top_k=2).eval()
    f = torch.randn(5, 4, 10)
    w = gate(f)
    assert torch.allclose(w.sum(-1), torch.ones(5, 4), atol=1e-6)
    logits = gate.gate(f)
    top = logits.topk(2, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, top, True)
    assert (w[mask].reshape(5, 4, 2).min(-1).values >= w[~mask].reshape(5, 4, 4).max(-1).values).all()
    gate.train()
    assert not torch.equal(gate(f), gate(f))  # noisy in training only


def test_ams_matches_per_channel_loop_and_importance_loss() -> None:
    torch.manual_seed(0)
    ams = AdaptiveMultiPredictorSynthesis(8, 5, experts=4, top_k=2, ff_dim=16, dropout=0.0, loss_coef=1.0).eval()
    x, emb = torch.randn(6, 3, 8), torch.randn(6, 3, 8)
    out, loss = ams(x, emb)
    expected_loss = 0.0
    for c in range(3):
        g = ams.gating(emb[:, c])  # [B, E]
        o = torch.stack([e(x[:, c]) for e in ams.experts], dim=1)  # [B, E, H]
        assert torch.allclose(out[:, c], (g.unsqueeze(-1) * o).sum(1), atol=1e-5)
        imp = g.sum(0)[:, None].expand(4, 5)  # replicated over the horizon
        expected_loss += imp.var() / (imp.mean() ** 2 + 1e-10)
    assert torch.allclose(loss, expected_loss, atol=1e-5)


def test_model_sets_aux_loss_only_in_training_and_validates() -> None:
    torch.manual_seed(0)
    m = Model(32, 6, 3, patch=8, mix_layer_num=2, num_experts=4, ff_dim=16)
    x = torch.randn(4, 32, 3)
    m.train()
    assert m(x).shape == (4, 6, 3) and m.aux_loss is not None and m.aux_loss.ndim == 0
    m.eval()
    m(x)
    assert m.aux_loss is None
    with pytest.raises(ValueError):
        Model(30, 6, 3, patch=8)
    with pytest.raises(ValueError):
        Model(32, 6, 3, patch=8, top_k=9)
