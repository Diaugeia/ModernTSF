"""MoU: mixture of feature extractors, mixture-of-architectures block order, shapes."""

from __future__ import annotations

import pytest
import torch

from tsflab.models.mou.model import (
    MixtureOfArchitecturesBlock,
    MixtureOfFeatureExtractors,
    Model,
)


def _small(**kwargs) -> Model:
    return Model(32, 6, 3, patch_len=8, stride=4, d_model=16, n_heads=4, d_ff=32, d_state=4, **kwargs)


def test_mof_gates_are_sparse_normalized_and_match_expert_mix() -> None:
    torch.manual_seed(0)
    mof = MixtureOfFeatureExtractors(8, 16, experts=4, top_k=2).eval()
    with torch.no_grad():
        mof.w_gate.normal_()
    x = torch.randn(5, 7, 8)
    gates = mof.gates(x)
    assert ((gates > 0).sum(-1) == 2).all()
    assert torch.allclose(gates.sum(-1), torch.ones(5, 7), atol=1e-5)
    expected = sum(gates[..., i : i + 1] * mof.extractors[i](x) for i in range(4))
    assert torch.allclose(mof(x), expected, atol=1e-6)


def test_mof_noise_only_in_training_and_zero_init_gate_is_uniform() -> None:
    mof = MixtureOfFeatureExtractors(4, 8, experts=4, top_k=4)
    x = torch.randn(3, 2, 4)
    mof.eval()
    assert torch.allclose(mof.gates(x), torch.full((3, 2, 4), 0.25), atol=1e-5)
    mof.train()
    assert not torch.equal(mof.gates(x), mof.gates(x))


def test_moa_block_applies_layers_in_paper_order() -> None:
    torch.manual_seed(0)
    block = MixtureOfArchitecturesBlock(16, 4, 32, 4, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).eval()
    x = torch.randn(2, 5, 16)
    h = block.mamba_norm(x + block.mamba(x))
    h = block.shared_norm(h + block.ffn(h))
    conv = block.conv_norm(h + block.conv(h.transpose(1, 2)).transpose(1, 2))
    h = block.shared_norm(h + conv)
    assert torch.allclose(block(x), block.attention(h), atol=1e-5)


def test_model_shapes_patch_count_and_validation() -> None:
    m = _small().eval()
    assert m.position.shape == ((32 - 8) // 4 + 2, 16)
    assert m(torch.randn(2, 32, 3)).shape == (2, 6, 3)
    with pytest.raises(ValueError):
        Model(32, 6, 3, d_model=10, n_heads=4)
    with pytest.raises(ValueError):
        Model(32, 6, 3, top_k=5, num_experts=4)
    with pytest.raises(ValueError):
        m(torch.randn(2, 30, 3))


def test_channels_are_independent() -> None:
    torch.manual_seed(0)
    m = _small().eval()
    x = torch.randn(2, 32, 3)
    y = m(x)
    x2 = x.clone()
    x2[..., 2] = torch.randn(2, 32)
    assert torch.allclose(m(x2)[..., :2], y[..., :2], atol=1e-5)
