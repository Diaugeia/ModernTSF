"""Numerical fixes in shared components: masked attention rows, band boundaries, band tiling."""

from __future__ import annotations

import pytest
import torch

from tsflab.models._components.freq_band_moe import FrequencyBandMixtureOfExperts
from tsflab.models._components.frequency_band_sampler import HierarchicalFrequencySampler
from tsflab.models._components.graph_masked_attention import GlobalLocalGraphAttention


def test_fully_masked_attention_row_is_zero_not_nan() -> None:
    torch.manual_seed(0)
    attn = GlobalLocalGraphAttention(model_dim=8, num_heads=2).eval()
    x = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[1] = False  # query 1 sees nothing
    out = attn(x, x, x, adj_mask=mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, 1], attn.out_proj.bias.expand(2, 8))
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_masked_attention_rows_with_visible_keys_unchanged() -> None:
    torch.manual_seed(1)
    attn = GlobalLocalGraphAttention(model_dim=8, num_heads=2).eval()
    x = torch.randn(3, 5, 8)
    mask = torch.rand(5, 5) > 0.5
    mask |= torch.eye(5, dtype=torch.bool)
    out = attn(x, x, x, adj_mask=mask)
    q, k, v = (attn._split_heads(f(x)) for f in (attn.fc_q, attn.fc_k, attn.fc_v))
    scores = q @ k.transpose(-1, -2) / attn.head_dim**0.5
    ref = (scores.softmax(-1) + scores.masked_fill(~mask, float("-inf")).softmax(-1)) / 2
    expected = attn.out_proj(attn._merge_heads(ref @ v, [3], 5))
    assert torch.equal(out, expected)
    # Mixed: masking one row leaves the others bit-identical.
    partial = mask.clone()
    partial[2] = False
    out2 = attn(x, x, x, adj_mask=partial)
    keep = [0, 1, 3, 4]
    assert torch.equal(out2[:, keep], out[:, keep])


def test_band_boundaries_default_is_buffer_and_loads_old_checkpoint() -> None:
    moe = FrequencyBandMixtureOfExperts(4, 32)
    assert "band_boundaries" not in dict(moe.named_parameters())
    assert "band_boundaries" in dict(moe.named_buffers())
    x = torch.randn(2, 3, 32)
    moe(x)[0].sum().backward()
    assert moe.band_boundaries.grad is None
    state = moe.state_dict()
    assert "band_boundaries" in state  # same key as the former Parameter
    other = FrequencyBandMixtureOfExperts(4, 32)
    other.load_state_dict(state)
    assert torch.equal(other(x)[0], moe(x)[0])
    # A learnable module's checkpoint (Parameter key) also loads into the default buffer module.
    learnable = FrequencyBandMixtureOfExperts(4, 32, learnable_boundaries=True)
    other.load_state_dict(learnable.state_dict())
    assert torch.equal(other.band_boundaries, learnable.band_boundaries.detach())


def test_learnable_boundaries_receive_gradient_with_identical_forward() -> None:
    torch.manual_seed(2)
    fixed = FrequencyBandMixtureOfExperts(4, 32)
    learn = FrequencyBandMixtureOfExperts(4, 32, learnable_boundaries=True)
    learn.load_state_dict(fixed.state_dict())
    x = torch.randn(2, 3, 32)
    out_fixed, b_fixed, g_fixed = fixed(x)
    out_learn, b_learn, g_learn = learn(x)
    assert torch.equal(out_fixed, out_learn)
    assert torch.equal(b_fixed, b_learn.detach())
    assert torch.equal(g_fixed, g_learn)
    (out_learn**2).sum().backward()
    grad = learn.band_boundaries.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_single_expert_has_no_boundaries() -> None:
    moe = FrequencyBandMixtureOfExperts(1, 16, learnable_boundaries=True)
    out, bounds, _ = moe(torch.randn(1, 2, 16))
    assert bounds.tolist() == [0.0, 1.0] and torch.isfinite(out).all()


@pytest.mark.parametrize("bins", [1, 5, 17, 49, 97, 100])
@pytest.mark.parametrize("layers", [1, 2, 3, 4, 7])
def test_sampler_tiling_covers_spectrum_exactly(bins: int, layers: int) -> None:
    if bins < layers:
        pytest.skip("fewer bins than layers cannot tile without empty bands")
    sampler = HierarchicalFrequencySampler(layers, alpha=1.0 / layers)
    bands = [sampler.band(bins, i) for i in range(layers)]
    assert bands[0][1] == bins and bands[-1][0] == 0
    for upper, lower in zip(bands[:-1], bands[1:]):
        assert lower[1] == upper[0]  # adjacent bands meet exactly
    covered = sorted(b for s, e in bands for b in range(s, e))
    assert covered == list(range(bins))


def test_sampler_sliding_regime_reaches_both_ends() -> None:
    for bins in (17, 49, 50):
        for alpha in (0.3, 0.55, 1.0):
            sampler = HierarchicalFrequencySampler(3, alpha=alpha)
            bands = [sampler.band(bins, i) for i in range(3)]
            assert bands[0][1] == bins and bands[-1][0] == 0
            assert all(0 <= s < e <= bins for s, e in bands)
    assert HierarchicalFrequencySampler(3, alpha=1.0).band(49, 1) == (0, 49)
