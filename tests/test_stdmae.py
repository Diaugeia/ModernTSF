"""Paper-structure and reference-formula checks for the STD-MAE forecasting stage."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsflab.models.stdmae.model import (
    DecoupledMaskedEncoder,
    Model,
    sincos_2d,
)
from tsflab.models.stdmae.spec import SPEC, ModelParameterConfig


def tiny(**overrides) -> Model:
    options = dict(
        seq_len=24, pred_len=5, num_nodes=6, history_len=3, patch_size=4, embed_dim=16,
        num_heads=2, encoder_depth=1, residual_channels=8, dilation_channels=8,
        skip_channels=16, end_channels=16, blocks=1, layers=2, dropout=0.0,
        encoder_dropout=0.0,
    )
    options.update(overrides)
    torch.manual_seed(0)
    return Model(**options).eval()


def inputs(batch: int = 2, length: int = 24, nodes: int = 6):
    values = torch.randn(batch, length, nodes)
    marks = torch.zeros(batch, length, 6)
    marks[..., 3] = torch.arange(length) % 7  # weekday
    marks[..., 4] = torch.arange(length) % 24  # hour
    return values, marks


def test_forward_shape_and_finite() -> None:
    model = tiny()
    out = model(*inputs())
    assert out.shape == (2, 5, 6) and torch.isfinite(out).all()


def test_position_table_matches_axis_sincos_layout() -> None:
    table = sincos_2d(3, 5, 8)
    assert table.shape == (3, 5, 8)
    channels = 4  # ceil(8 / 4) * 2
    freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
    row = torch.arange(3).float()[:, None] * freq[None, :]
    expected_row = torch.stack((row.sin(), row.cos()), dim=-1).flatten(-2, -1)
    torch.testing.assert_close(table[:, 0, :channels], expected_row)
    assert torch.equal(table[:, 0, :channels], table[:, 4, :channels])  # row code ignores column
    col = torch.arange(5).float()[:, None] * freq[None, :]
    expected_col = torch.stack((col.sin(), col.cos()), dim=-1).flatten(-2, -1)
    torch.testing.assert_close(table[0, :, channels:], expected_col)
    odd = sincos_2d(2, 2, 6)  # width not divisible by four is truncated
    assert odd.shape == (2, 2, 6)


def test_temporal_encoder_attends_within_nodes_and_spatial_across_nodes() -> None:
    kwargs = dict(num_nodes=4, seq_len=16, patch_size=4, embed_dim=8, num_heads=2,
                  mlp_ratio=2, depth=1, dropout=0.0)
    temporal = DecoupledMaskedEncoder(**kwargs, spatial=False).eval()
    spatial = DecoupledMaskedEncoder(**kwargs, spatial=True).eval()
    x = torch.randn(2, 16, 4)
    other = x.clone()
    other[:, :, 3] += torch.randn(2, 16)  # perturb one node
    t_diff = (temporal(other) - temporal(x)).abs().amax(dim=(0, 2, 3))
    s_diff = (spatial(other) - spatial(x)).abs().amax(dim=(0, 2, 3))
    assert torch.all(t_diff[:3] < 1e-6) and t_diff[3] > 1e-4  # nodes never mix in T-MAE
    assert torch.all(s_diff[:3] > 1e-6)  # S-MAE mixes nodes within each patch
    assert temporal(x).shape == (2, 4, 4, 8)  # (B, N, P, d)


def test_patch_embedding_equals_linear_on_non_overlapping_patches() -> None:
    encoder = DecoupledMaskedEncoder(2, 8, 4, 6, 2, 2, 1, 0.0, spatial=False)
    x = torch.randn(1, 8, 2)
    series = x[0, :, 0]
    first_patch = series[:4]
    weight = encoder.patch_embedding.weight[:, 0, :, 0]  # (d, patch)
    expected = weight @ first_patch + encoder.patch_embedding.bias
    got = encoder.patch_embedding(x.transpose(1, 2).reshape(2, 1, 8, 1))[0, :, 0, 0]
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_context_heads_inject_into_the_skip_path_before_the_head() -> None:
    model = tiny()
    values, marks = inputs()
    base = model(values, marks)
    with torch.no_grad():
        for head in (model.temporal_context, model.spatial_context):
            head[2].weight.zero_()
            head[2].bias.zero_()
    assert not torch.allclose(model(values, marks), base)
    # with zeroed contexts the long history (outside the short window) is irrelevant
    changed = values.clone()
    changed[:, :-3] = torch.randn(2, 21, 6)
    torch.testing.assert_close(model(changed, marks), model(values, marks))


def test_short_history_is_the_last_history_len_steps_with_two_features() -> None:
    model = tiny()
    assert model.in_dim == 2 and model.history_len == 3 and model.receptive_field == 4
    values, marks = inputs()
    other_marks = marks.clone()
    other_marks[..., 3] = (other_marks[..., 3] + 3) % 7  # day-of-week is the third feature
    torch.testing.assert_close(model(values, other_marks), model(values, marks))
    other_marks[..., 4] = (other_marks[..., 4] + 5) % 24  # time-of-day is used
    assert not torch.allclose(model(values, other_marks), model(values, marks))


def test_receptive_field_and_final_length_for_default_architecture() -> None:
    model = Model(seq_len=24, pred_len=12, num_nodes=4, patch_size=12, embed_dim=8,
                  num_heads=2, encoder_depth=1, residual_channels=4, dilation_channels=4,
                  skip_channels=8, end_channels=8)
    assert model.receptive_field == 13  # 1 + 4 blocks * (1 + 2)
    assert model(*inputs(1, 24, 4)).shape == (1, 12, 4)


def test_graph_supports_come_from_adjacency_and_adaptive_embedding() -> None:
    adjacency = np.random.default_rng(0).random((6, 6)).astype(np.float32)
    model = tiny(adj_mx=adjacency)
    assert model.forward_support.shape == (6, 6)
    assert torch.allclose(model.forward_support.sum(1), torch.ones(6), atol=1e-5)
    assert model.source_nodes.shape == (6, 10) and model.target_nodes.shape == (10, 6)
    assert len(model.blocks) == 2 and all(b.graph.support_len == 3 for b in model.blocks)


def test_block_uses_valid_dilated_convolution_with_shrinking_time() -> None:
    model = tiny()
    x = torch.randn(1, 8, 6, 4)
    supports = [model.forward_support, model.reverse_support, torch.eye(6)]
    out, skip = model.blocks[0](x, supports)
    assert out.shape == (1, 8, 6, 3) and skip.shape == (1, 16, 6, 3)  # dilation 1, kernel 2
    out2, _ = model.blocks[1](out, supports)
    assert out2.shape[-1] == 1  # dilation 2


def test_gradients_reach_encoders_and_backend_when_trainable() -> None:
    model = tiny().train()
    model(*inputs()).sum().backward()
    for module in (model.temporal_encoder, model.spatial_encoder, model.start_conv,
                   model.temporal_context, model.spatial_context, model.end_conv_2):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads)
    assert model.source_nodes.grad is not None


def test_frozen_encoders_receive_no_gradient_and_stay_in_eval() -> None:
    model = tiny(freeze_encoders=True).train()
    assert not model.temporal_encoder.training and not model.spatial_encoder.training
    assert model.blocks[0].training
    model(*inputs()).sum().backward()
    assert all(not p.requires_grad for p in model.temporal_encoder.parameters())
    assert all(p.grad is None for p in model.spatial_encoder.parameters())
    assert model.end_conv_2.weight.grad is not None


def test_validation_and_spec() -> None:
    with pytest.raises(ValueError):
        tiny(seq_len=25)
    with pytest.raises(ValueError):
        tiny(history_len=4)  # exceeds receptive field - 1
    with pytest.raises(ValueError):
        tiny(embed_dim=15)
    with pytest.raises(ValueError):
        tiny()(torch.randn(2, 24, 5))
    assert SPEC.capabilities == frozenset({"spatiotemporal"}) and "tst_transformer" in SPEC.components
    assert ModelParameterConfig(enc_in=3).model_dump()["patch_size"] == 12
