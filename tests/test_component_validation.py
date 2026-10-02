"""Argument-validation and edge-case contracts for shared components."""

from __future__ import annotations

import pytest
import torch

from tsflab.models._components.fft_extrapolation_conv import FFTExtrapolationConv
from tsflab.models._components.graph_masked_attention import GlobalLocalGraphAttention
from tsflab.models._components.hyper_state_scan import diagonal_selective_scan
from tsflab.models._components.revin import RevIN
from tsflab.models._components.topk_expert_attention import TopKExpertAttention
from tsflab.models._components.topk_expert_router import topk_dense_mix
from tsflab.models._components.transformer_encdec import (
    ConvLayer,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)
from tsflab.models._components.wavelet import DecimatedWaveletTransform
from tsflab.models._components.weight_set_router import mix_weight_sets


def test_topk_dense_mix_validation():
    weights = torch.softmax(torch.randn(2, 4), -1)
    for bad_k in (0, 5, -1):
        with pytest.raises(ValueError, match="k must be"):
            topk_dense_mix(weights, bad_k, 0.1)
    with pytest.raises(ValueError, match="floor"):
        topk_dense_mix(weights, 2, -0.1)
    torch.testing.assert_close(topk_dense_mix(weights, 4, 0.0), weights)


def test_topk_expert_attention_validation():
    with pytest.raises(ValueError, match="num_heads"):
        TopKExpertAttention(8, num_heads=0)
    with pytest.raises(ValueError, match="topk"):
        TopKExpertAttention(8, num_heads=2, topk=-1)
    with pytest.raises(ValueError, match="qk_dim"):
        TopKExpertAttention(8, num_heads=2, qk_dim=7)
    TopKExpertAttention(8, num_heads=2, topk=0, qk_dim=6)


def test_graph_masked_attention_validation():
    with pytest.raises(ValueError, match="num_heads"):
        GlobalLocalGraphAttention(8, num_heads=0)
    with pytest.raises(ValueError, match="divisible"):
        GlobalLocalGraphAttention(8, num_heads=3)


def test_diagonal_selective_scan_rejects_empty_length():
    empty = torch.zeros(1, 2, 0)
    with pytest.raises(ValueError, match="length"):
        diagonal_selective_scan(empty, empty, torch.zeros(2), empty)


def _attention(*_args, **_kwargs):
    return torch.zeros(1), None


def test_encoder_conv_layer_count_and_activation():
    layers = [EncoderLayer(_attention, 8, 16, 0.0) for _ in range(3)]
    with pytest.raises(ValueError, match="conv_layers"):
        Encoder(layers, [ConvLayer(8)])
    Encoder(layers, [ConvLayer(8), ConvLayer(8)])
    with pytest.raises(ValueError, match="activation"):
        EncoderLayer(_attention, 8, 16, 0.0, activation="swish")
    with pytest.raises(ValueError, match="activation"):
        DecoderLayer(_attention, _attention, 8, 16, 0.0, activation="ReLU")
    assert EncoderLayer(_attention, 8, 16, 0.0, activation="gelu").activation is not None


def test_revin_center_and_scale_share_axes_at_rank_four():
    revin = RevIN(3, affine=False, subtract_last=True)
    values = torch.randn(2, 5, 4, 3)
    out = revin(values, "norm")
    assert revin._center.shape == revin._scale.shape == (2, 1, 1, 3)
    expected_center = values[:, -1:].mean(dim=2, keepdim=True)
    torch.testing.assert_close(revin._center, expected_center)
    torch.testing.assert_close(revin(out, "denorm"), values, atol=1e-5, rtol=1e-5)
    rank3 = torch.randn(2, 5, 3)
    revin(rank3, "norm")
    torch.testing.assert_close(revin._center, rank3[:, -1:])


def test_decimated_wavelet_explicit_trims():
    transform = DecimatedWaveletTransform("haar", level=2)
    x = torch.randn(1, 2, 9)
    coeffs = transform.decompose(x)
    trims = list(transform.last_trims)
    transform.decompose(torch.randn(1, 2, 8))  # overwrites the recorded state
    torch.testing.assert_close(
        transform.reconstruct(coeffs, trims=trims), x, atol=1e-5, rtol=1e-5
    )
    with pytest.raises(ValueError, match="trim"):
        transform.reconstruct(coeffs, trims=[0])


def test_fft_extrapolation_conv_uses_mix_weight_sets():
    conv = FFTExtrapolationConv(8, 4, num_sets=2)
    routing = torch.softmax(torch.randn(2, 3), dim=0)
    weight = torch.complex(conv.real_weight, conv.imag_weight)
    expected = torch.einsum("sc,sf->cf", routing.to(weight.dtype), weight)
    torch.testing.assert_close(mix_weight_sets(weight, routing.to(weight.dtype)), expected)
    assert conv(torch.randn(1, 3, 8), routing).shape == (1, 3, 4)
