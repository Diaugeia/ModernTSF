"""Contract tests for the periodic_alibi_bias component."""

from __future__ import annotations

import pytest
import torch

from tsflab.models._components.periodic_alibi_bias import periodic_alibi_bias


def test_plain_alibi_for_one_group_matches_hand_values() -> None:
    bias = periodic_alibi_bias(4, 2)
    assert bias.shape == (2, 4, 4) and bias.dtype == torch.float32
    # n = 2 heads: slopes 2**-4 and 2**-8
    distance = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=torch.float32)
    torch.testing.assert_close(bias[0], -(2.0**-4) * distance)
    torch.testing.assert_close(bias[1], -(2.0**-8) * distance)


def test_power_of_two_heads_use_standard_alibi_slopes() -> None:
    bias = periodic_alibi_bias(3, 8)
    slopes = torch.tensor([2.0 ** -(k) for k in range(1, 9)])
    torch.testing.assert_close(bias[:, 0, 1], -slopes)


def test_periodic_group_is_a_triangle_wave_of_the_distance() -> None:
    bias = periodic_alibi_bias(8, 2, [4, None])
    # group 0 (head 0): n = 1, slope 2**-8; distances 0..7 mod 4 -> 0,1,2,1,0,1,2,1
    expected = torch.tensor([0, 1, 2, 1, 0, 1, 2, 1], dtype=torch.float32) * -(2.0**-8)
    torch.testing.assert_close(bias[0, 0], expected)
    # group 1 (head 1) is plain |i - j| with the same slope
    torch.testing.assert_close(bias[1, 0], torch.arange(8, dtype=torch.float32) * -(2.0**-8))


def test_bias_is_symmetric_nonpositive_zero_diagonal_and_bounded_by_half_period() -> None:
    bias = periodic_alibi_bias(20, 4, [6, 10], dtype=torch.float64)
    assert bias.dtype == torch.float64
    torch.testing.assert_close(bias, bias.transpose(1, 2))
    assert (bias <= 0).all()
    assert (bias.diagonal(dim1=1, dim2=2) == 0).all()
    slopes = 2.0 ** (-8.0 * torch.arange(1, 3, dtype=torch.float64) / 2)
    assert (bias[:2].min(dim=-1).values.min(dim=-1).values >= -slopes * 3 - 1e-12).all()  # P/2 = 3
    assert (bias[2:].min(dim=-1).values.min(dim=-1).values >= -slopes * 5 - 1e-12).all()  # P/2 = 5


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        periodic_alibi_bias(0, 2)
    with pytest.raises(ValueError):
        periodic_alibi_bias(4, 3, [2, 3])
    with pytest.raises(ValueError):
        periodic_alibi_bias(4, 2, [0, 2])
