"""Contract tests for the softmax_gate component."""

from __future__ import annotations

import pytest
import torch

from tests.component_reference import assert_reference
from tsflab.models._components.softmax_gate import SoftmaxGate


def test_softmax_gate_contract_and_reference() -> None:
    torch.manual_seed(0)
    gate = SoftmaxGate(5)
    assert set(gate.state_dict()) == {"score.weight", "score.bias"}
    x = torch.randn(2, 3, 5, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
    out = gate(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    weights = torch.softmax(gate.score(x), dim=-1)
    torch.testing.assert_close(out, x * weights)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3))
    assert (out.abs() <= x.abs() + 1e-6).all()
    out.square().sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in gate.parameters())
    assert gate(torch.randn(7, 5)).shape == (7, 5)
    with pytest.raises(ValueError):
        gate(torch.randn(2, 4))
    with pytest.raises(ValueError):
        SoftmaxGate(0)
    assert_reference("softmax_gate", {"out": out, "x_grad": x.grad})
