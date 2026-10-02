"""Equivalence tests for the ``topk_expert_router`` extraction.

DUET and DynamicTMoE previously implemented their own gated top-k expert
mixing inline. Both were refactored to consume the shared
``tsflab.models._components.topk_expert_router`` component. These tests prove the
refactor left every consumer's behavior unchanged:

- ``tests/fixtures/duet_pre_refactor.pt`` and
  ``tests/fixtures/dynamic_tmoe_pre_refactor.pt`` were captured from the
  unmodified models (``torch.manual_seed(0)`` construction, a fixed random
  input, eval mode) before the extraction. Re-running the same construction
  and input against the refactored models must reproduce identical
  ``state_dict`` keys/shapes/values, forward outputs, and input gradients.
- A pure-tensor test proves ``topk_dense_mix`` reproduces, term for term, the
  two independent inline formulas it replaced.

DUET's per-expert moving average was evaluated for extraction into
``series_decomposition`` and deliberately kept model-local instead (see
``TestDuetMovingAverageStaysLocal`` below for the reason).
"""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from tsflab.models._components.topk_expert_router import GatingMLP, topk_dense_mix
from tsflab.models.duet.model import Model as DuetModel
from tsflab.models.dynamic_tmoe.model import Model as DynamicTMoEModel

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestTopKDenseMixMatchesOriginalFormulas(unittest.TestCase):
    """``topk_dense_mix`` must reproduce both consumers' inline math exactly."""

    def test_matches_duet_original_sparse_scatter_and_renormalize(self) -> None:
        torch.manual_seed(0)
        weights = torch.softmax(torch.randn(4, 6), dim=-1)
        k, floor = 2, 1e-3

        top = weights.topk(k, dim=-1).indices
        sparse = torch.zeros_like(weights).scatter(-1, top, weights.gather(-1, top))
        expected = (sparse + floor * weights) / (sparse + floor * weights).sum(-1, keepdim=True)

        actual = topk_dense_mix(weights, k, floor)
        torch.testing.assert_close(actual, expected)

    def test_matches_dynamic_tmoe_original_hard_mask_and_floor(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(3, 5, 5)
        k, floor = 3, 1e-4

        soft = logits.softmax(-1)
        indices = logits.topk(k, dim=-1).indices
        hard = torch.zeros_like(soft).scatter(-1, indices, 1.0)
        concentrated = soft * hard
        expected = (concentrated + floor * soft) / (concentrated + floor * soft).sum(-1, keepdim=True)

        actual = topk_dense_mix(soft, k, floor)
        torch.testing.assert_close(actual, expected)

    def test_output_sums_to_one(self) -> None:
        weights = torch.softmax(torch.randn(2, 4), dim=-1)
        mixed = topk_dense_mix(weights, k=2, floor=1e-3)
        torch.testing.assert_close(mixed.sum(-1), torch.ones(2))


class TestGatingMLPMatchesDuetDistributionalRouter(unittest.TestCase):
    """``GatingMLP`` must reproduce DUET's former ``DistributionalRouter`` gate."""

    def test_noisy_gate_disabled_in_eval_matches_plain_softmax(self) -> None:
        torch.manual_seed(0)
        gate = GatingMLP(in_features=6, experts=4, hidden=8, noisy=True)
        gate.eval()
        features = torch.randn(3, 6)
        expected = torch.softmax(gate.network(features), -1)
        torch.testing.assert_close(gate(features), expected)

    def test_state_dict_attribute_names_preserved(self) -> None:
        gate = GatingMLP(in_features=6, experts=4, hidden=8, noisy=True)
        keys = set(gate.state_dict().keys())
        self.assertEqual(
            keys,
            {
                "network.0.weight", "network.0.bias",
                "network.2.weight", "network.2.bias",
                "noise_scale",
            },
        )


def _duet_reference() -> tuple[DuetModel, dict]:
    payload = torch.load(FIXTURES / "duet_pre_refactor.pt", weights_only=False)
    torch.manual_seed(0)
    model = DuetModel(
        seq_len=16, pred_len=8, enc_in=3, features="M", d_model=16, n_heads=4,
        e_layers=2, d_ff=16, dropout=0.1, fc_dropout=0.1,
        moving_avg=7, num_experts=4, k=2, hidden_size=8, noisy_gating=True,
    )
    model.eval()
    return model, payload


def _dynamic_tmoe_reference() -> tuple[DynamicTMoEModel, dict]:
    payload = torch.load(FIXTURES / "dynamic_tmoe_pre_refactor.pt", weights_only=False)
    torch.manual_seed(0)
    model = DynamicTMoEModel(
        seq_len=16, pred_len=8, enc_in=3, d_model=16, patch_len=4, stride=2,
        top_k=3, memory_slots=4, relation_period=6, routing_floor=1e-4, use_revin=True,
    )
    model.eval()
    return model, payload


class TestDuetUnchangedAfterExtraction(unittest.TestCase):
    """DUET's construction, forward, and gradients must match the pre-refactor fixture."""

    def test_state_dict_keys_shapes_and_values_are_unchanged(self) -> None:
        model, payload = _duet_reference()
        state = model.state_dict()
        old_state = payload["state_dict"]
        self.assertEqual(set(state.keys()), set(old_state.keys()))
        for key, tensor in state.items():
            self.assertEqual(tuple(tensor.shape), payload["shapes"][key])
            torch.testing.assert_close(tensor, old_state[key], atol=1e-6, rtol=0)

    def test_forward_output_is_unchanged(self) -> None:
        model, payload = _duet_reference()
        output = model(payload["input"])
        torch.testing.assert_close(output, payload["output"], atol=1e-6, rtol=0)

    def test_gradients_are_unchanged(self) -> None:
        model, payload = _duet_reference()
        x = payload["input"].clone().detach().requires_grad_(True)
        output = model(x)
        (grad_x,) = torch.autograd.grad(output.sum(), x, retain_graph=True)
        torch.testing.assert_close(grad_x, payload["grad_input"], atol=1e-5, rtol=1e-5)
        output.sum().backward()
        for name, param in model.named_parameters():
            expected_grad = payload["grad_params"][name]
            if expected_grad is None:
                self.assertIsNone(param.grad)
            else:
                torch.testing.assert_close(param.grad, expected_grad, atol=1e-5, rtol=1e-5)


class TestDynamicTMoEUnchangedAfterExtraction(unittest.TestCase):
    """DynamicTMoE's construction, forward, and gradients must match the fixture."""

    def test_state_dict_keys_shapes_and_values_are_unchanged(self) -> None:
        model, payload = _dynamic_tmoe_reference()
        state = model.state_dict()
        old_state = payload["state_dict"]
        self.assertEqual(set(state.keys()), set(old_state.keys()))
        for key, tensor in state.items():
            self.assertEqual(tuple(tensor.shape), payload["shapes"][key])
            torch.testing.assert_close(tensor, old_state[key], atol=1e-6, rtol=0)

    def test_forward_output_is_unchanged(self) -> None:
        model, payload = _dynamic_tmoe_reference()
        output = model(payload["input"])
        torch.testing.assert_close(output, payload["output"], atol=1e-6, rtol=0)

    def test_gradients_are_unchanged(self) -> None:
        model, payload = _dynamic_tmoe_reference()
        x = payload["input"].clone().detach().requires_grad_(True)
        output = model(x)
        (grad_x,) = torch.autograd.grad(output.sum(), x, retain_graph=True)
        torch.testing.assert_close(grad_x, payload["grad_input"], atol=1e-5, rtol=1e-5)
        output.sum().backward()
        for name, param in model.named_parameters():
            expected_grad = payload["grad_params"][name]
            if expected_grad is None:
                self.assertIsNone(param.grad)
            else:
                torch.testing.assert_close(param.grad, expected_grad, atol=1e-5, rtol=1e-5)


class TestDuetMovingAverageStaysLocal(unittest.TestCase):
    """DUET's per-expert moving average was deliberately NOT extracted.

    ``series_decomposition.EdgePaddedMovingAverage`` validates ``kernel_size``
    as strictly odd, and BiST (``models/bist/model.py``) relies on that
    validation to reject misconfigured even kernel sizes. DUET's local
    ``moving_average`` helper additionally supports even kernels via
    asymmetric padding, so extending the shared component's contract to match
    it would silently change BiST's error contract. These tests pin both
    halves of that reasoning: the shared component still rejects even
    kernels, and DUET's local helper still accepts them.
    """

    def test_shared_component_still_rejects_even_kernel(self) -> None:
        from tsflab.models._components.series_decomposition import EdgePaddedMovingAverage

        with self.assertRaisesRegex(ValueError, "positive odd"):
            EdgePaddedMovingAverage(kernel_size=4)

    def test_duet_local_helper_still_accepts_even_kernel(self) -> None:
        from tsflab.models.duet.model import moving_average

        torch.manual_seed(0)
        x = torch.randn(2, 20, 3)
        kernel = 6
        left, right = (kernel - 1) // 2, kernel // 2
        padded = torch.cat(
            (x[:, :1].expand(-1, left, -1), x, x[:, -1:].expand(-1, right, -1)), 1
        )
        expected = torch.nn.functional.avg_pool1d(
            padded.transpose(1, 2), kernel, stride=1
        ).transpose(1, 2)
        torch.testing.assert_close(moving_average(x, kernel), expected)


if __name__ == "__main__":
    unittest.main()
