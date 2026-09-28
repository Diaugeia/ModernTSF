"""Paper-structure and runtime tests for TimeExpert."""
from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models.timeexpert.model import Model as TimeExpert


def marks(batch=2, steps=12, offset=0):
    rows = [[2026, 8, 1 + i // 24, 4, (i + offset) % 24, 0] for i in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def factory():
    return TimeExpert(
        12, 3, 4, d_model=8, n_heads=2, e_layers=1, patch_len=4, stride=2, dropout=0, topk=2, shared=True
    )


class PaperStructureTests(unittest.TestCase):
    def test_timeexpert_routes_queries_to_topk_local_experts(self):
        model = factory().eval()
        attention = model.layers[0].attention
        self.assertEqual(attention.topk, 2)
        self.assertTrue(attention.shared)
        model(torch.randn(2, 12, 4))
        weight = attention.last_route_weight
        self.assertIsNotNone(weight)
        torch.testing.assert_close(weight.sum(-1), torch.ones(weight.shape[:-1]))

    def test_marks_do_not_change_output(self):
        # TimeExpert's channel-independent patch encoder never consumes marks.
        model = factory().eval()
        x = torch.randn(2, 12, 4)
        first = model(x, marks(offset=0))
        second = model(x, marks(offset=7))
        torch.testing.assert_close(first, second)


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_active_parameters_and_round_trip(self):
        torch.manual_seed(260827)
        x = torch.randn(2, 12, 4)
        model = factory().cpu().eval()
        value = x.clone().requires_grad_(True)
        output = model(value, marks(), x_mark_dec=marks(2, 3, 5))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(value.grad)
        self.assertGreater(value.grad.abs().max().item(), 0)
        for parameter_name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, parameter_name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
        clone = factory().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        torch.testing.assert_close(
            clone(x, marks(), x_mark_dec=marks(2, 3, 5)),
            model(x, marks(), x_mark_dec=marks(2, 3, 5)),
        )
        self.assertEqual(model(x[:1], marks(1), x_mark_dec=marks(1, 3)).shape, (1, 3, 4))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 11, 4))

    def test_topk_zero_falls_back_to_full_attention(self):
        torch.manual_seed(0)
        model = TimeExpert(
            12, 3, 4, d_model=8, n_heads=2, e_layers=1, patch_len=4, stride=2, dropout=0, topk=0, shared=False
        ).eval()
        output = model(torch.randn(2, 12, 4))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
