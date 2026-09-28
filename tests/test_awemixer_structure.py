"""Paper-structure and runtime tests for AWEMixer."""
from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models.awemixer.model import Model as AWEMixer


def marks(batch=2, steps=12, offset=0):
    rows = [[2026, 8, 1 + i // 24, 4, (i + offset) % 24, 0] for i in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def factory():
    return AWEMixer(
        12, 3, 4, d_model=8, dropout=0, num_scales=2, wavelet_level=2, wavelet="db2", num_fusion_layers=1
    )


class PaperStructureTests(unittest.TestCase):
    def test_awemixer_routes_subbands_with_a_softmax_frequency_router(self):
        model = factory().eval()
        self.assertEqual(model.num_bands, 3)
        model(torch.randn(2, 12, 4))
        self.assertEqual(model.last_router_weights.shape[-2:], (3, 1))
        torch.testing.assert_close(model.last_router_weights.sum(-2).squeeze(-1), torch.ones(2 * 4))
        self.assertEqual(len(model.last_gates), 1)

    def test_marks_are_accepted_but_do_not_change_output(self):
        # AWEMixer is an encoder-only architecture that ignores marks entirely.
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


if __name__ == "__main__":
    unittest.main()
