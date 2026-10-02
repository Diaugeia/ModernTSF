"""Paper-structure and runtime tests for DPWMixer."""
from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.dpwmixer.model import Model as DPWMixer


def marks(batch=2, steps=12, offset=0):
    rows = [[2026, 8, 1 + i // 24, 4, (i + offset) % 24, 0] for i in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def factory():
    return DPWMixer(12, 3, 4, d_model=8, dropout=0, patch_len=4, stride=2, down_sampling_layers=1)


class PaperStructureTests(unittest.TestCase):
    def test_dpwmixer_builds_a_lossless_wavelet_pyramid_with_one_mixer_per_scale(self):
        model = factory().eval()
        self.assertEqual(len(model.wavelets), 1)
        self.assertEqual(len(model.mixers), 2)
        approx, detail = model.wavelets[0].decompose(torch.randn(2, 4, 12))
        self.assertEqual(approx.shape, (2, 4, 6))
        self.assertEqual(detail.shape, (2, 4, 6))
        model(torch.randn(2, 12, 4))
        torch.testing.assert_close(model.last_fusion_weights.sum(0), torch.ones(4))

    def test_marks_are_accepted_but_do_not_change_output(self):
        # DPWMixer is an encoder-only architecture that ignores marks entirely.
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
