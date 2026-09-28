"""Paper-structure and runtime tests for WDformer."""
from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models.wdformer.model import Model as WDformer


def marks(batch=2, steps=12, offset=0):
    rows = [[2026, 8, 1 + i // 24, 4, (i + offset) % 24, 0] for i in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def factory():
    return WDformer(12, 3, 4, d_model=8, n_heads=2, e_layers=1, d_ff=16, dropout=0, wave_size=1)


class PaperStructureTests(unittest.TestCase):
    def test_wdformer_embeds_wavelet_subbands_and_reconstructs_via_inverse_wavelet(self):
        model = factory().eval()
        self.assertEqual(len(model.subband_embedding), 2)
        self.assertEqual(sum(model.output_split_sizes), model.projector.out_features)
        model(torch.randn(2, 12, 4))
        lambda_value = model.layers[0].attention.attention._lambda()
        self.assertTrue(torch.isfinite(lambda_value).all())

    def test_marks_extend_the_variate_token_axis(self):
        model = factory().eval()
        x = torch.randn(2, 12, 4)
        without_marks = model(x)
        with_marks = model(x, marks())
        self.assertEqual(without_marks.shape, with_marks.shape)
        self.assertGreater((without_marks - with_marks).abs().max().item(), 0)


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
