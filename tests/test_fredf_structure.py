"""Paper-structure and runtime tests for FreDF (Frequency Dynamic Fusion, ACM MM 2024)."""
from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.fredf.model import FrequencyDynamicFusionBlock, Model as FreDF


def marks(batch=2, steps=12, offset=0):
    rows = [[2026, 8, 1 + i // 24, 4, (i + offset) % 24, 0] for i in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def factory(layers=1):
    return FreDF(12, 4, 3, d_model=8, e_layers=layers, dropout=0.0)


class PaperStructureTests(unittest.TestCase):
    def test_block_has_one_transfer_matrix_and_weight_per_frequency_bin(self):
        block = FrequencyDynamicFusionBlock(16, 8)
        self.assertEqual(block.bins, 9)  # K = (T + S) / 2 + 1
        self.assertEqual(tuple(block.transfer.shape), (9, 8, 8, 2))
        self.assertEqual(tuple(block.frequency_weight.shape), (9,))
        self.assertAlmostEqual(block.frequency_weight.sum().item(), 1.0, places=5)

    def test_fused_block_equals_algorithm_one_masked_per_frequency_form(self):
        torch.manual_seed(0)
        for length in (16, 15):
            block = FrequencyDynamicFusionBlock(length, 6)
            values = torch.randn(2, length, 6)
            torch.testing.assert_close(
                block(values), block.decoupled_reference(values), rtol=1e-4, atol=1e-5
            )

    def test_single_frequency_input_is_scaled_by_its_own_transfer_matrix_and_weight(self):
        torch.manual_seed(1)
        block = FrequencyDynamicFusionBlock(16, 4)
        k = 3
        spectrum = torch.zeros(1, 9, 4, dtype=torch.complex64)
        spectrum[0, k] = torch.randn(4, dtype=torch.complex64)
        values = torch.fft.irfft(spectrum, n=16, dim=1, norm="ortho")
        expected = torch.zeros_like(spectrum)
        expected[0, k] = (
            spectrum[0, k] @ torch.view_as_complex(block.transfer)[k].T * block.frequency_weight[k]
        )
        torch.testing.assert_close(
            torch.fft.rfft(block(values), dim=1, norm="ortho"), expected, rtol=1e-4, atol=1e-5
        )

    def test_zero_padded_future_and_marks_reach_the_embedding(self):
        model = factory().eval()
        x = torch.randn(2, 12, 3)
        base = model(x, marks(), x_mark_dec=marks(2, 4, 5))
        other = model(x, marks(), x_mark_dec=marks(2, 4, 9))
        self.assertFalse(torch.allclose(base, other))
        torch.testing.assert_close(model(x), model(x))
        self.assertEqual(len(factory(2).blocks), 2)

    def test_forecast_is_equivariant_to_input_offset_and_scale(self):
        model = factory().eval()
        x = torch.randn(2, 12, 3)
        torch.testing.assert_close(
            model(x * 4 + 2), model(x) * 4 + 2, rtol=1e-3, atol=1e-3
        )


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_active_parameters_and_round_trip(self):
        torch.manual_seed(7)
        x = torch.randn(2, 12, 3)
        model = factory().cpu().eval()
        value = x.clone().requires_grad_(True)
        output = model(value, marks(), x_mark_dec=marks(2, 4, 5))
        self.assertEqual(output.shape, (2, 4, 3))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertGreater(value.grad.abs().max().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        clone = factory().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        torch.testing.assert_close(
            clone(x, marks(), x_mark_dec=marks(2, 4, 5)),
            model(x, marks(), x_mark_dec=marks(2, 4, 5)),
        )
        self.assertEqual(model(x[:1]).shape, (1, 4, 3))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 11, 3))


if __name__ == "__main__":
    unittest.main()
