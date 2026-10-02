"""Paper-equation checks for the local TimeKAN."""

from __future__ import annotations

import unittest

import torch

from tsflab.models.timekan.model import ChebyshevKAN, Model, frequency_upsample


class LocalTimeKANTests(unittest.TestCase):
    def test_frequency_upsampling_preserves_constant_signal(self) -> None:
        values = torch.full((2, 4, 3), 2.5)
        expected = torch.full((2, 10, 3), 2.5)
        torch.testing.assert_close(frequency_upsample(values, 10), expected)

    def test_chebyshev_kan_uses_polynomial_recurrence(self) -> None:
        layer = ChebyshevKAN(width=1, order=2)
        with torch.no_grad():
            layer.coefficients.zero_()
            layer.coefficients[0, 0, 2] = 1.0
        values = torch.tensor([[[0.25], [-0.5]]])
        bounded = values.tanh()
        expected = 2 * bounded.square() - 1
        torch.testing.assert_close(layer(values), expected)

    def test_orders_increase_toward_high_frequency_band(self) -> None:
        model = Model(12, 4, 0, "M", 2, d_model=4, e_layers=1,
                      down_sampling_layers=2, begin_order=1)
        orders = [learner.kan.order for learner in model.blocks[0].learners]
        self.assertEqual(orders, [3, 2, 1])
        self.assertEqual(model(torch.randn(2, 12, 2)).shape, (2, 4, 2))


if __name__ == "__main__":
    unittest.main()
