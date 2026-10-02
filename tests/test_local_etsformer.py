"""Equation and structure checks for the local ETSformer."""

from __future__ import annotations

import math
import unittest

import torch

from tsflab.models.etsformer.model import ExponentialSmoothing, FrequencyAttention, Model


class LocalETSformerTests(unittest.TestCase):
    def test_exponential_smoothing_obeys_recurrence(self) -> None:
        layer = ExponentialSmoothing(1)
        with torch.no_grad():
            layer.alpha_logit.zero_()  # alpha = 1/2
            layer.initial.fill_(2.0)
        values = torch.tensor([[[4.0], [8.0], [0.0]]])
        actual = layer(values)
        expected = torch.tensor([[[3.0], [5.5], [2.75]]])
        torch.testing.assert_close(actual, expected)

    def test_frequency_attention_extrapolates_selected_basis(self) -> None:
        positions = torch.arange(8, dtype=torch.float32)
        values = torch.cos(2 * math.pi * positions / 4).view(1, 8, 1)
        history, future = FrequencyAttention(top_k=1)(values, 4)
        torch.testing.assert_close(history, values, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(future, values[:, :4], atol=1e-5, rtol=1e-5)

    def test_model_exposes_level_growth_seasonality_stacks(self) -> None:
        model = Model(12, 4, 2, d_model=8, e_layers=2, d_ff=16,
                      top_k=2, dropout=0.0)
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(all(hasattr(layer, "frequency") for layer in model.layers))
        self.assertTrue(all(hasattr(layer, "smoothing") for layer in model.layers))
        output = model(torch.randn(2, 12, 2))
        self.assertEqual(output.shape, (2, 4, 2))


if __name__ == "__main__":
    unittest.main()
