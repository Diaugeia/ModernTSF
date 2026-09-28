"""Paper-equation checks for local MoFo."""

from __future__ import annotations

import unittest

import torch

from moderntsf.models.mofo.model import Model, RegulatedRelaxation, period_structured_patches


class LocalMoFoTests(unittest.TestCase):
    def test_regulated_relaxation_has_paper_boundary_values(self) -> None:
        regulator = RegulatedRelaxation()
        distances = torch.tensor([0.0, 100.0])
        values = regulator(distances)
        torch.testing.assert_close(values[0], torch.tensor(1.0))
        self.assertLess(values[1].item(), 1e-6)

    def test_period_structured_rows_are_phase_aligned(self) -> None:
        values = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        patches = period_structured_patches(values, period=4)
        expected = torch.tensor([[[[0.0, 4.0, 8.0],
                                   [1.0, 5.0, 9.0],
                                   [2.0, 6.0, 10.0],
                                   [3.0, 7.0, 11.0]]]])
        torch.testing.assert_close(patches, expected)

    def test_model_uses_period_modulated_future_queries(self) -> None:
        model = Model(12, 5, 2, d_model=8, periodic=4, head=2,
                      d_layers=2, bias=1, cias=1)
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(all(hasattr(layer.attention, "modulator") for layer in model.layers))
        self.assertEqual(model(torch.randn(2, 12, 2)).shape, (2, 5, 2))


if __name__ == "__main__":
    unittest.main()
