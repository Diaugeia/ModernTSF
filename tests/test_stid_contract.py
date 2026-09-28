"""Contract for the STID model input contract."""

from __future__ import annotations

import unittest

import torch

from moderntsf.models.stid.model import Model as STID


class STIDContractTests(unittest.TestCase):
    def test_model_uses_values_and_calendar_input_channels(self):
        model = STID(12, 6, 4, input_dim=3, embed_dim=8, num_layers=1)
        self.assertEqual(model.input_projection.in_features, 36)
        self.assertEqual(model.encoder[0].dropout.p, 0.15)
        values = torch.randn(2, 12, 4, requires_grad=True)
        marks = torch.zeros(2, 12, 6)
        marks[..., 3] = 2
        marks[..., 4] = 5
        output = model(values, marks)
        self.assertEqual(output.shape, (2, 6, 4))
        output.sum().backward()
        self.assertIsNotNone(values.grad)

if __name__ == "__main__":
    unittest.main()
