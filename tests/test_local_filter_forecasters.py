"""Paper-equation checks for local FilterNet variants."""

from __future__ import annotations

import unittest

import torch

from moderntsf.models.paifilter.model import PlainShapingFilter
from moderntsf.models.texfilter.model import ContextualShapingFilter, _complex_linear


class LocalFilterForecasterTests(unittest.TestCase):
    def test_plain_filter_is_frequency_multiplication(self) -> None:
        values = torch.randn(2, 12, 3)
        layer = PlainShapingFilter(12)
        with torch.no_grad():
            layer.weight_real.fill_(0.5)
            layer.weight_imag.zero_()
        torch.testing.assert_close(layer(values), values * 0.5, atol=1e-6, rtol=1e-6)

    def test_contextual_embedding_uses_complex_linear_algebra(self) -> None:
        values = torch.tensor([[[1 + 2j, 3 + 4j]]])
        real = torch.tensor([[2.0], [1.0]])
        imag = torch.tensor([[1.0], [-1.0]])
        expected = values @ torch.complex(real, imag)
        torch.testing.assert_close(_complex_linear(values, real, imag), expected)

    def test_contextual_filter_has_input_dependent_kernel(self) -> None:
        layer = ContextualShapingFilter(12, 8)
        first = layer(torch.randn(2, 12, 3))
        second = layer(torch.randn(2, 12, 3))
        self.assertEqual(first.shape, (2, 8, 3))
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
