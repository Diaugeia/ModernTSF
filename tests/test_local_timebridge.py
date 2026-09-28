"""Equation and structure checks for local TimeBridge."""

from __future__ import annotations

import unittest

import torch

from moderntsf.models.timebridge.model import IntegratedAttention, Model


class _Capture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.arguments = None

    def forward(self, query, key, value):
        self.arguments = (query, key, value)
        return value


class LocalTimeBridgeTests(unittest.TestCase):
    def test_integrated_attention_detrends_queries_but_not_values(self) -> None:
        layer = IntegratedAttention(4, 1, 8, stable_len=3,
                                    attention_dropout=0.0, dropout=0.0,
                                    activation="gelu")
        capture = _Capture()
        layer.block = capture
        patches = torch.arange(20, dtype=torch.float32).reshape(1, 1, 5, 4)
        layer(patches)
        query, key, value = capture.arguments
        torch.testing.assert_close(query, key)
        torch.testing.assert_close(value, patches.reshape(1, 5, 4))
        self.assertFalse(torch.equal(query, value))

    def test_model_orders_integrated_downsample_cointegrated_stages(self) -> None:
        model = Model(24, 6, 3, period=4, ia_layers=2, pd_layers=1,
                      ca_layers=2, stable_len=2, d_model=8, n_heads=2,
                      d_ff=16, attn_dropout=0.0, dropout=0.0)
        self.assertEqual(len(model.integrated), 2)
        self.assertEqual(len(model.downsample), 1)
        self.assertEqual(len(model.cointegrated), 2)
        self.assertEqual(model.long_count, 3)
        self.assertEqual(model(torch.randn(2, 24, 3)).shape, (2, 6, 3))


if __name__ == "__main__":
    unittest.main()
