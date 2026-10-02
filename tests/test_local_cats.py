"""Structural checks for the local cross-attention-only CATS."""

from __future__ import annotations

import unittest

import torch

from tsflab.models.cats.model import Model


class LocalCATSTests(unittest.TestCase):
    def test_model_contains_cross_attention_but_no_self_attention(self) -> None:
        model = Model(12, 8, 2, patch_len=4, stride=4, d_model=8,
                      n_heads=2, d_ff=16, n_layers=2, dropout=0.0)
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(all(hasattr(layer, "attention") for layer in model.layers))
        self.assertFalse(any(hasattr(layer, "self_attention") for layer in model.layers))

    def test_future_queries_and_projection_are_shared_across_horizons(self) -> None:
        model = Model(12, 8, 3, patch_len=4, stride=4, d_model=8,
                      n_heads=2, d_ff=16, n_layers=1, dropout=0.0)
        self.assertEqual(model.future_queries.shape, (1, 2, 4))
        self.assertEqual(model.projection.out_features, 4)
        self.assertEqual(model(torch.randn(2, 12, 3)).shape, (2, 8, 3))

    def test_query_adaptive_mask_schedule_increases_by_layer(self) -> None:
        model = Model(12, 8, 2, patch_len=4, stride=4, d_model=8,
                      n_heads=2, d_ff=16, n_layers=3, QAM_start=0.1, QAM_end=0.5)
        probabilities = [layer.masking_probability for layer in model.layers]
        self.assertEqual(probabilities, [0.10000000149011612, 0.30000001192092896, 0.5])


if __name__ == "__main__":
    unittest.main()
