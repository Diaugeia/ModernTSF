"""Paper-equation checks for locally assembled attention forecasters."""

from __future__ import annotations

import math
import unittest

import torch

from moderntsf.models._components.self_attention_family import FullAttention, ProbAttention
from moderntsf.models.informer.model import Model as Informer
from moderntsf.models.transformer.model import Model as Transformer


class LocalAttentionForecasterTests(unittest.TestCase):
    def test_scaled_dot_product_attention_matches_equation(self) -> None:
        query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        key = query.clone()
        value = torch.tensor([[[[2.0, 3.0]], [[5.0, 7.0]]]])
        layer = FullAttention(mask_flag=False, attention_dropout=0.0)
        actual, _ = layer(query, key, value, None)
        scores = torch.einsum("blhe,bshe->bhls", query, key) / math.sqrt(2.0)
        expected = torch.einsum("bhls,bshd->blhd", scores.softmax(-1), value)
        torch.testing.assert_close(actual, expected)

    def test_transformer_uses_full_attention_in_all_three_roles(self) -> None:
        model = Transformer(8, 3, 2, "M", 2, d_model=8, n_heads=2,
                            e_layers=1, d_layers=1, d_ff=16, dropout=0.0)
        encoder = model.encoder.attn_layers[0].attention.inner_attention
        decoder = model.decoder.layers[0]
        self.assertIsInstance(encoder, FullAttention)
        self.assertIsInstance(decoder.self_attention.inner_attention, FullAttention)
        self.assertTrue(decoder.self_attention.inner_attention.mask_flag)
        self.assertIsInstance(decoder.cross_attention.inner_attention, FullAttention)
        self.assertFalse(decoder.cross_attention.inner_attention.mask_flag)

    def test_informer_uses_probsparse_and_distilling(self) -> None:
        model = Informer(16, 4, 2, "M", 2, d_model=8, n_heads=2,
                         e_layers=3, d_layers=1, d_ff=16, dropout=0.0,
                         factor=2, distil=True)
        self.assertEqual(len(model.encoder.conv_layers), 2)
        self.assertTrue(all(
            isinstance(layer.attention.inner_attention, ProbAttention)
            for layer in model.encoder.attn_layers
        ))
        decoder = model.decoder.layers[0]
        self.assertIsInstance(decoder.self_attention.inner_attention, ProbAttention)
        self.assertTrue(decoder.self_attention.inner_attention.mask_flag)
        self.assertIsInstance(decoder.cross_attention.inner_attention, FullAttention)

    def test_one_shot_decoder_returns_only_forecast_horizon(self) -> None:
        for cls in (Transformer, Informer):
            with self.subTest(model=cls.__name__):
                model = cls(8, 3, 2, "M", 2, d_model=8, n_heads=2,
                            e_layers=1, d_layers=1, d_ff=16, dropout=0.0)
                x = torch.randn(2, 8, 2)
                dec = torch.zeros(2, 5, 2)
                output = model(x, None, dec, None)
                self.assertEqual(output.shape, (2, 3, 2))


if __name__ == "__main__":
    unittest.main()
