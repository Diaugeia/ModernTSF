"""Regression coverage for model state that used to escape ``state_dict``."""

from __future__ import annotations

import io
import unittest

import numpy as np
import torch

from tsflab.benchmark.model_contracts import audit_model_contracts
from tsflab.models.fedformer.model import FrequencyEnhancedAttention, FrequencyEnhancedBlock
from tsflab.models.koopa.model import Model as Koopa
from tsflab.models.pcdcnet.model import Model as PCDCNet


class StateDictRoundTripRegressionTests(unittest.TestCase):
    def test_strict_round_trip_for_affected_catalog_models(self) -> None:
        failures = audit_model_contracts(
            names=["DCRNN", "GTS", "FEDformer", "Koopa", "PCDCNet"],
            strict=True,
        )
        self.assertEqual(failures, [])

    def test_fourier_mode_indices_are_persistent_buffers(self) -> None:
        source_block = FrequencyEnhancedBlock(8, 2, 96, 8, "random")
        source_cross = FrequencyEnhancedAttention(8, 2, 60, 96, 8, "random")

        block_state = source_block.state_dict()
        cross_state = source_cross.state_dict()
        self.assertIn("mode_indices", block_state)
        self.assertIn("query_modes", cross_state)
        self.assertIn("key_modes", cross_state)

        restored_block = FrequencyEnhancedBlock(8, 2, 96, 8, "random")
        restored_cross = FrequencyEnhancedAttention(8, 2, 60, 96, 8, "random")
        restored_block.load_state_dict(block_state, strict=True)
        restored_cross.load_state_dict(cross_state, strict=True)
        torch.testing.assert_close(restored_block.mode_indices, source_block.mode_indices)
        torch.testing.assert_close(restored_cross.query_modes, source_cross.query_modes)
        torch.testing.assert_close(restored_cross.key_modes, source_cross.key_modes)

    def test_koopa_state_survives_serialization(self) -> None:
        torch.manual_seed(3)
        source = Koopa(
            seq_len=24,
            pred_len=6,
            enc_in=2,
            dynamic_dim=8,
            hidden_dim=8,
            num_blocks=1,
        ).eval()
        comparison_batch = torch.randn(2, 24, 2)
        expected = source(comparison_batch)

        payload = io.BytesIO()
        torch.save(source.state_dict(), payload)
        payload.seek(0)
        restored = Koopa(
            seq_len=24,
            pred_len=6,
            enc_in=2,
            dynamic_dim=8,
            hidden_dim=8,
            num_blocks=1,
        ).eval()
        restored.load_state_dict(
            torch.load(payload, weights_only=True), strict=True
        )

        torch.testing.assert_close(restored(comparison_batch), expected)

    def test_pcdc_eval_is_deterministic(self) -> None:
        model = PCDCNet(
            seq_len=6,
            pred_len=3,
            enc_in=3,
            adj_mx=np.eye(3, dtype=np.float32),
            cov_dim=0,
            d_model=4,
            dropout=0.0,
        ).eval()
        values = torch.randn(2, 6, 3)
        torch.manual_seed(4)
        first = model(values)
        torch.manual_seed(5)
        second = model(values)
        torch.testing.assert_close(second, first, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
