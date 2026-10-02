"""Paper-structure and runtime-contract tests for SDMixer."""
from __future__ import annotations

import copy
import math
import unittest

import torch

from tsflab.models.sdmixer.model import Model as SDMixer


SEQ_LEN = 24
PRED_LEN = 6
ENC_IN = 4
CHANNEL_SPARSE_RATIO = 0.5
ATTN_SPARSE_RATIO = 0.5


def factory():
    return SDMixer(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        enc_in=ENC_IN,
        c_out=ENC_IN,
        spectral_top_k=3,
        channel_sparse_ratio=CHANNEL_SPARSE_RATIO,
        attn_sparse_ratio=ATTN_SPARSE_RATIO,
        d_ff=16,
        dropout=0.0,
    )


class PaperStructureTests(unittest.TestCase):
    """Check the paper's defining sparse dual-mixer mechanisms (Eq. 3-13)."""

    def test_temporal_flow_channel_gate_keeps_exact_top_k_per_step(self):
        # Eq. (8): the sparse gate G_T must retain exactly k = ceil(alpha*C)
        # channels at every time step and mask the rest to zero.
        model = factory().eval()
        model(torch.randn(2, SEQ_LEN, ENC_IN))
        gate = model.temporal_flow.last_gate
        expected_k = math.ceil(CHANNEL_SPARSE_RATIO * ENC_IN)
        self.assertEqual(model.temporal_flow.k, expected_k)
        self.assertEqual(tuple(gate.shape), (2, SEQ_LEN, ENC_IN))
        self.assertTrue(torch.logical_or(gate == 0, gate == 1).all())
        torch.testing.assert_close(gate.sum(dim=-1), torch.full((2, SEQ_LEN), float(expected_k)))

    def test_cross_mixer_sparsifies_attention_to_declared_ratio(self):
        # Eq. (12): alpha = TopK(Softmax(QK^T / sqrt(C))) keeps only the
        # declared fraction of key positions per query row.
        model = factory().eval()
        model(torch.randn(2, SEQ_LEN, ENC_IN))
        alpha = model.cross_mixer.last_attention
        expected_keep = math.ceil(ATTN_SPARSE_RATIO * SEQ_LEN)
        self.assertEqual(tuple(alpha.shape), (2, SEQ_LEN, SEQ_LEN))
        nonzero_per_row = (alpha > 0).sum(dim=-1)
        self.assertTrue((nonzero_per_row <= expected_keep).all())
        self.assertTrue((nonzero_per_row.max() == expected_keep).item())
        # Non-negative softmax-derived weights; masked entries are exactly zero.
        self.assertTrue((alpha >= 0).all())

    def test_spectral_decomposition_season_and_trend_reconstruct_input(self):
        # Eq. (5): X^trend = X - X^season, i.e. the two branches are an
        # exact additive decomposition of the (RevIN-normalized) input.
        model = factory().eval()
        x = torch.randn(2, SEQ_LEN, ENC_IN)
        normalized = model.revin(x, "norm")
        season, trend = model.decomposition(normalized)
        torch.testing.assert_close(season + trend, normalized, atol=1e-5, rtol=1e-5)
        # The decomposition is non-trivial: season is not identically zero.
        self.assertGreater(season.abs().sum().item(), 0.0)


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_shapes_and_gradients(self):
        torch.manual_seed(260928)
        model = factory().cpu().eval()
        x = torch.randn(3, SEQ_LEN, ENC_IN, requires_grad=True)
        output = model(x, torch.randn(3, SEQ_LEN, 4), x_dec=None, x_mark_dec=None)
        self.assertEqual(output.shape, (3, PRED_LEN, ENC_IN))
        self.assertTrue(torch.isfinite(output).all())

        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().max().item(), 0.0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_state_dict_round_trip_is_deterministic(self):
        torch.manual_seed(7)
        model = factory().cpu().eval()
        clone = factory().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        x = torch.randn(2, SEQ_LEN, ENC_IN)
        torch.testing.assert_close(model(x), clone(x))

    def test_batch_size_one_and_invalid_seq_len(self):
        model = factory().cpu().eval()
        single = model(torch.randn(1, SEQ_LEN, ENC_IN))
        self.assertEqual(single.shape, (1, PRED_LEN, ENC_IN))
        with self.assertRaises(ValueError):
            model(torch.randn(1, SEQ_LEN - 1, ENC_IN))

    def test_enc_in_c_out_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            SDMixer(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=ENC_IN, c_out=ENC_IN + 1)


if __name__ == "__main__":
    unittest.main()
