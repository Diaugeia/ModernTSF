"""Paper-structure and runtime tests for Dualformer."""

from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models.dualformer.model import AutoCorrelationAttention, Model as Dualformer


def marks(batch: int = 2, length: int = 12) -> torch.Tensor:
    values = torch.zeros(batch, length, 6)
    values[..., 0] = 2026
    values[..., 1] = 3
    values[..., 2] = torch.arange(1, length + 1) % 28 + 1
    values[..., 3] = torch.arange(length) % 7
    values[..., 4] = torch.arange(length) % 24
    return values


def make_model(alpha: float = 0.5, e_layers: int = 2) -> Dualformer:
    return Dualformer(
        seq_len=12,
        pred_len=3,
        enc_in=4,
        c_out=4,
        d_model=8,
        n_heads=2,
        e_layers=e_layers,
        d_ff=16,
        dropout=0.0,
        factor=1.0,
        alpha=alpha,
        num_harmonics=2,
        dc_bins=1,
    )


class PaperStructureTests(unittest.TestCase):
    def test_hierarchical_frequency_sampling_narrows_toward_low_frequency_with_depth(self):
        """Paper: shallow layers keep high-frequency detail, deep layers get
        the low-frequency (near-DC) band. With alpha <= 1/e_layers the bands
        tile the spectrum and layer 0's band must sit at a strictly higher
        frequency than the last layer's band."""
        model = make_model(alpha=0.5, e_layers=2).eval()
        model(torch.randn(2, 12, 4))
        self.assertIsNotNone(model.last_bands)
        self.assertEqual(len(model.last_bands), 2)
        first_start, _ = model.last_bands[0]
        last_start, last_end = model.last_bands[-1]
        self.assertGreater(first_start, last_start)
        self.assertEqual(last_start, 0)
        self.assertNotEqual(model.last_bands[0], model.last_bands[1])

    def test_time_and_frequency_branches_use_distinct_attention_mechanisms(self):
        """Paper: a dual-branch architecture models the time domain and the
        frequency domain concurrently with different mechanisms per branch."""
        model = make_model().eval()
        time_attention = model.time_layers[0].attention
        freq_attention = model.freq_layers[0].attention
        self.assertIsInstance(time_attention, AttentionLayer)
        self.assertIsInstance(time_attention.inner_attention, FullAttention)
        self.assertIsInstance(freq_attention, AttentionLayer)
        self.assertIsInstance(freq_attention.inner_attention, AutoCorrelationAttention)

    def test_periodicity_aware_gate_fuses_branches_with_a_bounded_weight(self):
        """Paper: a periodicity-aware weighting mechanism fuses the dual
        branches based on the harmonic energy ratio, producing a weight used
        as `frequency * w + time * (1 - w)`."""
        model = make_model().eval()
        model(torch.randn(2, 12, 4))
        weight = model.last_gate_weight
        self.assertEqual(weight.shape, (2, 1, 8))
        self.assertTrue(torch.all(weight >= 0.0))
        self.assertTrue(torch.all(weight <= 1.0))


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_active_parameters_and_round_trip(self):
        torch.manual_seed(260928)
        model = make_model().cpu().eval()
        x = torch.randn(2, 12, 4)
        value = x.clone().requires_grad_(True)
        output = model(value, marks(2, 12))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(output).all())

        output.square().mean().backward()
        self.assertIsNotNone(value.grad)
        self.assertGreater(value.grad.abs().max().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

        clone = make_model().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(clone(x, marks(2, 12)), model(x, marks(2, 12)))

    def test_default_marks_and_batch_bounds(self):
        model = make_model().cpu().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 12, 4))
        self.assertEqual(output.shape, (1, 3, 4))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError):
            model(torch.randn(1, 11, 4))

    def test_enc_in_and_c_out_must_match(self):
        with self.assertRaises(ValueError):
            Dualformer(seq_len=12, pred_len=3, enc_in=4, c_out=5, d_model=8, n_heads=2, e_layers=1, d_ff=16)


if __name__ == "__main__":
    unittest.main()
