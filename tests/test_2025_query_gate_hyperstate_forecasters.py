"""Paper-structure tests for four 2025 additions.

Covers TQNet (temporal query attention), Gateformer (gated dual-attention),
TimePro (hyper-state selective scan), and CANet (chrono-adaptive
normalization), each exercising the model-local operation that most directly
implements its paper's contribution.
"""

from __future__ import annotations

import unittest

import torch
from pydantic import ValidationError

from moderntsf.models._components.periodic_query_bank import PeriodicQueryBank
from moderntsf.models.tqnet.model import Model as TQNet
from moderntsf.models.tqnet.spec import ModelParameterConfig as TQNetParameters

from moderntsf.models._components.gated_fusion import GatedFusion
from moderntsf.models.gateformer.model import Model as Gateformer
from moderntsf.models.gateformer.spec import ModelParameterConfig as GateformerParameters

from moderntsf.models._components.hyper_state_scan import diagonal_selective_scan, GridStateMixer
from moderntsf.models.timepro.model import Model as TimePro
from moderntsf.models.timepro.spec import ModelParameterConfig as TimeProParameters

from moderntsf.models._components.adain_style_norm import AdaptiveInstanceNorm1d
from moderntsf.models.canet.model import Model as CANet
from moderntsf.models.canet.spec import ModelParameterConfig as CANetParameters


def marks(batch, length):
    result = torch.zeros(batch, length, 6)
    result[..., 0] = 2024
    result[..., 1] = 1
    result[..., 2] = torch.arange(length) + 1
    result[..., 3] = torch.arange(length) % 7
    result[..., 4] = torch.arange(length) % 24
    return result


class PaperStructureTests(unittest.TestCase):
    # -- TQNet -----------------------------------------------------------
    def test_periodic_query_bank_gathers_wrapped_phase_windows(self):
        bank = PeriodicQueryBank(period=4, channels=2)
        with torch.no_grad():
            bank.table.copy_(torch.arange(8.0).reshape(4, 2))
        window = bank(torch.tensor([3, 0]), length=3)
        self.assertEqual(tuple(window.shape), (2, 3, 2))
        torch.testing.assert_close(window[0], bank.table[[3, 0, 1]])
        torch.testing.assert_close(window[1], bank.table[[0, 1, 2]])

    def test_tqnet_forward_uses_temporal_query_as_attention_query(self):
        torch.manual_seed(0)
        model = TQNet(seq_len=8, pred_len=4, enc_in=3, cycle=4, d_model=8, channel_aggre_heads=2)
        x = torch.randn(2, 8, 3)
        out = model(x, marks(2, 8))
        self.assertEqual(tuple(out.shape), (2, 4, 3))
        out.sum().backward()
        self.assertIsNotNone(model.temporal_query_bank.table.grad)
        with self.assertRaises(ValidationError):
            TQNetParameters(enc_in=0)

    # -- Gateformer --------------------------------------------------------
    def test_gated_fusion_interpolates_between_its_two_inputs(self):
        gate = GatedFusion(4)
        with torch.no_grad():
            gate.gate_a.weight.zero_()
            gate.gate_a.bias.fill_(10.0)
            gate.gate_b.weight.zero_()
            gate.gate_b.bias.zero_()
        a, b = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
        torch.testing.assert_close(gate(a, b), a, atol=1e-3, rtol=1e-3)

    def test_gateformer_fuses_temporal_and_variate_branches(self):
        torch.manual_seed(0)
        model = Gateformer(
            seq_len=16, pred_len=4, enc_in=3, patch_len=8, stride=4, d_model=8, n_heads=2, e_layers=1, d_ff=16
        )
        out = model(torch.randn(2, 16, 3))
        self.assertEqual(tuple(out.shape), (2, 4, 3))
        out.sum().backward()
        with self.assertRaises(ValidationError):
            GateformerParameters(enc_in=0)

    # -- TimePro -------------------------------------------------------
    def test_diagonal_selective_scan_matches_manual_recurrence(self):
        u = torch.randn(1, 2, 5)
        delta = torch.rand(1, 2, 5) + 0.1
        a = -torch.rand(2)
        b = torch.randn(1, 2, 5)
        h = diagonal_selective_scan(u, delta, a, b)
        length = u.shape[-1]
        state = torch.zeros(1, 2)
        outs = []
        for t in range(length):
            da = torch.exp(delta[..., t] * a.view(1, -1))
            state = da * state + delta[..., t] * b[..., t] * u[..., t]
            outs.append(state.clone())
        manual = torch.stack(outs, dim=-1)
        torch.testing.assert_close(h, manual, atol=1e-5, rtol=1e-5)

    def test_grid_state_mixer_preserves_grid_shape(self):
        mixer = GridStateMixer(channels=4)
        grid = torch.randn(2, 4, 3, 5)
        out = mixer(grid)
        self.assertEqual(tuple(out.shape), tuple(grid.shape))

    def test_timepro_forward_and_gradient(self):
        torch.manual_seed(0)
        model = TimePro(
            seq_len=16, pred_len=4, enc_in=3, patch_len=8, stride=4, d_model=8, e_layers=1, d_state=1
        )
        out = model(torch.randn(2, 16, 3))
        self.assertEqual(tuple(out.shape), (2, 4, 3))
        out.sum().backward()
        with self.assertRaises(ValidationError):
            TimeProParameters(enc_in=0)

    # -- CANet -----------------------------------------------------------
    def test_adain_rescales_normalized_features_to_style_statistics(self):
        adain = AdaptiveInstanceNorm1d()
        x = torch.randn(2, 5, 4) * 3 + 1
        style_mean = torch.ones(2, 1, 4) * 2.0
        style_std = torch.ones(2, 1, 4) * 0.5
        out = adain(x, style_mean, style_std)
        torch.testing.assert_close(out.mean(dim=1), style_mean.squeeze(1), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(out.std(dim=1, unbiased=True), style_std.squeeze(1), atol=5e-2, rtol=5e-2)

    def test_canet_forward_and_gradient(self):
        torch.manual_seed(0)
        model = CANet(seq_len=32, pred_len=4, enc_in=3, patch_sizes=(8, 16), embed_dim=8)
        out = model(torch.randn(2, 32, 3))
        self.assertEqual(tuple(out.shape), (2, 4, 3))
        out.sum().backward()
        with self.assertRaises(ValidationError):
            CANetParameters(enc_in=0)


if __name__ == "__main__":
    unittest.main()
