"""Focused equation and structure tests for recent implementations."""

from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models.apn.model import Model as APN
from moderntsf.models.cora.model import Model as CoRA
from moderntsf.models.hn_mvts.model import Model as HNMVTS
from moderntsf.models.interpdn.model import Model as InterPDN
from moderntsf.models.olinear.model import Model as OLinear, NormLin
from moderntsf.models.phaseformer.model import CrossPhaseRouter, Model as PhaseFormer
from moderntsf.models.sempo.model import Model as SEMPO
from moderntsf.models.sonnet.model import Model as Sonnet
from moderntsf.models.timemosaic.model import Model as TimeMosaic


PRIOR_CASES = {
    "OLinear": lambda length, horizon, channels: OLinear(length, horizon, channels, d_model=4),
    "PhaseFormer": lambda length, horizon, channels: PhaseFormer(
        length, horizon, channels, d_model=4, period=3, num_routers=2
    ),
    "InterPDN": lambda length, horizon, channels: InterPDN(
        length, horizon, channels, support_size=7
    ),
    "Sonnet": lambda length, horizon, channels: Sonnet(
        length, horizon, channels, d_model=4, num_wavelets=2
    ),
}


class PriorPaperEquationTests(unittest.TestCase):
    def test_normlin_is_positive_row_normalized_channel_mixing(self) -> None:
        layer = NormLin(2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[0.0, 1.0], [-1.0, 2.0]]))
        weight = layer.normalized_weight()
        self.assertTrue((weight > 0).all())
        torch.testing.assert_close(weight.sum(dim=-1), torch.ones(2))
        x = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2, 1)
        torch.testing.assert_close(layer(x), torch.einsum("oc,bcld->bold", weight, x))

    def test_olinear_requires_orthogonal_transform_bases(self) -> None:
        model = OLinear(4, 3, 2, d_model=4)
        input_basis, _ = torch.linalg.qr(torch.randn(4, 4))
        output_basis, _ = torch.linalg.qr(torch.randn(3, 3))
        model.set_orthogonal_bases(input_basis, output_basis)
        torch.testing.assert_close(model.input_basis.T @ model.input_basis, torch.eye(4))
        with self.assertRaises(AssertionError):
            model.set_orthogonal_bases(torch.ones(4, 4), output_basis)

    def test_phaseformer_uses_two_stage_cross_phase_routing(self) -> None:
        model = PhaseFormer(5, 4, 2, d_model=4, period=3, num_routers=2)
        self.assertEqual(tuple(model._tokenize(torch.randn(1, 5, 2)).shape), (1, 2, 3, 2))
        self.assertIsInstance(model.layers[0], CrossPhaseRouter)
        self.assertEqual(tuple(model.layers[0].routers.shape), (2, 4))

    def test_interpdn_outputs_normalized_interleaved_distributions(self) -> None:
        model = InterPDN(4, 3, 2, support_size=7)
        model(torch.randn(2, 4, 2))
        self.assertIsNotNone(model.last_probabilities)
        first, second = model.last_probabilities or (None, None)
        assert first is not None and second is not None
        torch.testing.assert_close(first.sum(dim=-1), torch.ones_like(first[..., 0]))
        torch.testing.assert_close(second.sum(dim=-1), torch.ones_like(second[..., 0]))
        self.assertFalse(torch.equal(model.support_first, model.support_second))

    def test_sonnet_wavelets_and_koopman_operator_follow_contract(self) -> None:
        model = Sonnet(4, 3, 2, d_model=4, num_wavelets=2)
        atoms = model.wavelets.atoms()
        self.assertEqual(tuple(atoms.shape), (2, 4, 4))
        operator = model.koopman.operator()
        torch.testing.assert_close(operator.mH @ operator, torch.eye(2, dtype=operator.dtype))


class PriorRewriteRuntimeTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(2026)
        for name, factory in PRIOR_CASES.items():
            with self.subTest(model=name):
                model = factory(4, 3, 2).cpu().eval()
                x = torch.randn(2, 4, 2, requires_grad=True)
                marks = torch.randn(2, 4, 3)
                adjacency = torch.eye(2)
                output = model(x, marks, adjacency)
                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)
                clone = factory(4, 3, 2).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach()), output.detach())
                self.assertEqual(tuple(model(torch.randn(1, 4, 2)).shape), (1, 3, 2))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 3, 2))
                torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))


class RecentCleanRoomRewriteTests(unittest.TestCase):
    def test_apn_soft_windows_and_time_contract(self) -> None:
        model = APN(8, 3, 2, d_model=8, d_time=4, num_patches=4)
        times = torch.tensor([[0.0, 0.05, 0.2, 0.21, 0.5, 0.7, 0.9, 1.0]])
        weights = model.patch_weights(times)
        self.assertEqual(weights.shape, (1, 2, 8, 4))
        self.assertTrue(bool((weights > 0).all()))
        output = model(torch.randn(1, 8, 2), times)
        self.assertEqual(output.shape, (1, 3, 2))

    def test_cora_low_rank_polynomial_correlation(self) -> None:
        model = CoRA(8, 3, 3, d_model=8, rank=2, polynomial_order=2, use_revin=False)
        x = torch.randn(2, 8, 3)
        representation = model.encoder(x.transpose(1, 2))
        correlation = model.dynamic_correlation(x, representation)
        self.assertEqual(correlation.shape, (2, 3, 3))
        self.assertEqual(model.polynomial_basis.shape, (3, 2))
        self.assertTrue(bool(torch.isfinite(correlation).all()))

    def test_hn_mvts_generated_projection_is_the_decoder(self) -> None:
        model = HNMVTS(4, 2, 2, d_model=3, embedding_dim=2, hyper_hidden=4, use_revin=False)
        x = torch.randn(1, 4, 2)
        hidden = model.temporal_encoder(x.transpose(1, 2))
        weights, bias = model.generated_projection()
        expected = (torch.einsum("bcd,chd->bch", hidden, weights) + bias.unsqueeze(0)).transpose(1, 2)
        torch.testing.assert_close(model(x), expected)

    def test_sempo_energy_branches_and_prompt_routing(self) -> None:
        model = SEMPO(8, 3, 2, d_model=8, patch_len=2, num_prompts=3, num_heads=2, dropout=0)
        history = torch.randn(2, 2, 8)
        reconstructed, high, low = model.energy_aware_decomposition(history)
        self.assertEqual(reconstructed.shape, history.shape)
        self.assertEqual(high.shape, low.shape)
        self.assertEqual(model.router.out_features, 3)
        self.assertEqual(model(torch.randn(2, 8, 2)).shape, (2, 3, 2))

    def test_timemosaic_aligned_candidates_and_segments(self) -> None:
        model = TimeMosaic(8, 5, 2, d_model=8, patch_sizes=(2, 4), num_segments=3,
                           num_heads=2, dropout=0)
        tokens, choices = model.adaptive_patch_tokens(torch.randn(2, 8))
        self.assertEqual(tokens.shape, (2, 4, 8))
        torch.testing.assert_close(choices.sum(-1), torch.ones_like(choices[..., 0]))
        self.assertEqual([head.out_features for head in model.segment_heads], [2, 2, 1])
        self.assertEqual(model(torch.randn(1, 8, 2)).shape, (1, 5, 2))


if __name__ == "__main__":
    unittest.main()
