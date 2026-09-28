"""Equation and runtime tests for the third recent-model batch."""

from __future__ import annotations

import copy
import math
import unittest

import torch

from moderntsf.models.fets.model import Model as FeTS
from moderntsf.models.implicitforecaster.model import Model as ImplicitForecaster
from moderntsf.models.occamvts.model import Model as OccamVTS
from moderntsf.models.pmdformer.model import Model as PMDformer


CASES = {
    "FeTS": lambda length, horizon, channels: FeTS(
        length, horizon, channels, d_model=4, patch_len=min(4, length),
        stride=min(2, length), fourier_order=1, polynomial_order=2,
    ),
    "ImplicitForecaster": lambda length, horizon, channels: ImplicitForecaster(
        length, horizon, channels, d_model=4, frequency_pool=max(2, 2 * horizon)
    ),
    "OccamVTS": lambda length, horizon, channels: OccamVTS(
        length, horizon, channels, d_model=4, patch_len=min(4, length),
        stride=min(2, length), period=4, num_heads=1,
    ),
    "PMDformer": lambda length, horizon, channels: PMDformer(
        length, horizon, channels, d_model=4, patch_len=min(4, length), num_heads=1
    ),
}


class RecentThirdEquationTests(unittest.TestCase):
    def test_fets_mask_is_binary_in_forward_and_center_thresholded(self) -> None:
        model = CASES["FeTS"](8, 3, 2)
        tokens = torch.linspace(-1, 1, 32).reshape(1, 2, 4, 4)
        _, scores = model.adaptive_features(tokens)
        mask, direct_scores = model.importance(tokens.reshape(-1, 4))
        self.assertTrue(bool(((mask == 0) | (mask == 1)).all()))
        torch.testing.assert_close(scores.reshape(-1, 4), direct_scores)
        expected = (direct_scores >= direct_scores.mean(-1, keepdim=True)).to(mask.dtype)
        torch.testing.assert_close(mask, expected)

    def test_implicit_decoder_respects_amplitude_and_phase_domains(self) -> None:
        model = CASES["ImplicitForecaster"](8, 3, 2)
        amplitude, phase = model.spectral_parameters(torch.randn(2, 8, 2))
        self.assertEqual(amplitude.shape[-1], model.pool_bins)
        self.assertTrue(bool((amplitude >= 0).all()))
        self.assertTrue(bool((phase >= -math.pi).all() and (phase <= math.pi).all()))

    def test_occam_visual_augmentation_contains_periodic_coordinates(self) -> None:
        model = CASES["OccamVTS"](8, 3, 2)
        history = torch.zeros(1, 2, 8)
        augmented = model.visual_augmentation(history)
        self.assertEqual(tuple(augmented.shape), (1, 2, 4, 8))
        torch.testing.assert_close(augmented[:, :, 2, 0], torch.zeros(1, 2))
        torch.testing.assert_close(augmented[:, :, 3, 0], torch.ones(1, 2))
        torch.testing.assert_close(augmented[:, :, 2, 4], augmented[:, :, 2, 0], atol=1e-6, rtol=0)

    def test_pmd_decouples_exact_patch_means(self) -> None:
        model = CASES["PMDformer"](8, 3, 2)
        x = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        residuals, means = model.patch_mean_decouple(x)
        torch.testing.assert_close(residuals.mean(-1), torch.zeros_like(means))
        reconstructed = residuals + means.unsqueeze(-1)
        torch.testing.assert_close(reconstructed.flatten(-2), x.transpose(1, 2))


class RecentThirdRuntimeTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(260831)
        for name, factory in CASES.items():
            with self.subTest(model=name):
                model = factory(8, 3, 2).cpu().eval()
                x = torch.randn(2, 8, 2, requires_grad=True)
                marks, adjacency = torch.randn(2, 8, 3), torch.eye(2)
                output = model(x, marks, adjacency)
                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertGreater(x.grad.abs().max().item(), 0.0)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)
                clone = factory(8, 3, 2).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach()), output.detach())
                self.assertEqual(tuple(model(torch.randn(1, 8, 2)).shape), (1, 3, 2))
                self.assertEqual(tuple(factory(1, 1, 2)(torch.randn(1, 1, 2)).shape), (1, 1, 2))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 7, 2))
                torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))


if __name__ == "__main__":
    unittest.main()
