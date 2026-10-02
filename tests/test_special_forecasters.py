"""Equation and runtime tests for six special-model implementations."""

from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.amrc.model import Model as AMRC
from tsflab.models.aurora.model import Model as Aurora
from tsflab.models.cosa.model import Model as COSA
from tsflab.models.distdf.model import Model as DistDF
from tsflab.models.dynamic_tmoe.model import Model as DynamicTMoE
from tsflab.models.ftp.model import Model as FTP, _right_padded_patches


CASES = {
    "AMRC": lambda length, horizon, channels: AMRC(
        length, horizon, channels, d_model=8, mask_samples=min(2, length)
    ),
    "Aurora": lambda length, horizon, channels: Aurora(
        length, horizon, channels, d_model=8, patch_len=min(4, length),
        num_heads=2, num_distill_tokens=1, num_prototypes=3, flow_steps=1,
        dropout=0,
    ),
    "COSA": lambda length, horizon, channels: COSA(
        length, horizon, channels, context_len=min(3, length)
    ),
    "DistDF": lambda length, horizon, channels: DistDF(length, horizon, channels),
    "DynamicTMoE": lambda length, horizon, channels: DynamicTMoE(
        length, horizon, channels, d_model=8, patch_len=min(4, length),
        stride=min(2, length), top_k=3, memory_slots=2, relation_period=4,
    ),
    "FTP": lambda length, horizon, channels: FTP(
        length, horizon, channels, d_model=8, num_layers=1,
        patch_unit=min(2, length), num_scales=2, stride=min(2, length),
        dropout=0,
    ),
}


class AdapterBatchAEquationTests(unittest.TestCase):
    def test_amrc_prefix_mask_and_esp_equations(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        masked = AMRC.prefix_mask(x, torch.tensor([1, 2]))
        torch.testing.assert_close(masked[0, 0], torch.zeros(2))
        torch.testing.assert_close(masked[1, :2], torch.zeros(2, 2))
        embedding = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(
            AMRC.embedding_similarity_penalty(embedding, target).item(), 0.75
        )

    def test_aurora_modality_guidance_changes_representation(self) -> None:
        torch.manual_seed(7)
        model = CASES["Aurora"](8, 3, 2).eval()
        x = torch.randn(2, 8, 2)
        baseline, _, _ = model.encode(x)
        text = torch.randn(2, 2, model.d_model)
        guided, distilled_text, distilled_image = model.encode(x, text_context=text)
        self.assertEqual(tuple(distilled_text.shape), (4, 1, model.d_model))
        self.assertEqual(tuple(distilled_image.shape), (4, 1, model.d_model))
        self.assertGreater((baseline - guided).abs().max().item(), 0)

    def test_cosa_exact_output_correction(self) -> None:
        model = COSA(4, 2, 1, context_len=2, gate_init=0.25)
        with torch.no_grad():
            model.residual.weight.fill_(0.5)
            model.residual.bias.fill_(0.1)
        base = torch.tensor([[[2.0], [3.0]]])
        context = torch.tensor([[[4.0], [5.0]]])
        result = model.correct(base, context)
        correction = 0.5 * torch.tensor([2.0, 3.0, 4.0, 5.0]).sum() + 0.1
        torch.testing.assert_close(result, base + model.gate.tanh() * correction)

    def test_distdf_bures_matches_diagonal_closed_form(self) -> None:
        mean_a = torch.tensor([0.0, 1.0])
        mean_b = torch.tensor([1.0, 3.0])
        covariance_a = torch.diag(torch.tensor([1.0, 4.0]))
        covariance_b = torch.diag(torch.tensor([9.0, 1.0]))
        expected = (mean_a - mean_b).square().sum() + torch.tensor(5.0)
        torch.testing.assert_close(
            DistDF.bures_wasserstein(mean_a, mean_b, covariance_a, covariance_b),
            expected,
        )

    def test_dynamic_tmoe_mmd_and_routing_equations(self) -> None:
        model = CASES["DynamicTMoE"](8, 3, 2)
        reference = torch.tensor([[[0.0], [0.1], [-0.1]]])
        identical = model.rbf_mmd(reference, reference)
        shifted = model.rbf_mmd(reference, reference + 5)
        torch.testing.assert_close(
            identical, torch.zeros_like(identical), atol=1e-6, rtol=0
        )
        self.assertGreater(shifted.item(), identical.item())
        threshold = model.adaptive_threshold(torch.tensor([[1.0, 2.0, 3.0]]), 1.0)
        torch.testing.assert_close(threshold, torch.tensor([2.0 + (2 / 3) ** 0.5]))
        weights, _ = model.routing_weights(
            torch.randn(2, model.num_patches, 2, model.d_model)
        )
        torch.testing.assert_close(
            weights.sum(-1), torch.ones_like(weights[..., 0])
        )

    def test_ftp_patch_and_channel_enhancement_contract(self) -> None:
        x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        patches = _right_padded_patches(x, patch_len=4, stride=2)
        self.assertEqual(tuple(patches.shape), (1, 1, 4, 4))
        model = CASES["FTP"](8, 3, 2)
        enhanced, weights = model.layers[0].channel_enhancement(
            torch.randn(2, 2, 8)
        )
        self.assertEqual(tuple(enhanced.shape), (2, 2, 8))
        torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[0]))


class AdapterBatchARuntimeTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(260827)
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
                self.assertGreater(x.grad.abs().max().item(), 0)
                for parameter_name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0, parameter_name)
                clone = factory(8, 3, 2).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach()), output.detach())
                self.assertEqual(tuple(model(torch.randn(1, 8, 2)).shape), (1, 3, 2))
                boundary = factory(1, 1, 2).cpu().eval()
                self.assertEqual(tuple(boundary(torch.randn(1, 1, 2)).shape), (1, 1, 2))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 7, 2))
                torch.testing.assert_close(
                    model(x.detach(), marks, adjacency), model(x.detach())
                )

    def test_training_objectives_are_finite_and_differentiable(self) -> None:
        torch.manual_seed(11)
        x = torch.randn(4, 8, 2)
        target = torch.randn(4, 3, 2)
        amrc = CASES["AMRC"](8, 3, 2)
        amrc_forecast, amrc_loss, amrc_parts = amrc.training_objective(
            x, target, mask_lengths=torch.tensor([1, 4])
        )
        self.assertEqual(tuple(amrc_forecast.shape), tuple(target.shape))
        self.assertEqual(set(amrc_parts), {"prediction", "aml", "esp"})
        self.assertTrue(torch.isfinite(amrc_loss))
        amrc_loss.backward()
        distdf = CASES["DistDF"](8, 3, 2)
        distdf_forecast, distdf_loss, distdf_parts = distdf.training_objective(x, target)
        self.assertEqual(tuple(distdf_forecast.shape), tuple(target.shape))
        self.assertEqual(set(distdf_parts), {"mse", "joint_wasserstein"})
        self.assertTrue(torch.isfinite(distdf_loss))
        distdf_loss.backward()


if __name__ == "__main__":
    unittest.main()
