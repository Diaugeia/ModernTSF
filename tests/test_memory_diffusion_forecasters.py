"""Paper-equation and runtime tests for six forecasting implementations."""

from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models.gotsf.model import Model as GOTSF
from moderntsf.models.gtr.model import Model as GTR
from moderntsf.models.hmformer.model import Model as HMformer
from moderntsf.models.kronos.model import Model as Kronos
from moderntsf.models.mafs.model import Model as MAFS
from moderntsf.models.mmpd.model import Model as MMPD


CASES = {
    "GOTSF": lambda length, horizon, channels: GOTSF(
        length, horizon, channels, d_model=8, dropout=0, num_intervals=3,
        interval_min=-1, interval_max=2, decay_rate=2,
    ),
    "GTR": lambda length, horizon, channels: GTR(
        length, horizon, channels, d_model=8, dropout=0,
        cycle_length=max(length, 12), local_period=3,
    ),
    "HMformer": lambda length, horizon, channels: HMformer(
        length, horizon, channels, d_model=8, dropout=0,
        patch_len=min(2, length), stride=1, num_scales=2, depth=1, num_heads=2,
    ),
    "Kronos": lambda length, horizon, channels: Kronos(
        length, horizon, channels, d_model=8, dropout=0,
        code_bits=4, num_layers=1, num_heads=2,
    ),
    "MAFS": lambda length, horizon, channels: MAFS(
        length, horizon, channels, d_model=8, dropout=0,
        num_agents=3, num_layers=1, num_heads=2, topology="star",
    ),
    "MMPD": lambda length, horizon, channels: MMPD(
        length, horizon, channels, d_model=8, dropout=0,
        patch_len=min(2, horizon), num_heads=2, adjacent_range=1,
        diffusion_steps=10, denoiser_depth=1,
    ),
}


class DefiningEquationTests(unittest.TestCase):
    def test_gotsf_soft_boundary_and_intersection_equations(self) -> None:
        model = CASES["GOTSF"](8, 3, 2)
        target = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        # First interval is [-1, 0], so only values outside it decay.
        expected = torch.exp(-2 * torch.tensor([0.0, 0.0, 0.5, 1.0]))
        torch.testing.assert_close(model.decay(target, 0), expected)
        self.assertEqual(model.intersecting_intervals((0.2, 0.8)).tolist(), [False, True, False])

    def test_gtr_cycle_alignment_and_joint_kernel(self) -> None:
        model = CASES["GTR"](8, 3, 2)
        indices = model.retriever.cycle_indices(torch.tensor([10, 11]), 2, torch.device("cpu"))
        torch.testing.assert_close(indices[0], torch.tensor([10, 11, 0, 1, 2, 3, 4, 5]))
        self.assertEqual(model.retriever.fusion.kernel_size, (2, 3))
        with torch.no_grad():
            model.retriever.global_embedding.normal_()
        sample = torch.randn(1, 8, 2)
        first = model.forecast_at(sample, start_index=0)
        second = model.forecast_at(sample, start_index=1)
        self.assertEqual(first.shape, second.shape)
        self.assertGreater((first - second).abs().max().item(), 0.0)

    def test_hmformer_safe_widths_and_cross_scale_mixing(self) -> None:
        model = CASES["HMformer"](8, 3, 2)
        self.assertEqual([branch.embedding.out_channels for branch in model.branches], [8, 16])
        self.assertEqual(model.cross_scale[0].kernel_size, (2,))
        self.assertEqual(model.cross_scale[0].stride, (2,))
        representations = model.branch_representations(torch.randn(1, 8, 2))
        self.assertEqual([value.shape[-1] for value in representations], [8, 16])

    def test_kronos_uses_binary_coarse_then_fine_probabilities(self) -> None:
        model = CASES["Kronos"](8, 3, 2)
        self.assertEqual(tuple(model.subtoken_codebook.shape), (4, 2))
        self.assertTrue(bool(((model.subtoken_codebook == -1) | (model.subtoken_codebook == 1)).all()))
        output = model(torch.randn(1, 8, 2))
        self.assertEqual(output.shape, (1, 3, 2))
        for probabilities in model.last_coarse_probabilities + model.last_fine_probabilities:
            torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))
        self.assertTrue(torch.isfinite(model.tokenizer_loss(torch.randn(1, 8, 2))))

    def test_mafs_topology_normalization_specialization_and_vote(self) -> None:
        model = CASES["MAFS"](8, 6, 2)
        adjacency = model.normalized_adjacency()
        torch.testing.assert_close(adjacency, adjacency.T)
        self.assertEqual(adjacency[1, 2].item(), 0.0)
        targets = model.specialization_targets(torch.randn(1, 6, 2))
        self.assertEqual([target.shape[1] for target in targets], [2, 4, 6])
        self.assertTrue(
            torch.isfinite(model.specialization_loss(torch.randn(1, 8, 2), torch.randn(1, 6, 2)))
        )
        model(torch.randn(2, 8, 2))
        assert model.last_voting_weights is not None
        torch.testing.assert_close(
            model.last_voting_weights.sum(-1), torch.ones(2), atol=1e-6, rtol=1e-6
        )

    def test_mmpd_neighbour_windows_and_diffusion_objective(self) -> None:
        model = CASES["MMPD"](8, 4, 2)
        patches = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        previous, following = model.denoiser.adjacent_patches(patches)
        torch.testing.assert_close(previous, torch.tensor([[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]]))
        torch.testing.assert_close(following, torch.tensor([[[3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]]))
        loss = model.diffusion_loss(torch.randn(2, 8, 2), torch.randn(2, 4, 2))
        self.assertTrue(torch.isfinite(loss))
        samples = model.sample(torch.randn(1, 8, 2), num_samples=2, steps=2)
        self.assertEqual(samples.shape, (2, 1, 4, 2))


class CompleteRuntimeTests(unittest.TestCase):
    def test_forward_backward_serialization_and_boundaries(self) -> None:
        torch.manual_seed(260827)
        for name, factory in CASES.items():
            with self.subTest(model=name):
                model = factory(8, 3, 2).cpu().eval()
                if name == "GTR":
                    with torch.no_grad():
                        model.retriever.global_embedding.normal_(std=0.1)
                x = torch.randn(2, 8, 2, requires_grad=True)
                marks, adjacency = torch.randn(2, 8, 3), torch.eye(2)
                output = model(x, marks, adjacency)
                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())
                loss = output.square().mean()
                if name == "MMPD":
                    loss = loss + model.diffusion_loss(x, torch.randn(2, 3, 2))
                loss.backward()
                self.assertIsNotNone(x.grad)
                self.assertGreater(x.grad.abs().max().item(), 0.0)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)

                clone = factory(8, 3, 2).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach()), model(x.detach()))
                torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))
                self.assertEqual(tuple(model(torch.randn(1, 8, 2)).shape), (1, 3, 2))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 7, 2))

        boundary_models = (
            GOTSF(1, 1, 1, d_model=4, num_intervals=2),
            GTR(1, 1, 1, d_model=4, cycle_length=1, local_period=1),
            HMformer(1, 1, 1, d_model=4, patch_len=1, stride=1, num_scales=1, num_heads=1),
            Kronos(1, 1, 1, d_model=4, code_bits=2, num_layers=1, num_heads=1),
            MAFS(1, 1, 1, d_model=4, num_agents=2, num_layers=1, num_heads=1),
            MMPD(1, 1, 1, d_model=4, patch_len=1, num_heads=1, diffusion_steps=4),
        )
        for model in boundary_models:
            self.assertEqual(tuple(model(torch.randn(1, 1, 1)).shape), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
