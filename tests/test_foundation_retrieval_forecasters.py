"""Paper-equation and runtime checks for six recent implementations."""

from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.pulse.model import Model as PULSE
from tsflab.models.symtime.model import Model as SymTime
from tsflab.models.timecap.model import Model as TimeCAP
from tsflab.models.timeo1.model import Model as TimeO1
from tsflab.models.tirex.model import Model as TiRex
from tsflab.models.tsrag.model import Model as TSRAG


CASES = {
    "PULSE": lambda length, horizon, channels: PULSE(
        length, horizon, channels, d_model=8, phase_period=4,
        phase_resolution=4, router_heads=2, dropout=0.0,
    ),
    "SymTime": lambda length, horizon, channels: SymTime(
        length, horizon, channels, d_model=8, patch_len=min(4, length),
        num_layers=1, num_heads=2, trend_kernel=3, dropout=0.0,
    ),
    "TimeCAP": lambda length, horizon, channels: TimeCAP(
        length, horizon, channels, d_model=8, patch_len=min(4, length),
        group_size=min(2, channels), group_stride=1, num_heads=2, dropout=0.0,
    ),
    "TimeO1": lambda length, horizon, channels: TimeO1(
        length, horizon, channels, d_model=8, alpha=0.75, rank_ratio=0.5,
    ),
    "TiRex": lambda length, horizon, channels: TiRex(
        length, horizon, channels, d_model=8, patch_len=min(2, length),
        num_layers=1, dropout=0.0,
    ),
    "TSRAG": lambda length, horizon, channels: TSRAG(
        length, horizon, channels, d_model=8, top_k=2, memory_size=3,
        num_heads=2, dropout=0.0,
    ),
}


class PhaseFoundationRetrievalEquationTests(unittest.TestCase):
    def test_pulse_phase_anchor_and_residual_normalization(self) -> None:
        model = CASES["PULSE"](8, 3, 2)
        self.assertEqual(model.phase_indices(8, future=False, device=torch.device("cpu")).tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
        x = torch.randn(2, 8, 2)
        normalized, anchor, mean, scale = model.disentangle(x)
        normalized_residual = normalized - anchor
        torch.testing.assert_close(normalized_residual.mean(1), torch.zeros(2, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close(normalized_residual.var(1, unbiased=False), torch.ones(2, 2), atol=2e-4, rtol=0)
        self.assertEqual(mean.shape, (2, 1, 2))
        self.assertEqual(scale.shape, (2, 1, 2))
        permutation = torch.tensor([1, 0])
        mixed, mixed_mean, mixed_scale = model.statistic_aware_mixup(
            normalized, mean, scale, permutation, 0.25
        )
        torch.testing.assert_close(mixed, 0.25 * normalized + 0.75 * normalized[permutation])
        torch.testing.assert_close(mixed_mean, 0.25 * mean + 0.75 * mean[permutation])
        torch.testing.assert_close(mixed_scale, 0.25 * scale + 0.75 * scale[permutation])
        self.assertTrue(bool((mixed_scale > 0).all()))
        torch.testing.assert_close(model.frequency_mae(x, x), torch.tensor(0.0))

    def test_symtime_nonoverlapping_patches_and_decomposition(self) -> None:
        model = CASES["SymTime"](8, 3, 2)
        x = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        periodic, trend = model.decomposition(x)
        torch.testing.assert_close(periodic + trend, x)
        patches = model.patch_series(periodic)
        self.assertEqual(tuple(patches.shape), (1, 2, 2, 4))
        torch.testing.assert_close(patches[0, 0, 0], periodic[0, :4, 0])

    def test_timecap_masks_are_time_aligned_and_groups_overlap(self) -> None:
        model = CASES["TimeCAP"](8, 3, 3)
        self.assertGreater(model.group_indices.numel(), model.enc_in)
        query = torch.tensor([0, 0, 1, 1])
        mask = model.channel_aware_mask(query, query)
        self.assertFalse(bool(mask[0, 1]))
        self.assertTrue(bool(mask[0, 2]))
        representation = model.groupwise_representation(torch.randn(2, 8, 3))
        self.assertEqual(tuple(representation.shape), (2, 3, 2, 8))

    def test_timeo1_svd_basis_and_loss_equation(self) -> None:
        model = CASES["TimeO1"](8, 4, 2)
        labels = torch.randn(32, 4, 2)
        basis = model.fit_projection(labels)
        identity = torch.eye(4).expand(2, -1, -1)
        torch.testing.assert_close(basis.transpose(-1, -2) @ basis, identity, atol=1e-5, rtol=1e-5)
        prediction = torch.randn(3, 4, 2)
        target = torch.randn(3, 4, 2)
        retained = 2
        expected = 0.75 * (model.transform(prediction)[:, :retained] - model.transform(target)[:, :retained]).abs().sum()
        expected = expected + 0.25 * (prediction - target).square().sum()
        torch.testing.assert_close(model.transformed_alignment_loss(prediction, target), expected)

    def test_tirex_cpm_and_non_crossing_quantiles(self) -> None:
        mask = TiRex.contiguous_patch_mask(2, 5, 2, 1.0)
        self.assertTrue(bool(mask.all()))
        output = CASES["TiRex"](8, 3, 2)(torch.randn(2, 8, 2))
        self.assertEqual(tuple(output.shape), (2, 3, 2, 9))
        self.assertTrue(bool((output[..., 1:] >= output[..., :-1]).all()))

    def test_tsrag_euclidean_retrieval_and_arm_weights(self) -> None:
        model = CASES["TSRAG"](8, 3, 2)
        query = torch.zeros(1, 8, 2)
        contexts = torch.stack([
            torch.zeros(8, 2), torch.ones(8, 2), torch.full((8, 2), 3.0)
        ]).unsqueeze(0)
        indices, distances = model.retrieve(query, contexts)
        self.assertEqual(indices[0, 0].item(), 0)
        self.assertTrue(bool((distances[:, 1:] >= distances[:, :-1]).all()))
        mixed, weights = model.arm(torch.randn(1, 8), torch.randn(1, 2, 8))
        self.assertEqual(tuple(mixed.shape), (1, 8))
        torch.testing.assert_close(weights.sum(-1), torch.ones(1))
        futures = torch.randn(1, 3, 3, 2)
        output = model.forecast_with_retrieval(
            query,
            retrieval_contexts=contexts,
            retrieval_futures=futures,
        )
        self.assertEqual(tuple(output.shape), (1, 3, 2))


class PhaseFoundationRetrievalRuntimeTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(260827)
        for name, factory in CASES.items():
            with self.subTest(model=name):
                model = factory(8, 3, 2).cpu().eval()
                x = torch.randn(2, 8, 2, requires_grad=True)
                marks, adjacency = torch.randn(2, 8, 6), torch.eye(2)
                output = model(x, marks, adjacency)
                expected = (2, 3, 2, 9) if name == "TiRex" else (2, 3, 2)
                self.assertEqual(tuple(output.shape), expected)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertGreater(x.grad.abs().max().item(), 0.0)
                for parameter_name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)
                clone = factory(8, 3, 2).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach(), marks, adjacency), output.detach())
                batch_one_shape = (1, 3, 2, 9) if name == "TiRex" else (1, 3, 2)
                self.assertEqual(tuple(model(torch.randn(1, 8, 2)).shape), batch_one_shape)
                boundary = self._boundary(name)
                boundary_shape = (1, 1, 2, 9) if name == "TiRex" else (1, 1, 2)
                self.assertEqual(tuple(boundary(torch.randn(1, 1, 2)).shape), boundary_shape)
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 7, 2))
                torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))

    @staticmethod
    def _boundary(name: str):
        if name == "PULSE":
            return PULSE(1, 1, 2, d_model=4, phase_period=1, phase_resolution=1, router_heads=1, dropout=0.0).eval()
        if name == "SymTime":
            return SymTime(1, 1, 2, d_model=4, patch_len=1, num_layers=1, num_heads=1, trend_kernel=1, dropout=0.0).eval()
        if name == "TimeCAP":
            return TimeCAP(1, 1, 2, d_model=4, patch_len=1, group_size=2, group_stride=1, num_heads=1, dropout=0.0).eval()
        if name == "TimeO1":
            return TimeO1(1, 1, 2, d_model=4).eval()
        if name == "TiRex":
            return TiRex(1, 1, 2, d_model=4, patch_len=1, num_layers=1, dropout=0.0).eval()
        return TSRAG(1, 1, 2, d_model=4, top_k=1, memory_size=1, num_heads=1, dropout=0.0).eval()


if __name__ == "__main__":
    unittest.main()
