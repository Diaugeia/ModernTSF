"""Equation, structure, and runtime tests for the xPatch implementation."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from tsflab.models.xpatch.layers import ExponentialDecomposition
from tsflab.models.xpatch.model import Model


class ExponentialDecompositionTests(unittest.TestCase):
    def test_ema_matches_paper_recurrence_and_residual_identity(self) -> None:
        x = torch.tensor([[[1.0], [3.0], [5.0]]])
        seasonal, trend = ExponentialDecomposition(alpha=0.5)(x)
        torch.testing.assert_close(trend, torch.tensor([[[1.0], [2.0], [3.5]]]))
        torch.testing.assert_close(seasonal + trend, x)

    def test_dema_is_explicit_holt_level_and_trend_extension(self) -> None:
        x = torch.tensor([[[1.0], [3.0], [4.0]]])
        seasonal, trend = ExponentialDecomposition(
            alpha=0.5, beta=0.5, kind="dema"
        )(x)
        torch.testing.assert_close(trend, torch.tensor([[[1.0], [3.0], [4.5]]]))
        torch.testing.assert_close(seasonal + trend, x)


class XPatchStructureTests(unittest.TestCase):
    def test_defining_dual_stream_structure(self) -> None:
        model = Model(8, 3, 2, patch_len=4, stride=2, hidden_dim=8)
        linear = model.forecaster.linear_stream
        nonlinear = model.forecaster.nonlinear_stream
        self.assertFalse(any(isinstance(module, nn.GELU) for module in linear.modules()))
        self.assertEqual(nonlinear.num_patches, (8 - 4) // 2 + 2)
        self.assertEqual(nonlinear.depthwise.groups, nonlinear.num_patches)
        self.assertEqual(nonlinear.depthwise.kernel_size, (4,))
        self.assertEqual(nonlinear.depthwise.stride, (4,))
        self.assertEqual(nonlinear.pointwise.kernel_size, (1,))

    def test_channel_independent_mapping_is_permutation_equivariant(self) -> None:
        torch.manual_seed(1931)
        model = Model(8, 3, 3, patch_len=4, stride=2, hidden_dim=8).eval()
        x = torch.randn(2, 8, 3)
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model(x[:, :, permutation]), model(x)[:, :, permutation])


class XPatchRuntimeTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(1933)
        model = Model(8, 3, 2, patch_len=4, stride=2, hidden_dim=8).cpu()
        x = torch.randn(2, 8, 2, requires_grad=True)
        marks = torch.randn(2, 8, 4)
        adjacency = torch.eye(2)
        output = model(x, marks, adjacency)
        self.assertEqual(tuple(output.shape), (2, 3, 2))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().max().item(), 0.0, name)

        model.eval()
        expected = model(x.detach(), marks, adjacency)
        clone = Model(8, 3, 2, patch_len=4, stride=2, hidden_dim=8).eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()))
        torch.testing.assert_close(clone(x.detach()), expected)
        torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))
        self.assertEqual(tuple(model(torch.randn(1, 8, 2)).shape), (1, 3, 2))

        boundary = Model(1, 1, 1, patch_len=4, stride=2, hidden_dim=4)
        self.assertEqual(tuple(boundary(torch.randn(1, 1, 1)).shape), (1, 1, 1))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 7, 2))


if __name__ == "__main__":
    unittest.main()
