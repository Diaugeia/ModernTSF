"""Paper-structure checks for TimeMachine."""

from __future__ import annotations

import unittest

import torch

from tsflab.models._components.mamba import MambaBlock
from tsflab.models.timemachine.model import Model


def make(**kwargs) -> Model:
    options = dict(n1=8, n2=4, d_state=4, d_conv=2, dropout=0.0)
    options.update(kwargs)
    return Model(12, 5, 3, **options)


class TimeMachineTests(unittest.TestCase):
    def test_four_kernel_free_mambas_with_axis_widths(self) -> None:
        """Channel independence gives one width-1 Mamba in each pair; mixing does not."""
        independent = make(ch_ind=True)
        mixing = make(ch_ind=False)
        for model in (independent, mixing):
            for name in ("mamba1", "mamba2", "mamba3", "mamba4"):
                self.assertIsInstance(getattr(model, name), MambaBlock)
        widths = lambda m: [getattr(m, n).in_proj.in_features for n in ("mamba1", "mamba2", "mamba3", "mamba4")]
        self.assertEqual(widths(independent), [1, 4, 8, 1])
        self.assertEqual(widths(mixing), [4, 4, 8, 8])

    def test_mambas_use_reference_dt_initialisation(self) -> None:
        model = make(ch_ind=False)
        for name in ("mamba1", "mamba2", "mamba3", "mamba4"):
            block = getattr(model, name)
            step = torch.nn.functional.softplus(block.dt_proj.bias)
            self.assertGreaterEqual(float(step.min()), 1e-4 - 1e-7)
            self.assertLessEqual(float(step.max()), 0.1 + 1e-6)
            self.assertLessEqual(float(block.dt_proj.weight.abs().max()), block.dt_rank**-0.5 + 1e-6)

    def test_forward_matches_stage_by_stage_equations(self) -> None:
        """y = P2( (P1(x(2)-branch) + x(1)) || (mamba3 + mamba4 branches) ) for both layouts."""
        torch.manual_seed(0)
        for ch_ind in (True, False):
            model = make(ch_ind=ch_ind, revin=False).eval()
            x = torch.randn(2, 12, 3) * 2 + 1
            mean = x.mean(1, keepdim=True)
            std = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            tokens = ((x - mean) / std).transpose(1, 2)
            if ch_ind:
                tokens = tokens.reshape(6, 1, 12)
            swap = (lambda t: t.transpose(1, 2)) if ch_ind else (lambda t: t)
            e1 = model.embed1(tokens)
            outer = swap(model.mamba4(swap(e1))) + model.mamba3(e1)
            e2 = model.embed2(e1)
            inner = swap(model.mamba1(swap(e2))) + model.mamba2(e2) + e2
            fused = model.proj1(inner) + e1
            y = model.proj2(torch.cat([fused, outer], -1))
            if ch_ind:
                y = y.reshape(2, 3, 5)
            expected = y.transpose(1, 2) * std + mean
            torch.testing.assert_close(model(x), expected, atol=1e-5, rtol=1e-5)

    def test_residual_flag_removes_both_skip_paths(self) -> None:
        torch.manual_seed(1)
        with_skip, without = make(residual=True).eval(), make(residual=False).eval()
        without.load_state_dict(with_skip.state_dict())
        x = torch.randn(2, 12, 3)
        self.assertFalse(torch.allclose(with_skip(x), without(x)))

    def test_channel_independence_versus_mixing(self) -> None:
        torch.manual_seed(2)
        x = torch.randn(2, 12, 3)
        perturbed = x.clone()
        perturbed[..., 0] += torch.randn(2, 12)  # scans are causal over channel tokens
        independent = make(ch_ind=True).eval()
        torch.testing.assert_close(independent(x)[..., 2], independent(perturbed)[..., 2])
        mixing = make(ch_ind=False).eval()
        self.assertFalse(torch.allclose(mixing(x)[..., 2], mixing(perturbed)[..., 2]))

    def test_revin_flag_controls_affine_parameters(self) -> None:
        self.assertTrue(hasattr(make(revin=True).norm, "affine_weight"))
        self.assertFalse(hasattr(make(revin=False).norm, "affine_weight"))

    def test_gradients_reach_every_parameter(self) -> None:
        for ch_ind in (True, False):
            model = make(ch_ind=ch_ind)
            model(torch.randn(2, 12, 3)).square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
