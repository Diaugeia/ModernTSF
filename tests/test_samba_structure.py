"""Paper-structure and runtime tests for SAMBA / SDE-Mamba (KDD 2025)."""
from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.samba.model import Model as SAMBA, SimplifiedMambaMixer


def factory(e_layers=1, d_layers=1):
    return SAMBA(
        24, 6, 3, d_model=16, d_ff=16, e_layers=e_layers, d_layers=d_layers,
        patch_len=8, stride=4, dropout=0.0,
    )


class PaperStructureTests(unittest.TestCase):
    def test_patches_are_end_padded_and_flattened_by_the_head(self):
        model = factory()
        self.assertEqual(model.patch_num, (24 - 8) // 4 + 2)
        self.assertEqual(model.head.linear.in_features, 16 * model.patch_num)
        self.assertEqual(model.position.shape, (model.patch_num, 1))

    def test_mixer_has_no_activation_between_convolution_and_ssm(self):
        torch.manual_seed(0)
        mixer = SimplifiedMambaMixer(16, 4).eval()
        x = torch.randn(2, 10, 16)
        signal, gate = mixer.in_proj(x).split([mixer.d_inner, mixer.d_inner], dim=-1)
        conv = mixer.conv1d(signal.transpose(1, 2))[:, :, :10].transpose(1, 2)
        expected = mixer.out_proj(mixer.ssm(conv) * torch.nn.functional.silu(gate))
        torch.testing.assert_close(mixer(x), expected)
        activated = SimplifiedMambaMixer(16, 4, use_act=True)
        activated.load_state_dict(mixer.state_dict())
        self.assertFalse(torch.allclose(activated(x), mixer(x)))

    def test_mixer_is_causal_and_step_size_is_initialized_in_range(self):
        mixer = SimplifiedMambaMixer(16, 4).eval()
        x = torch.randn(1, 10, 16)
        changed = x.clone()
        changed[:, 6:] += 3
        torch.testing.assert_close(mixer(x)[:, :6], mixer(changed)[:, :6])
        step = torch.nn.functional.softplus(mixer.dt_proj.bias)
        self.assertGreaterEqual(step.min().item(), 1e-4 * 0.99)
        self.assertLessEqual(step.max().item(), 0.1 * 1.01)

    def test_time_and_variate_encoders_are_parallel_branches(self):
        model = factory().eval()
        x = torch.randn(2, 24, 3)
        base = model(x)
        # Perturbing the time-branch stack alone changes the output ...
        with torch.no_grad():
            model.encoder_time[0].mlp.fc2.weight.add_(1.0)
        self.assertFalse(torch.allclose(model(x), base))
        # ... and so does perturbing the variate-branch stack alone.
        model = factory().eval()
        base = model(x)
        with torch.no_grad():
            model.encoder_var[0].mlp.fc2.weight.add_(1.0)
        self.assertFalse(torch.allclose(model(x), base))

    def test_variate_branch_mixes_channels_but_time_branch_alone_does_not(self):
        x = torch.randn(2, 24, 3)
        changed = x.clone()
        changed[:, :, 2] += torch.randn(2, 24)
        time_only = factory(1, 0).eval()
        out, out_changed = time_only(x), time_only(changed)
        torch.testing.assert_close(out[:, :, :2], out_changed[:, :, :2], rtol=1e-4, atol=1e-4)
        both = factory(1, 1).eval()
        self.assertFalse(
            torch.allclose(both(x)[:, :, :2], both(changed)[:, :, :2], atol=1e-6)
        )

    def test_single_encoder_variants_and_invalid_depth(self):
        for e, d in ((1, 0), (0, 1)):
            self.assertEqual(factory(e, d)(torch.randn(2, 24, 3)).shape, (2, 6, 3))
        with self.assertRaises(ValueError):
            factory(0, 0)


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_active_parameters_and_round_trip(self):
        torch.manual_seed(5)
        x = torch.randn(2, 24, 3)
        model = factory().cpu().eval()
        value = x.clone().requires_grad_(True)
        output = model(value)
        self.assertEqual(output.shape, (2, 6, 3))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertGreater(value.grad.abs().max().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        clone = factory().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        torch.testing.assert_close(clone(x), model(x))
        self.assertEqual(model(x[:1]).shape, (1, 6, 3))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 23, 3))


if __name__ == "__main__":
    unittest.main()
