"""Paper-structure and runtime tests for PatchTSMixer."""
from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from tsflab.models.patchtsmixer.model import Model as PatchTSMixer


def factory(mode="common_channel", gated_attn=True):
    return PatchTSMixer(
        20, 5, 3, d_model=8, patch_len=6, stride=4, num_layers=2, expansion_factor=2,
        dropout=0, head_dropout=0, mode=mode, gated_attn=gated_attn,
    )


class PaperStructureTests(unittest.TestCase):
    def test_patching_keeps_latest_samples_and_orders_channel_patch_axes(self):
        model = factory()
        self.assertEqual(model.num_patches, 4)  # floor((20 - 6) / 4) + 1
        self.assertEqual(model.start, 2)  # the oldest 2 samples are dropped
        x = torch.randn(2, 20, 3)
        patches = model.patchify(x)
        self.assertEqual(patches.shape, (2, 3, 4, 6))
        torch.testing.assert_close(patches[1, 2, 3], x[1, 2 + 3 * 4 : 2 + 3 * 4 + 6, 2])

    def test_each_mixer_has_gated_attention_after_its_mlp(self):
        model = factory()
        layer = model.layers[0]
        self.assertIsNone(layer.channel_mixer)
        x = torch.randn(2, 3, 4, 8)
        sub = layer.patch_mixer
        h = sub.mlp(sub.norm(x).transpose(2, 3))
        expected = x + sub.gate(h).transpose(2, 3)
        torch.testing.assert_close(sub(x), expected)
        self.assertEqual(sub.mlp.fc1.weight.shape, (8, 4))  # n x (n * ef)
        self.assertEqual(layer.feature_mixer.mlp.fc1.weight.shape, (16, 8))
        self.assertEqual(sub.gate.score.weight.shape, (4, 4))
        self.assertIsNone(factory(gated_attn=False).layers[0].patch_mixer.gate)

    def test_mix_channel_adds_inter_channel_mixer_before_patch_mixer(self):
        model = factory("mix_channel")
        layer = model.layers[0]
        self.assertEqual(layer.channel_mixer.mlp.fc1.weight.shape, (6, 3))  # c x (c * ef)
        x = torch.randn(2, 3, 4, 8)
        sub = layer.channel_mixer
        h = sub.mlp(sub.norm(x).permute(0, 3, 2, 1))
        torch.testing.assert_close(sub(x), x + sub.gate(h).permute(0, 3, 2, 1))
        torch.testing.assert_close(layer(x), layer.feature_mixer(layer.patch_mixer(sub(x))))
        # channels interact only in mix_channel mode
        a, b = torch.randn(1, 20, 3), torch.randn(1, 20, 3)
        b[..., 1:] = a[..., 1:]
        common = factory().eval()
        diff = (common(a) - common(b)).abs()
        self.assertLess(diff[..., 1:].max().item(), 1e-5)
        mixed = factory("mix_channel").eval()
        self.assertGreater((mixed(a) - mixed(b)).abs()[..., 1:].max().item(), 1e-6)

    def test_head_flattens_patches_then_features_with_one_shared_linear(self):
        model = factory().eval()
        h = torch.randn(2, 3, 4, 8)
        expected = F.linear(h.reshape(2, 3, 32), model.head.linear.weight, model.head.linear.bias)
        torch.testing.assert_close(model.head(h), expected)
        self.assertEqual(model.head.linear.weight.shape, (5, 32))

    def test_revin_is_non_affine_and_restores_scale(self):
        model = factory().eval()
        self.assertEqual([n for n, _ in model.revin.named_parameters()], [])
        x = torch.randn(2, 20, 3)
        shifted = model(x * 5 + 100)
        torch.testing.assert_close(shifted, model(x) * 5 + 100, atol=1e-3, rtol=1e-4)

    def test_marks_are_ignored(self):
        model = factory().eval()
        x = torch.randn(2, 20, 3)
        torch.testing.assert_close(model(x, torch.randn(2, 20, 6)), model(x))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            PatchTSMixer(6, 2, 3, patch_len=6)
        with self.assertRaises(ValueError):
            PatchTSMixer(20, 2, 3, mode="flatten")


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_active_parameters_and_round_trip(self):
        for mode in ("common_channel", "mix_channel"):
            with self.subTest(mode=mode):
                torch.manual_seed(7)
                x = torch.randn(2, 20, 3)
                model = factory(mode).cpu().eval()
                value = x.clone().requires_grad_(True)
                output = model(value)
                self.assertEqual(output.shape, (2, 5, 3))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertGreater(value.grad.abs().max().item(), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                clone = factory(mode).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(clone(x), model(x))
                self.assertEqual(model(x[:1]).shape, (1, 5, 3))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 19, 3))


if __name__ == "__main__":
    unittest.main()
