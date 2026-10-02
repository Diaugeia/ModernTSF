"""Paper-structure and runtime tests for PENGUIN."""
from __future__ import annotations

import copy
import math
import unittest

import torch

from tsflab.models.penguin.model import Model as PENGUIN
from tsflab.models.penguin.model import PeriodicNestedGroupAttention


def factory(**kwargs):
    args = dict(d_model=8, d_ff=16, n_heads=4, e_layers=2, patch_len=4, stride=2, dropout=0.0, periods=(4, 8))
    args.update(kwargs)
    return PENGUIN(16, 5, 3, **args)


class PaperStructureTests(unittest.TestCase):
    def test_patch_count_includes_the_replicate_padded_patch(self):
        model = factory()
        self.assertEqual(model.num_patches, (16 - 4) // 2 + 2)
        tokens, n_vars = model.patch_embedding(torch.randn(2, 3, 16))
        self.assertEqual(tokens.shape, (6, 8, 8))
        self.assertEqual(n_vars, 3)
        self.assertEqual(model.head.linear.weight.shape, (5, 8 * 8))

    def test_periods_are_converted_to_patch_units(self):
        attention = factory().layers[0].attention
        self.assertEqual(attention.periods, (2, 4))  # (4, 8) / stride 2
        self.assertEqual(attention.groups, 2)
        # one K/V head per group, h query heads
        self.assertEqual(attention.key.weight.shape, (2 * 2, 8))
        self.assertEqual(attention.query.weight.shape, (8, 8))

    def test_group_attention_matches_explicit_per_head_computation(self):
        torch.manual_seed(0)
        attn = PeriodicNestedGroupAttention(8, 4, (2, 3), 0.0, alibi=True, causal=True).eval()
        x = torch.randn(2, 6, 8)
        out = attn(x)
        q = attn.query(x).view(2, 6, 4, 2)
        k = attn.key(x).view(2, 6, 2, 2)
        v = attn.value(x).view(2, 6, 2, 2)
        heads = []
        for h in range(4):
            group, period = h // 2, (2, 3)[h // 2]
            slope = 2.0 ** (-8.0 * (h % 2 + 1) / 2)
            i = torch.arange(6)
            u = (i.view(-1, 1) - i.view(1, -1)).abs() % period
            tri = torch.where(u < period / 2, u, period - u).float()
            score = q[:, :, h] @ k[:, :, group].transpose(1, 2) / math.sqrt(2) - slope * tri
            score = score.masked_fill(torch.ones(6, 6, dtype=torch.bool).triu(1), float("-inf"))
            heads.append(torch.softmax(score, -1) @ v[:, :, group])
        expected = attn.out(torch.cat(heads, dim=-1))
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)

    def test_causal_mask_hides_later_patches_and_full_attention_does_not(self):
        torch.manual_seed(1)
        x = torch.randn(1, 6, 8)
        changed = x.clone()
        changed[:, -1] += 5.0
        causal = PeriodicNestedGroupAttention(8, 4, (2,), 0.0, True, True).eval()
        torch.testing.assert_close(causal(x)[:, :-1], causal(changed)[:, :-1])
        full = PeriodicNestedGroupAttention(8, 4, (2,), 0.0, True, False)
        full.load_state_dict(causal.state_dict())
        self.assertGreater((full.eval()(x)[:, :-1] - full(changed)[:, :-1]).abs().max().item(), 1e-6)

    def test_no_period_uses_plain_alibi_and_alibi_can_be_disabled(self):
        model = factory(periods=())
        self.assertIsNone(model.layers[0].attention.periods)
        self.assertEqual(model.layers[0].attention.groups, 1)
        plain = factory(alibi=False)
        self.assertFalse(plain.layers[0].attention.alibi)

    def test_rmsnorm_option_replaces_layer_norm(self):
        self.assertIsInstance(factory().layers[0].norm1, torch.nn.LayerNorm)
        self.assertEqual(type(factory(use_rmsnorm=True).layers[0].norm1).__name__, "RMSNorm")

    def test_marks_are_ignored(self):
        model = factory().eval()
        x = torch.randn(2, 16, 3)
        torch.testing.assert_close(model(x, torch.randn(2, 16, 6)), model(x))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            factory(periods=(5,))  # not a multiple of stride
        with self.assertRaises(ValueError):
            factory(n_heads=3)
        with self.assertRaises(ValueError):
            factory(n_heads=2, periods=(4, 8, 12))


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_round_trip_and_bounds(self):
        torch.manual_seed(7)
        x = torch.randn(4, 16, 3)
        model = factory().cpu().eval()
        value = x.clone().requires_grad_(True)
        output = model(value)
        self.assertEqual(output.shape, (4, 5, 3))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertGreater(value.grad.abs().max().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        clone = factory().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        torch.testing.assert_close(clone(x), model(x))
        self.assertEqual(model(x[:1]).shape, (1, 5, 3))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 15, 3))


if __name__ == "__main__":
    unittest.main()
