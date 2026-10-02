"""Paper-structure and runtime tests for GPHT (KDD 2024)."""
from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.gpht.model import Model as GPHT


def factory(pred_len=8, rates=(4, 2, 1)):
    return GPHT(
        16, pred_len, 3, token_len=8, pooling_rates=rates,
        d_model=8, d_ff=16, n_heads=2, e_layers=1, dropout=0.0,
    )


class PaperStructureTests(unittest.TestCase):
    def test_one_stage_per_pooling_rate_with_token_aligned_patches(self):
        model = factory()
        self.assertEqual(len(model.stages), 3)
        self.assertEqual([s.down_sample.kernel_size for s in model.stages], [4, 2, 1])
        self.assertEqual([s.patch_embedding.patch_len for s in model.stages], [2, 4, 8])
        out = model.stages[0](torch.randn(2, 16, 3))
        self.assertEqual(out.shape, (2, 16, 3))  # one token_len prediction per token

    def test_stage_outputs_are_causal_over_tokens(self):
        model = factory().eval()
        x = torch.randn(2, 16, 3)
        changed = x.clone()
        changed[:, 8:] += 5.0  # perturb only the second token
        a = model.stages[2](x)
        b = model.stages[2](changed)
        torch.testing.assert_close(a[:, :8], b[:, :8])
        self.assertFalse(torch.allclose(a[:, 8:], b[:, 8:]))

    def test_iterative_residual_and_sum_follow_equations_4_to_6(self):
        torch.manual_seed(0)
        model = factory().eval()
        x = torch.randn(2, 16, 3) * 2 + 1
        normalized = (x - x.mean(1, keepdim=True)) / torch.sqrt(
            x.var(1, keepdim=True, unbiased=False) + 1e-5
        )
        residual, total = normalized, torch.zeros_like(normalized)
        for stage in model.stages:
            out = stage(residual)
            total = total + out
            pad = torch.zeros(2, 8, 3)
            residual = residual - torch.cat((pad, out[:, :8]), dim=1)
        expected = total * torch.sqrt(
            x.var(1, keepdim=True, unbiased=False) + 1e-5
        ) + x.mean(1, keepdim=True)
        torch.testing.assert_close(model.next_token_series(x), expected, rtol=1e-4, atol=1e-5)

    def test_forecast_rolls_the_window_one_token_at_a_time(self):
        model = factory(pred_len=12).eval()
        x = torch.randn(2, 16, 3)
        first = model.next_token_series(x)[:, -8:]
        window = torch.cat((x, first), dim=1)[:, -16:]
        second = model.next_token_series(window)[:, -8:]
        forecast = model(x)
        self.assertEqual(forecast.shape, (2, 12, 3))
        torch.testing.assert_close(forecast, torch.cat((first, second), 1)[:, :12])

    def test_objective_is_next_token_mse_over_the_whole_window(self):
        model = factory().eval()
        x, y = torch.randn(2, 16, 3), torch.randn(2, 8, 3)
        series = model.next_token_series(x)
        target = torch.cat((x[:, 8:], y), dim=1)
        torch.testing.assert_close(
            model.training_objective(x, y), ((series - target) ** 2).mean()
        )
        with self.assertRaises(ValueError):
            model.training_objective(x, y[:, :4])

    def test_shape_contract_is_enforced(self):
        with self.assertRaises(ValueError):
            GPHT(15, 8, 3, token_len=8, pooling_rates=(1,), d_model=8, d_ff=8, n_heads=2)
        with self.assertRaises(ValueError):
            GPHT(16, 8, 3, token_len=8, pooling_rates=(3,), d_model=8, d_ff=8, n_heads=2)


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_active_parameters_and_round_trip(self):
        torch.manual_seed(3)
        x = torch.randn(2, 16, 3)
        model = factory().cpu().eval()
        value = x.clone().requires_grad_(True)
        output = model(value)
        self.assertEqual(output.shape, (2, 8, 3))
        self.assertTrue(torch.isfinite(output).all())
        model.train()
        loss = model.training_objective(value, torch.randn(2, 8, 3))
        loss.backward()
        self.assertGreater(value.grad.abs().max().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        clone = factory().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        torch.testing.assert_close(clone(x), model(x))
        self.assertEqual(model(x[:1]).shape, (1, 8, 3))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 15, 3))


if __name__ == "__main__":
    unittest.main()
