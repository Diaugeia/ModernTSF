"""PGN/TPGN: paper-equation structure checks for the local implementation."""

from __future__ import annotations

import unittest

import torch

from tsflab.models.pgn.model import Model, ParallelGatedNetwork, VariateLinear


class ParallelGatedNetworkTests(unittest.TestCase):
    def test_history_aggregates_only_strictly_earlier_rows(self) -> None:
        torch.manual_seed(0)
        rows, cols, variates, features, hidden = 4, 2, 3, 5, 6
        net = ParallelGatedNetwork(rows, variates, features, hidden)
        x = torch.randn(2, rows, cols, variates, features)
        window = rows - 1
        expected = torch.zeros(2, rows, cols, variates, hidden)
        for r in range(rows):
            expected[:, r] = net.hie_bias
            for j in range(window):
                source = r + j - window  # rows r-window .. r-1, zero before the series
                if source < 0:
                    continue
                expected[:, r] = expected[:, r] + torch.einsum(
                    "bpvf,vof->bpvo", x[:, source], net.hie_weight[..., j]
                )
        torch.testing.assert_close(net.history(x), expected, atol=1e-5, rtol=1e-5)

    def test_gate_mixes_history_and_candidate(self) -> None:
        torch.manual_seed(1)
        net = ParallelGatedNetwork(3, 2, 4, 5)
        x = torch.randn(2, 3, 2, 2, 4)
        hist = net.history(x)
        gates = net.gate(torch.cat([x, hist], dim=-1))
        update = torch.sigmoid(gates[..., :5])
        candidate = torch.tanh(gates[..., 5:])
        torch.testing.assert_close(net(x), hist * update + (1 - update) * candidate)

    def test_each_row_ignores_later_rows(self) -> None:
        torch.manual_seed(2)
        net = ParallelGatedNetwork(5, 2, 3, 4)
        x = torch.randn(1, 5, 2, 2, 3)
        changed = x.clone()
        changed[:, 3:] += torch.randn_like(changed[:, 3:])
        torch.testing.assert_close(net(x)[:, :3], net(changed)[:, :3])
        self.assertFalse(torch.allclose(net(x)[:, 3:], net(changed)[:, 3:]))

    def test_rejects_a_single_row(self) -> None:
        with self.assertRaises(ValueError):
            ParallelGatedNetwork(1, 2, 3, 4)


class VariateLinearTests(unittest.TestCase):
    def test_variates_use_independent_parameters(self) -> None:
        layer = VariateLinear(3, 4, 2)
        x = torch.randn(5, 3, 4)
        out = layer(x)
        for v in range(3):
            expected = x[:, v] @ layer.weight[v].T + layer.bias[v]
            torch.testing.assert_close(out[:, v], expected, atol=1e-6, rtol=1e-5)


class TPGNTests(unittest.TestCase):
    def test_forward_shape_and_missing_marks_equal_zero_marks(self) -> None:
        torch.manual_seed(3)
        model = Model(seq_len=48, pred_len=24, enc_in=3, period=12, d_model=8).eval()
        x = torch.randn(2, 48, 3)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 24, 3))
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, model(x, torch.zeros(2, 48, 4)))

    def test_calendar_features_enter_the_forecast(self) -> None:
        torch.manual_seed(4)
        model = Model(seq_len=48, pred_len=24, enc_in=2, period=12, d_model=8).eval()
        x = torch.randn(2, 48, 2)
        self.assertFalse(torch.allclose(model(x), model(x, torch.rand(2, 48, 4))))

    def test_branch_head_widths_and_ablation(self) -> None:
        both = Model(seq_len=24, pred_len=24, enc_in=2, period=6, d_model=8)
        long_only = Model(seq_len=24, pred_len=24, enc_in=2, period=6, d_model=8, use_short_branch=False)
        self.assertEqual(both.head.weight.shape, (2, 4, 16))  # variates, pred rows, 2 * d_model
        self.assertEqual(long_only.head.weight.shape, (2, 4, 8))
        self.assertFalse(hasattr(long_only, "short_row"))
        out = long_only(torch.randn(3, 24, 2))
        self.assertEqual(tuple(out.shape), (3, 24, 2))

    def test_gradients_reach_every_parameter_group(self) -> None:
        torch.manual_seed(5)
        model = Model(seq_len=24, pred_len=12, enc_in=2, period=6, d_model=8)
        model(torch.randn(3, 24, 2), torch.rand(3, 24, 4)).square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_rejects_lengths_that_are_not_period_multiples(self) -> None:
        with self.assertRaises(ValueError):
            Model(seq_len=50, pred_len=24, enc_in=2, period=12)
        with self.assertRaises(ValueError):
            Model(seq_len=48, pred_len=20, enc_in=2, period=12)
        with self.assertRaises(ValueError):
            Model(seq_len=12, pred_len=12, enc_in=2, period=12)


if __name__ == "__main__":
    unittest.main()
