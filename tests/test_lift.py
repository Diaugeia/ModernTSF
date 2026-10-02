"""Paper-equation and reference checks for LIFT."""

from __future__ import annotations

import unittest

import torch

from tsflab.models.lift.model import Model


def make(seq_len: int = 64, pred_len: int = 8, channels: int = 3, **kwargs) -> Model:
    return Model(seq_len, pred_len, channels, leader_num=kwargs.pop("leader_num", 1), state_num=2,
                 kernel_size=5, **kwargs)


class LIFTTests(unittest.TestCase):
    def test_chunked_lead_estimation_matches_single_chunk(self) -> None:
        torch.manual_seed(3)
        x = torch.randn(2, 7, 64)
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)
        reference = make(channels=7, leader_num=3, lead_chunk_size=64).estimate_leaders(x)
        for chunk in (1, 2, 3, 7):
            chunked = make(channels=7, leader_num=3, lead_chunk_size=chunk).estimate_leaders(x)
            for expected, actual in zip(reference, chunked):
                torch.testing.assert_close(actual, expected)

    def test_lead_estimator_recovers_circular_lag_and_sign(self) -> None:
        """Eq. (2)-(4): variate 1 repeats variate 0 five steps later, variate 2 is -variate 0."""
        torch.manual_seed(0)
        base = torch.randn(1, 1, 64)
        x = torch.cat([base, torch.roll(base, 5, dims=-1), -torch.roll(base, 9, dims=-1)], 1)
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)
        model = make(leader_num=2)
        leaders, shift, corr = model.estimate_leaders(x)
        # target 1: leader 0 at lag 5, positive correlation of about one
        position = (leaders[0, 1] == 0).nonzero().item()
        self.assertEqual(shift[0, 1, position].item(), 5)
        self.assertAlmostEqual(corr[0, 1, position].item(), 1.0, places=4)
        # target 2: leader 0 at lag 9 with a flipped sign
        position = (leaders[0, 2] == 0).nonzero().item()
        self.assertEqual(shift[0, 2, position].item(), 9)
        self.assertAlmostEqual(corr[0, 2, position].item(), -1.0, places=4)

    def test_cross_correlation_matches_direct_sum(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(1, 3, 32)
        model = make(32, 4, 3, leader_num=3)
        leaders, shift, corr = model.estimate_leaders(x)
        for target in range(3):
            for slot in range(3):
                leader, lag = leaders[0, target, slot].item(), shift[0, target, slot].item()
                direct = (torch.roll(x[0, target], -lag) * x[0, leader]).mean()
                self.assertAlmostEqual(corr[0, target, slot].item(), direct.item(), places=5)

    def test_target_oriented_shift_follows_eq_5_6(self) -> None:
        torch.manual_seed(2)
        model = make(16, 6, 3, leader_num=2)
        x, y = torch.randn(2, 3, 16), torch.randn(2, 3, 6)
        leaders, shift, corr = model.estimate_leaders(x)
        out = model.shift_leaders(x, y, leaders, shift, corr)
        self.assertEqual(out.shape, (2, 3, 2, 6))
        for b in range(2):
            for j in range(3):
                for k in range(2):
                    row = torch.cat([x[b, leaders[b, j, k]], y[b, leaders[b, j, k]]])
                    expected = torch.stack(
                        [row[16 + h - shift[b, j, k]] for h in range(6)]
                    ) * torch.sign(corr[b, j, k])
                    torch.testing.assert_close(out[b, j, k], expected)

    def test_filters_have_2k_plus_one_per_frequency_bin(self) -> None:
        model = make(16, 7, 3, leader_num=2)
        filters = model.lead_filters(torch.randn(2, 3, 16), torch.rand(2, 3, 2) - 0.5)
        self.assertEqual(filters.shape, (2, 3, 5, 7 // 2 + 1))

    def test_refiner_is_residual_on_the_normalized_backbone_forecast(self) -> None:
        """With the mixing map zeroed, the refined output equals the backbone forecast."""
        model = make(24, 6, 3, leader_num=2).eval()
        with torch.no_grad():
            for name in ("weight_re", "weight_im", "bias_re", "bias_im"):
                getattr(model.mix, name).zero_()
        x = torch.randn(2, 24, 3) * 3 + 2
        torch.testing.assert_close(model(x), model.backbone(x), atol=1e-5, rtol=1e-5)

    def test_forward_gradients_reach_refiner_and_backbone(self) -> None:
        model = make(32, 6, 4, leader_num=2)
        out = model(torch.randn(3, 32, 4))
        self.assertEqual(out.shape, (3, 6, 4))
        out.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_leader_count_is_capped_and_state_round_trips(self) -> None:
        model = Model(16, 4, 2, leader_num=4, state_num=3, kernel_size=5)
        self.assertEqual(model.leaders, 2)
        clone = Model(16, 4, 2, leader_num=4, state_num=3, kernel_size=5)
        clone.load_state_dict(model.state_dict())
        x = torch.randn(2, 16, 2)
        torch.testing.assert_close(clone(x), model(x))


if __name__ == "__main__":
    unittest.main()
