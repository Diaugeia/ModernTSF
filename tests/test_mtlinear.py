"""Paper-equation and reference checks for MTLinear."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from tsflab.models.mtlinear.model import LAYER_TYPES, Model
from tsflab.models.mtlinear.spec import training_objective, training_setup


def correlated_series(steps: int = 200) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(steps, generator=g), torch.randn(steps, generator=g)
    noise = 0.05 * torch.randn(steps, 5, generator=g)
    return torch.stack([a, a, -a, b, b], 1) + noise


class MTLinearTests(unittest.TestCase):
    def test_each_group_has_its_own_shared_forecaster(self) -> None:
        """Sec. 4.2/4.4: variates of one group share weights; groups do not."""
        model = Model(16, 4, 4, max_clusters=2, kernel_size=3).eval()
        model.group_of.copy_(torch.tensor([0, 1, 0, 1]))
        x = torch.randn(3, 16, 4)
        out = model(x)
        for group in (0, 1):
            cols = (model.group_of == group).nonzero().flatten()
            series = x[:, :, cols].permute(0, 2, 1).reshape(-1, 16, 1)
            alone = model.heads[group](series).reshape(3, 2, 4).permute(0, 2, 1)
            torch.testing.assert_close(out[:, :, cols], alone)
        # swapping a variate between groups changes its forecast (separate weights)
        model.group_of.copy_(torch.tensor([1, 0, 0, 1]))
        self.assertFalse(torch.allclose(model(x)[:, :, 0], out[:, :, 0]))

    def test_layer_types_run_and_are_finite(self) -> None:
        for layer_type in LAYER_TYPES:
            model = Model(24, 6, 3, layer_type=layer_type, kernel_size=5, max_clusters=2)
            out = model(torch.randn(2, 24, 3) + 5.0)
            self.assertEqual(out.shape, (2, 6, 3))
            self.assertTrue(torch.isfinite(out).all())

    def test_nlinear_head_is_level_equivariant(self) -> None:
        model = Model(12, 3, 2, layer_type="NLinear", max_clusters=1).eval()
        x = torch.randn(2, 12, 2)
        torch.testing.assert_close(model(x + 7.0), model(x) + 7.0, atol=1e-5, rtol=0)

    def test_clustering_matches_complete_linkage_reference(self) -> None:
        """Official grouping is complete-linkage agglomeration on 1 - corr."""
        series = correlated_series()
        corr = np.corrcoef(series.numpy().T)
        for threshold in (0.1, 0.5, 1.5):
            model = Model(8, 2, 5, cluster_dist=threshold, max_clusters=5)
            model.fit_clusters(series)
            reference = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold, metric="precomputed",
                linkage="complete",
            ).fit(1.0 - corr).labels_
            ours = model.group_of.numpy()
            same = ours[:, None] == ours[None]
            ref_same = reference[:, None] == reference[None]
            np.testing.assert_array_equal(same, ref_same)
        model = Model(8, 2, 5, cluster_dist=0.5, max_clusters=5)
        model.fit_clusters(series)
        self.assertEqual(model.group_of.tolist(), [0, 0, 1, 2, 2])  # signed: -a apart
        self.assertTrue(bool(model.clusters_fitted))

    def test_cluster_cap_and_extremes(self) -> None:
        series = correlated_series()
        model = Model(8, 2, 5, cluster_dist=0.0, max_clusters=5)
        model.fit_clusters(series)
        self.assertEqual(model.group_of.unique().numel(), 5)
        capped = Model(8, 2, 5, cluster_dist=0.0, max_clusters=2)
        capped.fit_clusters(series)
        self.assertEqual(capped.group_of.unique().numel(), 2)
        one = Model(8, 2, 5, cluster_dist=2.5, max_clusters=5)
        one.fit_clusters(series)
        self.assertEqual(one.group_of.unique().numel(), 1)

    def test_penalty_weights_follow_eq_7_8(self) -> None:
        """w_ij = (K_j H_i)^-a, K_j/H_i the group-mean errors of horizon j / variate i."""
        model = Model(8, 3, 4, max_clusters=2, penalty_param=2.0)
        model.group_of.copy_(torch.tensor([0, 0, 1, 1]))
        error = torch.rand(5, 3, 4) + 0.1
        weights = model.penalty_weights(error)
        for group, cols in enumerate(([0, 1], [2, 3])):
            sub = error[:, :, cols]
            for j in range(3):
                for k, col in enumerate(cols):
                    expected = (sub[:, j, :].mean() * sub[:, :, k].mean()) ** -2
                    torch.testing.assert_close(weights[j, col], expected)

    def test_penalized_loss_is_weighted_squared_error_and_detached(self) -> None:
        model = Model(8, 3, 4, max_clusters=2, penalty_param=1.0)
        model.group_of.copy_(torch.tensor([0, 0, 1, 1]))
        pred = torch.randn(5, 3, 4, requires_grad=True)
        target = torch.randn(5, 3, 4)
        loss = model.penalized_loss(pred, target)
        weights = model.penalty_weights((pred - target).abs().detach())
        torch.testing.assert_close(loss, (weights * (pred - target).pow(2).mean(0)).sum())
        loss.backward()
        expected = 2 * weights * (pred - target) / 5
        torch.testing.assert_close(pred.grad, expected.expand_as(pred))  # weights are constants

    def test_single_axis_penalties_and_plain_mse(self) -> None:
        pred, target = torch.randn(4, 3, 2), torch.randn(4, 3, 2)
        plain = Model(8, 3, 2, use_horizon_penalty=False, use_variates_penalty=False)
        torch.testing.assert_close(plain.penalized_loss(pred, target), (pred - target).pow(2).mean())
        only_v = Model(8, 3, 2, max_clusters=1, use_horizon_penalty=False, penalty_param=1.0)
        e_v = (pred - target).abs().mean((0, 1))
        expected = ((pred - target).pow(2).mean((0, 1)) / e_v).sum()
        torch.testing.assert_close(only_v.penalized_loss(pred, target), expected)

    def test_training_setup_uses_full_series_when_dataset_has_one(self) -> None:
        series = correlated_series(60)
        seen = []

        class Loader:
            dataset = type("D", (), {"data": series.numpy()})()

            def __iter__(self):
                raise AssertionError("must not iterate when the dataset exposes the series")

        model = Model(8, 2, 5, max_clusters=5)
        original = model.fit_clusters
        model.fit_clusters = lambda x: (seen.append(x), original(x))[1]
        training_setup(model, Loader(), pred_len=2, features="M")
        self.assertEqual(tuple(seen[0].shape), (60, 5))
        self.assertTrue(bool(model.clusters_fitted))

    def test_training_hooks_fit_clusters_and_return_forecast(self) -> None:
        series = correlated_series(120)
        windows = [(series[i:i + 8][None], series[i + 8:i + 10][None]) for i in range(100)]
        model = Model(8, 2, 5, max_clusters=5)
        training_setup(model, windows, pred_len=2, features="M")
        self.assertTrue(bool(model.clusters_fitted))

        class Batch:
            x = series[:8][None].repeat(2, 1, 1)
            target = series[8:10][None].repeat(2, 1, 1)

            @staticmethod
            def forecast(m):
                return m(Batch.x)

            @staticmethod
            def align(outputs):
                return outputs

        forecast, loss = training_objective(model, Batch, torch.nn.functional.l1_loss)
        self.assertEqual(forecast.shape, (2, 2, 5))
        loss.backward()

    def test_state_dict_round_trip_keeps_grouping(self) -> None:
        model = Model(8, 2, 5, max_clusters=5)
        model.fit_clusters(correlated_series())
        clone = Model(8, 2, 5, max_clusters=5)
        clone.load_state_dict(model.state_dict())
        self.assertEqual(clone.group_of.tolist(), model.group_of.tolist())
        x = torch.randn(2, 8, 5)
        torch.testing.assert_close(clone(x), model(x))


if __name__ == "__main__":
    unittest.main()
