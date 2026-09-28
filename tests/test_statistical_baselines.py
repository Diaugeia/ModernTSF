"""Equation and runtime tests for local statistical baselines."""

from __future__ import annotations

import copy
import math
import unittest

import torch

from moderntsf.models.arima_ts.model import Model as ARIMA
from moderntsf.models.bayesian_ridge_ts.model import Model as BayesianRidge
from moderntsf.models.elastic_net_ts.model import Model as ElasticNet
from moderntsf.models.gaussian_process_ts.model import Model as GaussianProcess
from moderntsf.models.kalman_filter_ts.model import Model as AlphaBeta
from moderntsf.models.svr_forecaster_ts.model import Model as EpsilonRBF


CASES = {
    "BayesianRidgeTS": lambda length, horizon, channels: BayesianRidge(length, horizon, channels, 0.2),
    "ElasticNetTS": lambda length, horizon, channels: ElasticNet(length, horizon, channels, 0.2, 0.4),
    "KalmanFilterTS": lambda length, horizon, channels: AlphaBeta(length, horizon, channels, 0.5, 0.25),
    "GaussianProcessTS": lambda length, horizon, channels: GaussianProcess(length, horizon, channels, 4, 1.0, 0.1),
    "SVRForecasterTS": lambda length, horizon, channels: EpsilonRBF(length, horizon, channels, 4, 0.7, 0.1, 0.2),
    "ARIMATS": lambda length, horizon, channels: ARIMA(length, horizon, channels, 2, 1),
}


class StatisticalRewriteEquationTests(unittest.TestCase):
    def test_bayesian_ridge_uses_gaussian_weight_prior(self) -> None:
        model = BayesianRidge(2, 1, 1, 2.0)
        with torch.no_grad():
            model.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model(torch.ones(1, 2, 1))
        expected = 5.0 - math.log(2.0)
        torch.testing.assert_close(model.aux_loss, torch.tensor(expected), atol=2e-7, rtol=1e-6)

    def test_elastic_net_penalty_combines_l1_and_l2(self) -> None:
        model = ElasticNet(2, 1, 1, penalty=0.5, l1_ratio=0.25)
        with torch.no_grad():
            model.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model(torch.ones(1, 2, 1))
        expected = 0.5 * (0.25 * 3.0 + 0.5 * 0.75 * 5.0)
        torch.testing.assert_close(model.aux_loss, torch.tensor(expected))

    def test_alpha_beta_predict_update_recurrence(self) -> None:
        model = AlphaBeta(3, 2, 1, initial_alpha=0.5, initial_beta=0.25)
        actual = model(torch.tensor([[[0.0], [2.0], [4.0]]]))
        torch.testing.assert_close(actual, torch.tensor([[[3.875], [5.0]]]))

    def test_sparse_gp_is_kernel_linear_solve(self) -> None:
        model = GaussianProcess(1, 1, 1, num_inducing=2, length_scale=1.0, noise=0.5)
        with torch.no_grad():
            model.inducing_inputs.copy_(torch.tensor([[0.0], [2.0]]))
            model.inducing_targets.copy_(torch.tensor([[10.0], [20.0]]))
        query = torch.tensor([[[0.0]]])
        z = torch.tensor([[0.0], [2.0]])
        k_xz = torch.exp(-0.5 * torch.cdist(torch.tensor([[0.0]]), z).square())
        k_zz = torch.exp(-0.5 * torch.cdist(z, z).square())
        expected = k_xz @ torch.linalg.solve(k_zz + model.noise.detach() * torch.eye(2), torch.tensor([[10.0], [20.0]]))
        torch.testing.assert_close(model(query), expected.reshape(1, 1, 1))

    def test_svr_rbf_expansion_and_epsilon_loss(self) -> None:
        model = EpsilonRBF(1, 1, 1, num_support=2, kernel_gamma=1.0, epsilon=0.5)
        with torch.no_grad():
            model.support_centres.copy_(torch.tensor([[0.0], [2.0]]))
            model.coefficients.copy_(torch.tensor([[10.0], [20.0]]))
            model.bias.zero_()
        expected = torch.tensor(10.0 + 20.0 * math.exp(-4.0)).reshape(1, 1, 1)
        actual = model(torch.tensor([[[0.0]]]))
        torch.testing.assert_close(actual, expected)
        loss = model.epsilon_insensitive_loss(torch.tensor([0.0, 2.0]), torch.tensor([0.25, 0.0]))
        torch.testing.assert_close(loss, torch.tensor(0.75))

    def test_arima_conditional_recurrence(self) -> None:
        model = ARIMA(3, 2, 1, ar_order=1, ma_order=1)
        with torch.no_grad():
            model.ar_coefficients.copy_(torch.tensor([0.5]))
            model.ma_coefficients.copy_(torch.tensor([0.25]))
        actual = model(torch.tensor([[[1.0], [3.0], [6.0]]]))
        torch.testing.assert_close(actual, torch.tensor([[[7.875], [8.8125]]]))


class StatisticalRewriteContractTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(947)
        for name, factory in CASES.items():
            with self.subTest(model=name):
                model = factory(4, 3, 2).cpu()
                x = torch.randn(2, 4, 2, requires_grad=True)
                marks, adjacency = torch.randn(2, 4, 3), torch.eye(2)
                output = model(x, marks, adjacency)
                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())
                objective = output.square().mean()
                if model.aux_loss is not None:
                    objective = objective + model.aux_loss
                objective.backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)

                clone = factory(4, 3, 2).cpu()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach()), output.detach())
                self.assertEqual(tuple(model(torch.randn(1, 4, 2)).shape), (1, 3, 2))
                boundary = factory(1, 1, 2).cpu()
                self.assertEqual(tuple(boundary(torch.randn(1, 1, 2)).shape), (1, 1, 2))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 3, 2))
                torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))


if __name__ == "__main__":
    unittest.main()
