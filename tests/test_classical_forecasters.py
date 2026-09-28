"""Equation and runtime-contract tests for local classical baselines."""

from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models.autoregressive_ts.model import Model as AutoRegressive
from moderntsf.models.exp_smoothing_ts.model import Model as ExpSmoothing
from moderntsf.models.knn_forecaster_ts.model import Model as SoftKNN
from moderntsf.models.lasso_regression_ts.model import Model as Lasso
from moderntsf.models.polynomial_regression_ts.model import Model as Polynomial
from moderntsf.models.ridge_regression_ts.model import Model as Ridge


CASES = {
    "AutoRegressiveTS": lambda length, horizon, channels: AutoRegressive(
        length, horizon, channels
    ),
    "ExpSmoothingTS": lambda length, horizon, channels: ExpSmoothing(
        length, horizon, channels, initial_alpha=0.4
    ),
    "RidgeRegressionTS": lambda length, horizon, channels: Ridge(
        length, horizon, channels, l2_penalty=0.2
    ),
    "LassoRegressionTS": lambda length, horizon, channels: Lasso(
        length, horizon, channels, l1_penalty=0.2
    ),
    "PolynomialRegressionTS": lambda length, horizon, channels: Polynomial(
        length, horizon, channels, degree=2
    ),
    "KNNForecasterTS": lambda length, horizon, channels: SoftKNN(
        length, horizon, channels, num_prototypes=4, kernel_gamma=0.7
    ),
}


class ClassicalRewriteEquationTests(unittest.TestCase):
    def test_autoregression_is_direct_lag_projection(self) -> None:
        model = AutoRegressive(3, 2, 1)
        with torch.no_grad():
            model.projection.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
            model.projection.bias.copy_(torch.tensor([0.5, -0.5]))
        x = torch.tensor([[[1.0], [2.0], [4.0]]])
        expected = torch.tensor([[[17.5], [2.5]]])
        torch.testing.assert_close(model(x), expected)

    def test_regularized_models_expose_exact_weight_penalty(self) -> None:
        ridge = Ridge(2, 1, 1, l2_penalty=0.25)
        lasso = Lasso(2, 1, 1, l1_penalty=0.5)
        with torch.no_grad():
            ridge.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
            lasso.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
        x = torch.ones(1, 2, 1)
        ridge(x)
        lasso(x)
        torch.testing.assert_close(ridge.aux_loss, torch.tensor(1.25))
        torch.testing.assert_close(lasso.aux_loss, torch.tensor(1.5))

    def test_polynomial_features_are_integer_lag_powers(self) -> None:
        model = Polynomial(2, 1, 1, degree=3)
        features = model.polynomial_features(torch.tensor([[[2.0], [-3.0]]]))
        expected = torch.tensor([[[2.0, -3.0, 4.0, 9.0, 8.0, -27.0]]])
        torch.testing.assert_close(features, expected)

    def test_simple_exponential_smoothing_recurrence(self) -> None:
        model = ExpSmoothing(3, 2, 1, initial_alpha=0.5)
        x = torch.tensor([[[2.0], [4.0], [8.0]]])
        # level_1=3 and level_2=5.5; simple smoothing repeats that level.
        torch.testing.assert_close(model(x), torch.tensor([[[5.5], [5.5]]]))

    def test_soft_knn_is_distance_weighted_reference_future(self) -> None:
        model = SoftKNN(1, 1, 1, num_prototypes=2, kernel_gamma=1.0)
        with torch.no_grad():
            model.reference_windows.copy_(torch.tensor([[[0.0]], [[2.0]]]))
            model.reference_futures.copy_(torch.tensor([[[10.0]], [[20.0]]]))
        x = torch.tensor([[[0.0]]])
        weights = torch.softmax(torch.tensor([0.0, -4.0]), dim=0)
        expected = (weights[0] * 10.0 + weights[1] * 20.0).reshape(1, 1, 1)
        torch.testing.assert_close(model(x), expected)


class ClassicalRewriteContractTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(731)
        for name, factory in CASES.items():
            with self.subTest(model=name):
                model = factory(4, 3, 2).cpu()
                x = torch.randn(2, 4, 2, requires_grad=True)
                marks = torch.randn(2, 4, 3)
                adjacency = torch.eye(2)
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


if __name__ == "__main__":
    unittest.main()
