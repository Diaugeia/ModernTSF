"""Equation, distinction, and runtime tests for tree baselines."""

from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models._components.soft_tree import SoftDecisionTree, SoftObliviousTree
from moderntsf.models.catboost_ts.model import Model as CatBoost
from moderntsf.models.decision_tree_ts.model import Model as DecisionTree
from moderntsf.models.extra_trees_ts.model import Model as ExtraTrees
from moderntsf.models.gradient_boosting_ts.model import Model as GradientBoosting
from moderntsf.models.lightgbm_ts.model import Model as LightGBM
from moderntsf.models.random_forest_ts.model import Model as RandomForest
from moderntsf.models.xgboost_ts.model import Model as XGBoost


CASES = {
    "DecisionTreeTS": lambda length, horizon, channels: DecisionTree(length, horizon, channels, 2),
    "RandomForestTS": lambda length, horizon, channels: RandomForest(length, horizon, channels, 3, 2),
    "ExtraTreesTS": lambda length, horizon, channels: ExtraTrees(length, horizon, channels, 3, 2),
    "GradientBoostingTS": lambda length, horizon, channels: GradientBoosting(length, horizon, channels, 3, 2),
    "CatBoostTS": lambda length, horizon, channels: CatBoost(length, horizon, channels, 3, 2),
    "LightGBMTS": lambda length, horizon, channels: LightGBM(length, horizon, channels, 3, 2),
    "XGBoostTS": lambda length, horizon, channels: XGBoost(length, horizon, channels, 3, 2),
}


class SoftTreeEquationTests(unittest.TestCase):
    def test_depth_one_tree_is_exact_sigmoid_leaf_interpolation(self) -> None:
        tree = SoftDecisionTree(1, 1, depth=1, temperature=2.0)
        with torch.no_grad():
            tree.split_weight.fill_(3.0)
            tree.threshold.fill_(1.0)
            tree.leaf_value.copy_(torch.tensor([[10.0], [22.0]]))
        x = torch.tensor([[1.0]])
        right = torch.sigmoid(torch.tensor(1.0))
        expected = (1.0 - right) * 10.0 + right * 22.0
        torch.testing.assert_close(tree(x), expected.reshape(1, 1))
        torch.testing.assert_close(tree.leaf_probabilities(x).sum(dim=-1), torch.ones(1))

    def test_oblivious_tree_shares_one_decision_per_depth(self) -> None:
        tree = SoftObliviousTree(2, 1, depth=2, temperature=1.0)
        with torch.no_grad():
            tree.split_weight.copy_(torch.eye(2))
            tree.threshold.zero_()
            tree.leaf_value.copy_(torch.arange(4, dtype=torch.float32).reshape(4, 1))
        x = torch.tensor([[0.0, 0.0]])
        torch.testing.assert_close(tree.leaf_probabilities(x), torch.full((1, 4), 0.25))
        torch.testing.assert_close(tree(x), torch.tensor([[1.5]]))

    def test_named_models_keep_distinct_compositions(self) -> None:
        decision = DecisionTree(4, 2, 2, 2)
        forest = RandomForest(4, 2, 2, 3, 2)
        extra = ExtraTrees(4, 2, 2, 3, 2)
        boosted = GradientBoosting(4, 2, 2, 3, 2)
        cat = CatBoost(4, 2, 2, 3, 2)
        light = LightGBM(4, 2, 2, 3, 2)
        xgb = XGBoost(4, 2, 2, 3, 2)
        self.assertIsInstance(decision.tree, SoftDecisionTree)
        self.assertTrue(any("split_weight" in name for name, _ in forest.named_parameters()))
        self.assertFalse(any("split_weight" in name for name, _ in extra.named_parameters()))
        self.assertEqual(len(boosted.backcasts), 2)
        self.assertTrue(all(isinstance(tree, SoftObliviousTree) for tree in cat.trees))
        self.assertTrue(hasattr(light, "feature_logits"))
        self.assertEqual(tuple(xgb.column_masks.shape), (3, 8))


class TreeRewriteContractTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(1747)
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
                torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))


if __name__ == "__main__":
    unittest.main()
