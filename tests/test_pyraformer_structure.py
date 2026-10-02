"""Equation, structure, and runtime tests for the Pyraformer implementation."""

from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.pyraformer.model import (
    Model,
    PyramidalAttention,
    finest_ancestor_table,
    pyramid_neighbour_table,
    pyramid_sizes,
)


class PyraformerEquationTests(unittest.TestCase):
    def test_equation_two_neighbourhood_has_local_child_and_parent_edges(self) -> None:
        sizes = pyramid_sizes(8, (2, 2))
        self.assertEqual(sizes, (8, 4, 2))
        indices, valid = pyramid_neighbour_table(sizes, (2, 2), 3)

        fine_three = set(indices[3, valid[3]].tolist())
        self.assertEqual(fine_three, {2, 3, 4, 9})
        middle_one = set(indices[9, valid[9]].tolist())
        self.assertEqual(middle_one, {2, 3, 8, 9, 10, 12})

        ancestors = finest_ancestor_table(sizes, (2, 2))
        self.assertEqual(ancestors[7].tolist(), [7, 11, 13])

    def test_equation_three_attention_normalizes_only_over_neighbours(self) -> None:
        indices, valid = pyramid_neighbour_table((2,), (), 1)
        attention = PyramidalAttention(2, 1, indices, valid, dropout=0.0)
        with torch.no_grad():
            attention.query.weight.zero_()
            attention.query.bias.zero_()
            attention.key.weight.zero_()
            attention.key.bias.zero_()
            attention.value.weight.copy_(torch.eye(2))
            attention.value.bias.zero_()
            attention.output.weight.copy_(torch.eye(2))
            attention.output.bias.zero_()
        values = torch.tensor([[[1.0, 2.0], [9.0, 10.0]]])
        torch.testing.assert_close(attention(values), values)


class PyraformerRewriteContractTests(unittest.TestCase):
    @staticmethod
    def factory(seq_len: int = 8) -> Model:
        return Model(
            seq_len=seq_len,
            pred_len=3,
            enc_in=2,
            d_model=8,
            n_heads=2,
            e_layers=2,
            d_ff=16,
            dropout=0.0,
            window_size=(2, 2),
            inner_size=3,
        )

    @staticmethod
    def marks(batch: int, length: int) -> torch.Tensor:
        marks = torch.zeros(batch, length, 6)
        marks[..., 0] = 2024
        marks[..., 1] = 1
        marks[..., 2] = torch.arange(1, length + 1)
        marks[..., 3] = torch.arange(length) % 7
        marks[..., 4] = torch.arange(length) % 24
        return marks

    def test_complete_runtime_and_raw_marks_contract(self) -> None:
        torch.manual_seed(1907)
        model = self.factory().cpu()
        x = torch.randn(2, 8, 2, requires_grad=True)
        marks = self.marks(2, 8)
        output = model(x, marks, torch.eye(2))
        self.assertEqual(tuple(output.shape), (2, 3, 2))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().max().item(), 0.0, name)

        clone = self.factory().cpu()
        clone.load_state_dict(copy.deepcopy(model.state_dict()))
        torch.testing.assert_close(clone(x.detach(), marks), output.detach())
        self.assertEqual(tuple(model(torch.randn(1, 8, 2)).shape), (1, 3, 2))
        self.assertEqual(tuple(self.factory(4)(torch.randn(1, 4, 2)).shape), (1, 3, 2))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 7, 2))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 8, 2), torch.randn(1, 8, 5))

        changed_marks = marks.clone()
        changed_marks[..., 4] += 6
        self.assertFalse(torch.equal(model(x.detach(), marks), model(x.detach(), changed_marks)))

    def test_invalid_pyramid_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.factory(6)
        with self.assertRaises(ValueError):
            Model(8, 3, 2, d_model=8, n_heads=2, window_size=(2, 2), inner_size=2)


if __name__ == "__main__":
    unittest.main()
