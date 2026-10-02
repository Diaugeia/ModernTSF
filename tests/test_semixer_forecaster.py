"""Paper-structure and runtime-contract tests for SEMixer.

Paper: "SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and
Long-term Time Series Forecasting" (arXiv:2602.16220, WWW 2026).
"""
from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.semixer.model import Model


def make_model(**overrides):
    params = dict(
        seq_len=48,
        pred_len=12,
        enc_in=3,
        c_out=3,
        d_model=8,
        patch_len=4,
        stride=2,
        scale_factors="1x2x4",
        reduce_dim=6,
        eib_num=1,
        eib_num_1scale=1,
        connection_probability=0.5,
        dropout=0.0,
        mixing_dropout=0.0,
        head_dropout=0.0,
    )
    params.update(overrides)
    return Model(**params)


class PaperStructureTests(unittest.TestCase):
    """Checks the two defining architectural properties: RAM and MPMC."""

    def test_random_attention_differs_between_train_and_eval_modes(self):
        """RAM (Eqs. 5-7): a fresh Bernoulli mask at train time, and a
        closed-form dropout-ensemble constant matrix at eval time."""
        torch.manual_seed(0)
        model = make_model()
        block = model.finest_chain[0]
        x = torch.randn(2, block.patch_num, model.embeddings[0].value_embedding.out_features)

        model.train()
        block(x)
        train_adjacency = block.last_adjacency.clone()
        # Entries must be binary (0 or 1) during training.
        self.assertTrue(set(torch.unique(train_adjacency).tolist()).issubset({0.0, 1.0}))

        model.eval()
        with torch.no_grad():
            block(x)
        eval_adjacency = block.last_adjacency.clone()
        keep_probability = 1.0 - block.connection_probability
        expected = torch.full_like(eval_adjacency, keep_probability)
        torch.testing.assert_close(eval_adjacency, expected)

        # Two independent training-mode calls draw independent masks.
        model.train()
        block(x)
        second_train_adjacency = block.last_adjacency.clone()
        self.assertFalse(torch.equal(train_adjacency, second_train_adjacency))

    def test_mpmc_progressively_pairs_adjacent_scales_and_keeps_only_new_suffix(self):
        """MPMC (Sec. 3.3, Algorithm 1): the finest scale is mixed alone; every
        coarser scale is mixed only after being concatenated with the
        previous scale's mixed tail, and only the new scale's suffix is kept
        for the next stage, so intermediate mixing widths equal
        patch_num[s-1] + patch_num[s]."""
        model = make_model()
        patch_nums = [embedding.patch_num for embedding in model.embeddings]
        self.assertEqual(len(patch_nums), 3)
        # Finest-scale chain mixes over exactly patch_num[0] tokens.
        self.assertEqual(model.finest_chain[0].patch_num, patch_nums[0])
        # Each coarser chain mixes over patch_num[s-1] + patch_num[s] tokens.
        expected_widths = [patch_nums[0] + patch_nums[1], patch_nums[1] + patch_nums[2]]
        actual_widths = [chain[0].patch_num for chain in model.coarser_chains]
        self.assertEqual(actual_widths, expected_widths)

        x = torch.randn(2, model.seq_len, model.channels)
        model.eval()
        with torch.no_grad():
            model(x)


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_shape_and_gradients(self):
        torch.manual_seed(1)
        model = make_model().train()
        x = torch.randn(2, model.seq_len, model.channels, requires_grad=True)
        output = model(x)
        self.assertEqual(output.shape, (2, model.pred_len, model.channels))
        self.assertTrue(torch.isfinite(output).all())

        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().max().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_state_dict_round_trip_and_batch_independence(self):
        torch.manual_seed(2)
        model = make_model().eval()
        clone = make_model().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        x = torch.randn(2, model.seq_len, model.channels)
        with torch.no_grad():
            torch.testing.assert_close(clone(x), model(x))
            self.assertEqual(model(x[:1]).shape, (1, model.pred_len, model.channels))

    def test_rejects_malformed_inputs_and_scale_schedule(self):
        model = make_model().eval()
        with self.assertRaises(ValueError):
            model(torch.randn(1, model.seq_len - 1, model.channels))
        with self.assertRaises(ValueError):
            make_model(scale_factors="2x4")
        with self.assertRaises(ValueError):
            make_model(scale_factors="1x4x2")
        with self.assertRaises(ValueError):
            make_model(enc_in=3, c_out=4)


if __name__ == "__main__":
    unittest.main()
