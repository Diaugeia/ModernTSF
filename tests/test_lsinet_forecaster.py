"""Paper-structure and runtime tests for LSINet."""
from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.lsinet.model import Model
from tsflab.models._components.sparse_connection_router import SharedSparseConnectionRouter


def make_model(**overrides):
    params = dict(
        seq_len=12,
        pred_len=3,
        enc_in=4,
        c_out=4,
        d_model=8,
        n_heads=2,
        e_layers=1,
        n_patches=4,
        density=0.3,
        dropout=0.0,
    )
    params.update(overrides)
    return Model(**params)


class PaperStructureTests(unittest.TestCase):
    """LSINet's defining contribution is a shared, sample-independent sparse
    interaction (Multihead Sparse Interaction Mechanism / Shared Interaction
    Learning, Sec. "Shared Connection Matrix in SSCL" and Eqs. 3-6) rather than
    content-dependent self-attention scores."""

    def test_sparse_connection_router_is_input_independent(self):
        model = make_model().eval()
        model(torch.randn(2, 12, 4))
        block = model.blocks[0]
        connections_a, _ = block.router()
        connections_b, _ = block.router()
        # The router never receives the batch input: calling it twice with no
        # input yields exactly the same connection matrix (Shared Interaction
        # Learning: one adjacency shared by every sample and channel).
        torch.testing.assert_close(connections_a, connections_b)

    def test_learned_interaction_matrix_is_binary_and_respects_target_density(self):
        model = make_model(n_patches=4, density=0.25).eval()
        block = model.blocks[0]
        connections, probabilities = block.router()
        self.assertEqual(connections.shape, (block.n_heads, block.router.num_positions, block.router.num_positions))
        uniques = torch.unique(connections)
        self.assertTrue(set(uniques.tolist()).issubset({0.0, 1.0}))
        n = block.router.num_positions
        expected_ones = max(1, round(0.25 * n * n))
        for head in range(connections.shape[0]):
            self.assertEqual(int(connections[head].sum().item()), expected_ones)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_multihead_router_is_a_sparse_connection_router_component(self):
        model = make_model().eval()
        for block in model.blocks:
            self.assertIsInstance(block.router, SharedSparseConnectionRouter)

    def test_patch_geometry_follows_paper_eq9_derivation(self):
        # Eq. 9: stride = floor(seq_len / n_patches), patch_len = 2 * stride.
        model = make_model(seq_len=24, n_patches=6).eval()
        encoder = model.patch_encoder
        self.assertEqual(encoder.stride, 24 // 6)
        self.assertEqual(encoder.patch_len, 2 * encoder.stride)


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_finite_and_shape(self):
        torch.manual_seed(1234)
        model = make_model().eval()
        x = torch.randn(3, 12, 4, requires_grad=True)
        output = model(x)
        self.assertEqual(output.shape, (3, 3, 4))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().max().item(), 0)

    def test_active_gradients_on_all_parameters(self):
        torch.manual_seed(7)
        model = make_model().train()
        x = torch.randn(2, 12, 4)
        output = model(x)
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_state_dict_round_trip_is_deterministic_in_eval(self):
        torch.manual_seed(9)
        model = make_model().eval()
        clone = make_model().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        x = torch.randn(2, 12, 4)
        torch.testing.assert_close(model(x), clone(x))

    def test_batch_and_channel_bounds(self):
        model = make_model().eval()
        single = model(torch.randn(1, 12, 4))
        self.assertEqual(single.shape, (1, 3, 4))
        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 11, 4))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 12, 5))

    def test_ignores_optional_mark_and_decoder_inputs(self):
        model = make_model().eval()
        x = torch.randn(2, 12, 4)
        marks = torch.randn(2, 12, 3)
        dec = torch.randn(2, 3, 4)
        dec_marks = torch.randn(2, 3, 3)
        torch.testing.assert_close(model(x), model(x, marks, dec, dec_marks))

    def test_enc_in_must_match_c_out(self):
        with self.assertRaises(ValueError):
            Model(seq_len=12, pred_len=3, enc_in=4, c_out=5)


if __name__ == "__main__":
    unittest.main()
