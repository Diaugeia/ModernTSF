"""Equation, graph-contract, and runtime tests for six implementations."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from moderntsf.models.astgcn.model import Model as ASTGCN
from moderntsf.models.dcrnn.model import Model as DCRNN
from moderntsf.models.dgcrn.model import Model as DGCRN
from moderntsf.models.dstagnn.model import Model as DSTAGNN
from moderntsf.models.gclstm.model import Model as GCLSTM
from moderntsf.models.gts.model import Model as GTS


def raw_marks(batch: int, steps: int, offset: int = 0) -> torch.Tensor:
    rows = [[2026, 8, 1 + (index // 24), 5, (index + offset) % 24, 0] for index in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def adjacency(nodes: int) -> np.ndarray:
    graph = np.eye(nodes, dtype=np.float32)
    for index in range(nodes - 1):
        graph[index, index + 1] = 1.0
        graph[index + 1, index] = 0.5
    return graph


def factories(length: int = 6, horizon: int = 3, nodes: int = 4, graph: np.ndarray | None = None):
    graph = adjacency(nodes) if graph is None else graph
    return {
        "ASTGCN": lambda: ASTGCN(length, horizon, nodes, graph, cov_dim=2, nb_block=1, K=2, nb_chev_filter=8, nb_time_filter=8),
        "DCRNN": lambda: DCRNN(length, horizon, nodes, graph, input_dim=3, rnn_units=8, num_rnn_layers=1, max_diffusion_step=2),
        "DGCRN": lambda: DGCRN(length, horizon, nodes, graph, gcn_depth=1, rnn_size=8, node_dim=4, hyper_gnn_dim=4, middle_dim=2, dropout=0),
        "DSTAGNN": lambda: DSTAGNN(length, horizon, nodes, graph, d_model=8, d_k=2, d_v=2, n_heads=2),
        "GCLSTM": lambda: GCLSTM(length, horizon, nodes, graph, cov_dim=2, Ks=2, hidden_dim=8),
        "GTS": lambda: GTS(length, horizon, nodes, graph, input_dim=3, rnn_units=8, num_rnn_layers=1, max_diffusion_step=2, embedding_dim=8, temp=0.7, prior_strength=0.2),
    }


class PaperEquationTests(unittest.TestCase):
    def test_astgcn_attention_and_chebyshev_filter(self) -> None:
        model = factories()["ASTGCN"]()
        x = torch.randn(2, 6, 4, model.input_dim)
        spatial, temporal = model.blocks[0].attention(x)
        torch.testing.assert_close(spatial.sum(-1), torch.ones(2, 4))
        torch.testing.assert_close(temporal.sum(-1), torch.ones(2, 6))
        self.assertEqual(model.blocks[0].graph_convolution.chebyshev_supports.shape, (2, 4, 4))

    def test_dcrnn_has_dual_diffusion_in_every_gate(self) -> None:
        model = factories()["DCRNN"]()
        gate = model.encoder.cells[0].gates
        self.assertEqual(gate.supports.shape, (2, 4, 4))
        self.assertEqual(gate.projection.in_features, (3 + 8) * (1 + 2 * 2))

    def test_dgcrn_graph_changes_with_hidden_state(self) -> None:
        model = factories()["DGCRN"]().eval()
        zero = torch.zeros(2, 4, 8)
        changed = zero.clone()
        changed[:, 0] = 1
        first = model.graph_generator(zero)[0]
        second = model.graph_generator(changed)[0]
        self.assertGreater((first - second).abs().max().item(), 0)
        torch.testing.assert_close(first.sum(-1), torch.ones(2, 4))

    def test_dstagnn_uses_dynamic_graph_and_three_temporal_scales(self) -> None:
        model = factories()["DSTAGNN"]().eval()
        output = model(torch.randn(2, 6, 4))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual([layer.kernel_size for layer in model.block.temporal_convolution.filters], [(1, 3), (1, 5), (1, 7)])
        assert model.last_spatial_attention is not None
        torch.testing.assert_close(model.last_spatial_attention.sum(-1), torch.ones(2, 6, 4))

    def test_gclstm_graph_filter_drives_all_four_gates(self) -> None:
        model = factories()["GCLSTM"]()
        gate_projection = model.cell.gates
        self.assertEqual(gate_projection.supports.shape, (2, 4, 4))
        self.assertEqual(gate_projection.projection.out_features, 4 * model.cell.hidden_dim)

    def test_gts_edge_distribution_and_prior_objective(self) -> None:
        model = factories()["GTS"]().eval()
        x = torch.randn(2, 6, 4)
        graph = model.graph_discovery(x)
        self.assertEqual(graph.shape, (2, 4, 4))
        self.assertTrue(torch.equal(torch.diagonal(graph, dim1=-2, dim2=-1), torch.zeros(2, 4)))
        assert model.graph_discovery.last_edge_probabilities is not None
        probabilities = torch.softmax(model.graph_discovery.logits(x), -1)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 4, 4))
        self.assertTrue(torch.isfinite(model.graph_prior_loss(x)))


class CompleteRuntimeTests(unittest.TestCase):
    def test_forward_backward_active_gradients_and_round_trip(self) -> None:
        torch.manual_seed(260827)
        for name, factory in factories().items():
            with self.subTest(model=name):
                model = factory().cpu().eval()
                x = torch.randn(2, 6, 4, requires_grad=True)
                marks = raw_marks(2, 6)
                output = model(x, marks)
                self.assertEqual(output.shape, (2, 3, 4))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertGreater(x.grad.abs().max().item(), 0)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0, parameter_name)

                clone = factory().cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(clone(x.detach(), marks), model(x.detach(), marks))
                self.assertEqual(model(torch.randn(1, 6, 4), raw_marks(1, 6)).shape, (1, 3, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 5, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 6, 3))

    def test_marks_contract_is_explicit(self) -> None:
        x = torch.randn(2, 6, 4)
        for name, factory in factories().items():
            with self.subTest(model=name):
                model = factory().eval()
                unmarked = model(x)
                marked = model(x, raw_marks(2, 6, offset=7))
                if name == "DSTAGNN":
                    torch.testing.assert_close(marked, unmarked)
                else:
                    self.assertGreater((marked - unmarked).abs().max().item(), 0)

        dgcrn = factories()["DGCRN"]().eval()
        first = dgcrn(x, raw_marks(2, 6), x_mark_dec=raw_marks(2, 3))
        second = dgcrn(x, raw_marks(2, 6), x_mark_dec=raw_marks(2, 3, offset=8))
        self.assertGreater((first - second).abs().max().item(), 0)

    def test_adjacency_contract_is_active(self) -> None:
        x = torch.randn(2, 6, 4)
        identity = np.eye(4, dtype=np.float32)
        chain = adjacency(4)
        for name in factories().keys():
            with self.subTest(model=name):
                torch.manual_seed(91)
                first = factories(graph=identity)[name]().eval()
                torch.manual_seed(91)
                second = factories(graph=chain)[name]().eval()
                self.assertGreater((first(x) - second(x)).abs().max().item(), 0)

    def test_batch_node_and_sequence_boundaries(self) -> None:
        single = np.ones((1, 1), dtype=np.float32)
        boundary = {
            "ASTGCN": ASTGCN(1, 1, 1, single, cov_dim=0, nb_block=1, K=1, nb_chev_filter=2, nb_time_filter=2),
            "DCRNN": DCRNN(1, 1, 1, single, input_dim=1, rnn_units=2, max_diffusion_step=0),
            "DGCRN": DGCRN(1, 1, 1, single, rnn_size=2, node_dim=2, hyper_gnn_dim=2, middle_dim=1, dropout=0),
            "DSTAGNN": DSTAGNN(1, 1, 1, single, d_model=2, d_k=1, d_v=1, n_heads=1),
            "GCLSTM": GCLSTM(1, 1, 1, single, cov_dim=0, Ks=1, hidden_dim=2),
            "GTS": GTS(1, 1, 1, single, input_dim=1, rnn_units=2, max_diffusion_step=1, embedding_dim=2),
        }
        for name, model in boundary.items():
            with self.subTest(model=name):
                self.assertEqual(model.cpu().eval()(torch.randn(1, 1, 1)).shape, (1, 1, 1))

    def test_invalid_adjacency_is_rejected(self) -> None:
        bad = np.eye(3, dtype=np.float32)
        for name, factory in factories(graph=bad).items():
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    factory()


if __name__ == "__main__":
    unittest.main()
