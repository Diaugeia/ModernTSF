"""Paper-structure and runtime checks for the locally rewritten graph models."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from moderntsf.models.agcrn.model import Model as AGCRN
from moderntsf.models.d2stgnn.model import Model as D2STGNN
from moderntsf.models.dfdgcn.model import Model as DFDGCN
from moderntsf.models.extralonger.model import Model as Extralonger
from moderntsf.models.gwnet.model import Model as GWNet
from moderntsf.models.himnet.model import Model as HimNet
from moderntsf.models.ragc.model import Model as RAGC
from moderntsf.models.staeformer.model import Model as STAEformer
from moderntsf.models.stdn.model import Model as STDN
from moderntsf.models.stemgnn.model import Model as StemGNN
from moderntsf.models.stgcn.model import Model as STGCN
from moderntsf.models.stid.model import Model as STID
from moderntsf.models.stnorm.model import Model as STNorm
from moderntsf.models.st_ssdl.model import Model as STSSDL
from moderntsf.models.visifold.model import Model as VisiFold


def _graph(nodes: int = 4) -> np.ndarray:
    graph = np.eye(nodes, dtype=np.float32)
    for node in range(nodes - 1):
        graph[node, node + 1] = 1
        graph[node + 1, node] = 0.5
    return graph


def _marks(batch: int = 2, length: int = 6) -> torch.Tensor:
    rows = [[2026, 8, 1, index % 7, index % 24, 0] for index in range(length)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def _factories() -> dict[str, object]:
    graph = _graph()
    return {
        "AGCRN": lambda: AGCRN(6, 3, 4, graph, rnn_units=8, embed_dim=4),
        "D2STGNN": lambda: D2STGNN(6, 3, 4, graph, num_hidden=8, node_hidden=4, time_emb_dim=4, num_layers=2, forecast_dim=8),
        "DFDGCN": lambda: DFDGCN(6, 3, 4, graph, residual_channels=4, dilation_channels=4, skip_channels=8, end_channels=8, blocks=1, layers=2, fft_emb=4, identity_emb=4, hidden_emb=4),
        "Extralonger": lambda: Extralonger(
            6, 3, 4, graph,
            input_embedding_dim=4, tod_embedding_dim=2, dow_embedding_dim=2,
            spatial_embedding_dim=4, feed_forward_dim=8, num_heads=2, num_layers=1,
        ),
        "GWNet": lambda: GWNet(6, 3, 4, graph, residual_channels=4, dilation_channels=4, skip_channels=8, end_channels=8, blocks=1, layers=2),
        "HimNet": lambda: HimNet(6, 3, 4, graph, hidden_dim=8, node_embedding_dim=4, st_embedding_dim=4, tod_embedding_dim=4, dow_embedding_dim=4),
        "RAGC": lambda: RAGC(6, 3, 4, graph, spatial_dim=6, embed_dim=4, temp_dim_tid=4, temp_dim_diw=4, num_layer=2, order=2, sse_p=0.3),
        "STAEformer": lambda: STAEformer(6, 3, 4, graph, input_embedding_dim=4, tod_embedding_dim=4, dow_embedding_dim=4, adaptive_embedding_dim=4, feed_forward_dim=16, num_heads=2, num_layers=1),
        "STDN": lambda: STDN(6, 3, 4, graph, K=2, d=4, L=1, reference=2),
        "STGCN": lambda: STGCN(6, 3, 4, graph, Ks=2, hidden_dim=8, bottleneck_dim=4, out_hidden_dim=8, droprate=0),
        "STID": lambda: STID(6, 3, 4, graph, embed_dim=4),
        "STNorm": lambda: STNorm(6, 3, 4, graph, channels=4, blocks=1, layers=2),
        "ST-SSDL": lambda: STSSDL(6, 3, 4, graph, rnn_units=8, rnn_layers=1, cheb_k=2, prototype_num=4, prototype_dim=6),
        "StemGNN": lambda: StemGNN(6, 3, 4, multi_layer=2, dropout_rate=0),
        "VisiFold": lambda: VisiFold(
            6, 3, 4, graph,
            input_embedding_dim=4, tod_embedding_dim=4, dow_embedding_dim=4,
            spatial_embedding_dim=4, feed_forward_dim=8, num_heads=2, num_layers=1,
            mask_ratio=0.25, subgraph_size=2,
        ),
    }


class PaperStructureTests(unittest.TestCase):
    def test_each_model_exposes_its_defining_operation(self) -> None:
        models = {name: factory() for name, factory in _factories().items()}
        self.assertEqual(models["AGCRN"].cells[0].gates.order, 2)
        self.assertEqual(len(models["D2STGNN"].layers), 2)
        self.assertEqual(models["DFDGCN"].frequency_graph.spectrum.in_features, 4)
        self.assertEqual(models["Extralonger"].node_embedding.shape, (4, 4))  # node identity embedding for the spatial route
        self.assertTrue(models["Extralonger"].adj_mask.diagonal().all())  # global-local attention keeps self-loops
        self.assertEqual(len(models["Extralonger"].spatial_layers), 1)
        self.assertEqual(len(models["GWNet"].graph_supports()), 3)
        self.assertEqual(models["HimNet"].encoder[0].gates.order, 2)
        self.assertEqual(models["RAGC"].node_emb.shape, (4, 6))  # spatial_dim=6 node embedding regularized by SSE
        self.assertEqual(len(models["RAGC"].gconvs), 2)  # num_layer=2 efficient-cosine-operator blocks
        self.assertEqual(models["RAGC"].gconvs[0].order, 2)
        self.assertEqual(len(models["STAEformer"].spatial_layers), 1)
        self.assertEqual(models["STDN"].dynamic_diffusion.order, 2)
        self.assertEqual(models["STGCN"].block1.graph.supports.shape, (2, 4, 4))
        self.assertIsNotNone(models["STID"].node_embedding)
        self.assertIsNotNone(models["STNorm"].layers[0].spatial)
        self.assertIsNotNone(models["STNorm"].layers[0].temporal)
        self.assertEqual(models["ST-SSDL"].prototype_memory.prototypes.shape, (4, 6))  # prototype_num=4, prototype_dim=6
        self.assertEqual(models["ST-SSDL"].prototype_memory.num_prototypes, 4)
        self.assertEqual(len(models["StemGNN"].blocks), 2)
        self.assertEqual(models["VisiFold"].fold_input[0].in_features, 6)  # temporal folding graph over seq_len
        self.assertEqual(models["VisiFold"].subgraph_size, 2)  # node-visibility subgraph grouping

    def test_visifold_node_visibility_masks_and_regroups_during_training(self) -> None:
        model = _factories()["VisiFold"]()
        model.train()
        values = torch.randn(2, 6, 4)
        output = model(values, _marks())
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_st_ssdl_prototype_memory_deviation_reacts_to_input_perturbation(self) -> None:
        """The paper's self-supervised deviation signal must be sensitive to
        how different the current window's encoding is from a reference
        one — a near-identical reference should score near zero deviation
        while a strongly perturbed reference should score materially higher.
        """
        torch.manual_seed(0)
        model = _factories()["ST-SSDL"]().eval()
        marks = _marks()
        current = torch.randn(2, 6, 4)

        with torch.no_grad():
            close_losses = model.auxiliary_losses(current, marks, current.clone(), marks)
            far_losses = model.auxiliary_losses(current, marks, current + 5.0 * torch.randn(2, 6, 4), marks)

        for key in ("contrastive_loss", "deviation_loss"):
            self.assertTrue(torch.isfinite(close_losses[key]).all())
            self.assertTrue(torch.isfinite(far_losses[key]).all())

        # An identical historical window must retrieve the same nearest
        # prototype, so its own current-vs-historical distance is exactly
        # zero; the deviation loss compares that zero distance to the
        # latent-query distance, which is also (near) zero for an identical
        # window but grows for a heavily perturbed one.
        self.assertAlmostEqual(close_losses["deviation_loss"].item(), 0.0, places=5)
        self.assertGreater(far_losses["deviation_loss"].item(), close_losses["deviation_loss"].item())

    def test_st_ssdl_prototype_memory_retrieval_shapes(self) -> None:
        from moderntsf.models._components.deviation_memory import PrototypeMemory, deviation_score

        torch.manual_seed(0)
        memory = PrototypeMemory(query_dim=8, prototype_dim=6, num_prototypes=5)
        h = torch.randn(2, 4, 8)  # (B, N, query_dim)
        retrieval = memory(h)
        self.assertEqual(retrieval.value.shape, (2, 4, 6))
        self.assertEqual(retrieval.query.shape, (2, 4, 6))
        self.assertEqual(retrieval.nearest.shape, (2, 4, 6))
        self.assertEqual(retrieval.second_nearest.shape, (2, 4, 6))
        self.assertEqual(retrieval.indices.shape, (2, 4, 2))
        self.assertTrue(torch.equal(deviation_score(retrieval.query, retrieval.query), torch.zeros(2, 4)))
        self.assertTrue((deviation_score(retrieval.query, -retrieval.query) >= 0).all())

    def test_extralonger_global_local_attention_matches_paper_equation(self) -> None:
        from moderntsf.models._components.graph_masked_attention import GlobalLocalGraphAttention

        torch.manual_seed(0)
        attn = GlobalLocalGraphAttention(model_dim=8, num_heads=2).eval()
        x = torch.randn(2, 5, 8)
        adj_mask = torch.zeros(5, 5, dtype=torch.bool)
        adj_mask.fill_diagonal_(True)
        adj_mask[0, 1] = adj_mask[1, 0] = True

        with torch.no_grad():
            q, k, v = attn.fc_q(x), attn.fc_k(x), attn.fc_v(x)

            def split(t: torch.Tensor) -> torch.Tensor:
                return t.view(2, 5, 2, 4).movedim(-2, -3)

            scores = (split(q) @ split(k).transpose(-1, -2)) / (4**0.5)
            expected_global = torch.softmax(scores, dim=-1)
            expected_local = torch.softmax(scores.masked_fill(~adj_mask, float("-inf")), dim=-1)
            expected_attn = (expected_global + expected_local) / 2.0
            expected_out = expected_attn @ split(v)
            expected_out = expected_out.movedim(-3, -2).reshape(2, 5, 8)
            expected = attn.out_proj(expected_out)

            actual = attn(x, x, x, adj_mask=adj_mask)
            actual_global_only = attn(x, x, x)

        torch.testing.assert_close(actual, expected)
        # Without an adjacency mask the layer reduces to plain dense attention.
        expected_global_only = attn.out_proj(
            (expected_global @ split(v)).movedim(-3, -2).reshape(2, 5, 8)
        )
        torch.testing.assert_close(actual_global_only, expected_global_only)

    def test_ragc_efficient_cosine_operator_matches_dense_adjacency_matmul(self) -> None:
        """RAGC's kernel-trick diffusion (linear in N) must exactly match the
        dense-adjacency degree-normalized matmul it is algebraically
        equivalent to, for the implicit adjacency A = support @ support.T.
        """
        from moderntsf.models._components.regularized_adaptive_graph_conv import EfficientCosineGraphConv

        torch.manual_seed(0)
        gconv = EfficientCosineGraphConv(hidden_dim=5, spatial_dim=3, order=1).eval()
        node_embedding = torch.randn(4, 3)
        x = torch.randn(2, 4, 5)

        support = gconv.compute_support(node_embedding)
        dense_adj = support @ support.t()
        degree = dense_adj.sum(dim=1, keepdim=True) + 1e-6
        expected_hop = torch.einsum("nm,bmh->bnh", dense_adj, x) / degree

        actual_hop = gconv._kernelized_hop(x, support)
        torch.testing.assert_close(actual_hop, expected_hop, atol=1e-5, rtol=1e-5)

    def test_ragc_stochastic_shared_embedding_regularizes_only_in_training(self) -> None:
        from moderntsf.models._components.regularized_adaptive_graph_conv import StochasticSharedEmbedding

        torch.manual_seed(0)
        sse = StochasticSharedEmbedding(p=1.0)
        embeddings = torch.arange(12, dtype=torch.float32).view(4, 3)

        sse.eval()
        torch.testing.assert_close(sse(embeddings), embeddings)

        sse.train()
        regularized = sse(embeddings)
        self.assertEqual(regularized.shape, embeddings.shape)
        # p=1.0 always swaps every row with some (possibly itself) row from
        # the table, so every output row must still be one of the original
        # embedding rows.
        for row in regularized:
            self.assertTrue(any(torch.equal(row, candidate) for candidate in embeddings))

    def test_ragc_graph_regularization_loss_is_separate_from_forward(self) -> None:
        model = _factories()["RAGC"]()
        self.assertFalse(hasattr(model.forward, "graph_regularization_loss"))
        loss = model.graph_regularization_loss()
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_input_dependent_graphs_are_row_normalized(self) -> None:
        values = torch.randn(2, 6, 4)
        d2 = _factories()["D2STGNN"]().eval()
        data = torch.randn(2, 6, 4, 8)
        graph = d2.graph(data)
        torch.testing.assert_close(graph.sum(-1), torch.ones(2, 4))

        dfd = _factories()["DFDGCN"]().eval()
        first = dfd.frequency_graph(values)
        second = dfd.frequency_graph(values + torch.linspace(0, 1, 6).view(1, 6, 1))
        torch.testing.assert_close(first.sum(-1), torch.ones(2, 4))
        self.assertGreater((first - second).abs().max().item(), 0)


class RuntimeTests(unittest.TestCase):
    def test_forward_backward_and_state_round_trip(self) -> None:
        for name, factory in _factories().items():
            with self.subTest(model=name):
                torch.manual_seed(20260828)
                model = factory().eval()
                values = torch.randn(2, 6, 4, requires_grad=True)
                output = model(values, _marks(), x_mark_dec=_marks(length=3))
                self.assertEqual(output.shape, (2, 3, 4))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(values.grad)
                active = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
                self.assertTrue(active)
                self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in active))
                clone = factory().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(
                    clone(values.detach(), _marks(), x_mark_dec=_marks(length=3)),
                    model(values.detach(), _marks(), x_mark_dec=_marks(length=3)),
                )

    def test_shape_boundaries_fail_explicitly(self) -> None:
        for name, factory in _factories().items():
            with self.subTest(model=name):
                model = factory().eval()
                self.assertEqual(model(torch.randn(1, 6, 4), _marks(1)).shape, (1, 3, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 5, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 6, 3))


if __name__ == "__main__":
    unittest.main()
