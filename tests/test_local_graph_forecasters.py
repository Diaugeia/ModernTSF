"""Paper-structure and runtime checks for the locally rewritten graph models."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from moderntsf.models.agcrn.model import Model as AGCRN
from moderntsf.models.d2stgnn.model import Model as D2STGNN
from moderntsf.models.dfdgcn.model import Model as DFDGCN
from moderntsf.models.gwnet.model import Model as GWNet
from moderntsf.models.himnet.model import Model as HimNet
from moderntsf.models.staeformer.model import Model as STAEformer
from moderntsf.models.stdn.model import Model as STDN
from moderntsf.models.stemgnn.model import Model as StemGNN
from moderntsf.models.stgcn.model import Model as STGCN
from moderntsf.models.stid.model import Model as STID
from moderntsf.models.stnorm.model import Model as STNorm


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
        "GWNet": lambda: GWNet(6, 3, 4, graph, residual_channels=4, dilation_channels=4, skip_channels=8, end_channels=8, blocks=1, layers=2),
        "HimNet": lambda: HimNet(6, 3, 4, graph, hidden_dim=8, node_embedding_dim=4, st_embedding_dim=4, tod_embedding_dim=4, dow_embedding_dim=4),
        "STAEformer": lambda: STAEformer(6, 3, 4, graph, input_embedding_dim=4, tod_embedding_dim=4, dow_embedding_dim=4, adaptive_embedding_dim=4, feed_forward_dim=16, num_heads=2, num_layers=1),
        "STDN": lambda: STDN(6, 3, 4, graph, K=2, d=4, L=1, reference=2),
        "STGCN": lambda: STGCN(6, 3, 4, graph, Ks=2, hidden_dim=8, bottleneck_dim=4, out_hidden_dim=8, droprate=0),
        "STID": lambda: STID(6, 3, 4, graph, embed_dim=4),
        "STNorm": lambda: STNorm(6, 3, 4, graph, channels=4, blocks=1, layers=2),
        "StemGNN": lambda: StemGNN(6, 3, 4, multi_layer=2, dropout_rate=0),
    }


class PaperStructureTests(unittest.TestCase):
    def test_each_model_exposes_its_defining_operation(self) -> None:
        models = {name: factory() for name, factory in _factories().items()}
        self.assertEqual(models["AGCRN"].cells[0].gates.order, 2)
        self.assertEqual(len(models["D2STGNN"].layers), 2)
        self.assertEqual(models["DFDGCN"].frequency_graph.spectrum.in_features, 4)
        self.assertEqual(len(models["GWNet"].graph_supports()), 3)
        self.assertEqual(models["HimNet"].encoder[0].gates.order, 2)
        self.assertEqual(len(models["STAEformer"].spatial_layers), 1)
        self.assertEqual(models["STDN"].dynamic_diffusion.order, 2)
        self.assertEqual(models["STGCN"].block1.graph.supports.shape, (2, 4, 4))
        self.assertIsNotNone(models["STID"].node_embedding)
        self.assertIsNotNone(models["STNorm"].layers[0].spatial)
        self.assertIsNotNone(models["STNorm"].layers[0].temporal)
        self.assertEqual(len(models["StemGNN"].blocks), 2)

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
