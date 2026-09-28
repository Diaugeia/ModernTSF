"""Paper-structure and strict runtime tests for graph implementation batch C."""
from __future__ import annotations
import copy
import unittest
import numpy as np
import torch
from moderntsf.models.mage.model import Model as MAGE
from moderntsf.models.mgsfformer.model import Model as MGSFformer
from moderntsf.models.pcdcnet.model import Model as PCDCNet
from moderntsf.models.stop.model import Model as STOP
from moderntsf.models.sttn.model import Model as STTN
from moderntsf.models.stwave.model import Model as STWave, wavelet_disentangle


def graph(nodes: int) -> np.ndarray:
    result = np.eye(nodes, dtype=np.float32)
    for index in range(nodes - 1):
        result[index, index + 1] = 1.0
        result[index + 1, index] = 0.25
    return result


def marks(batch: int, steps: int, offset: int = 0) -> torch.Tensor:
    rows = [[2026, 8, 1 + index // 24, 5, (index + offset) % 24, 0] for index in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def factories():
    g = graph(4)
    return {
        "MAGE": lambda: MAGE(6, 3, 4, model_dim=8, recur_num=3, topk=2, node_dim=4),
        "MGSFformer": lambda: MGSFformer(24, 3, 4, IE_dim=8, dropout=0, num_head=2),
        "PCDCNet": lambda: PCDCNet(6, 3, 4, g, d_model=8, dropout=0),
        "STOP": lambda: STOP(6, 3, 4, model_dim=4, prompt_dim=2, num_layer=1, hid_dim=8, core=2, head=2),
        "STTN": lambda: STTN(6, 3, 4, g, d_model=8, num_layers=1, dropout=0),
        "STWave": lambda: STWave(6, 3, 4, g, hidden_size=4, layers=1),
    }


class PaperStructureTests(unittest.TestCase):
    def test_mage_sparse_factorised_experts(self) -> None:
        model = factories()["MAGE"]().eval()
        model(torch.randn(2, 6, 4), marks(2, 6))
        routing = model.blocks[0].last_routing
        assert routing is not None
        torch.testing.assert_close(routing.sum(-1), torch.ones(2, 4))
        self.assertEqual(model.blocks[0].experts[0].source.shape, (4, 4))
        self.assertFalse(any(parameter.ndim == 3 and parameter.shape[-2:] == (4, 4)
                             for parameter in model.parameters()))

    def test_mgsfformer_granularities_and_dynamic_fusion(self) -> None:
        model = factories()["MGSFformer"]().eval()
        output = model(torch.randn(2, 24, 4))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual(model.GRANULARITIES, (1, 3, 6, 12, 24))
        weights = model.fusion.last_weights
        assert weights is not None
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))

    def test_pcdcnet_equation_order_and_constraint(self) -> None:
        model = factories()["PCDCNet"]().eval()
        model(torch.randn(2, 6, 4), marks(2, 6))
        self.assertEqual(model.transport.laplacian.shape, (4, 4))
        self.assertTrue(torch.isfinite(model.domain_informed_constraint()))
        assert model.last_transport is not None
        self.assertEqual(model.last_transport.shape, (2, 3, 4))

    def test_stop_interaction_is_node_context_not_node_node(self) -> None:
        model = factories()["STOP"]().eval()
        x = torch.randn(2, 6, 4)
        model(x, marks(2, 6))
        aggregation = model.central.last_aggregation
        diffusion = model.central.last_diffusion
        assert aggregation is not None and diffusion is not None
        self.assertEqual(aggregation.shape[-2:], (2, 4))
        self.assertEqual(diffusion.shape[-2:], (4, 2))
        self.assertEqual(model.environment_forecasts(x, marks(2, 6), 2).shape, (2, 2, 3, 4))

    def test_sttn_dynamic_directed_and_stationary_paths(self) -> None:
        model = factories()["STTN"]().eval()
        model(torch.randn(2, 6, 4), marks(2, 6))
        attention = model.blocks[0].spatial.last_attention
        assert attention is not None
        self.assertEqual(attention.shape, (2, 6, 4, 4, 4))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 6, 4, 4))

    def test_stwave_disentanglement_and_query_sampling(self) -> None:
        x = torch.randn(2, 6, 4)
        low, high = wavelet_disentangle(x)
        torch.testing.assert_close(low + high, x)
        model = factories()["STWave"]().eval()
        model(x, marks(2, 6))
        sampled_mask = model.low_encoders[0].spatial.last_mask
        assert sampled_mask is not None
        self.assertTrue(sampled_mask.any())
        self.assertFalse(sampled_mask.all())


class CompleteRuntimeTests(unittest.TestCase):
    def test_forward_backward_roundtrip_batch_and_length_contracts(self) -> None:
        torch.manual_seed(260827)
        for name, factory in factories().items():
            with self.subTest(model=name):
                model = factory().cpu().eval()
                length = model.seq_len
                x = torch.randn(2, length, 4, requires_grad=True)
                calendar = marks(2, length)
                output = model(x, calendar)
                self.assertEqual(output.shape, (2, 3, 4))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertGreater(x.grad.abs().max().item(), 0)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name}: {parameter_name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all(), f"{name}: {parameter_name}")
                    if name == "MGSFformer" and parameter_name == "fusion.score.bias":
                        # The same scalar is added to every softmax logit. It
                        # cancels analytically; floating-point noise is not an
                        # active-gradient contract and differs across backends.
                        self.assertLessEqual(parameter.grad.abs().max().item(), 1e-6)
                    else:
                        self.assertGreater(parameter.grad.abs().max().item(), 0, f"{name}: {parameter_name}")
                clone = factory().cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(clone(x.detach(), calendar), model(x.detach(), calendar))
                self.assertEqual(model(torch.randn(1, length, 4), marks(1, length)).shape, (1, 3, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, length - 1, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, length, 3))

    def test_marks_contract(self) -> None:
        for name, factory in factories().items():
            with self.subTest(model=name):
                model = factory().eval()
                x = torch.randn(2, model.seq_len, 4)
                difference = (model(x, marks(2, model.seq_len, 7)) - model(x)).abs().max()
                if name == "MGSFformer":
                    self.assertEqual(difference.item(), 0)
                else:
                    self.assertGreater(difference.item(), 0)

    def test_graph_contract_and_effect(self) -> None:
        x = torch.randn(2, 6, 4)
        for cls, kwargs in (
            (PCDCNet, {"d_model": 8, "dropout": 0}),
            (STTN, {"d_model": 8, "num_layers": 1, "dropout": 0}),
            (STWave, {"hidden_size": 4, "layers": 1}),
        ):
            with self.subTest(model=cls.__name__):
                torch.manual_seed(91)
                first = cls(6, 3, 4, np.eye(4, dtype=np.float32), **kwargs).eval()
                torch.manual_seed(91)
                second = cls(6, 3, 4, graph(4), **kwargs).eval()
                self.assertGreater((first(x) - second(x)).abs().max().item(), 0)
                with self.assertRaises(ValueError):
                    cls(6, 3, 4, np.eye(3, dtype=np.float32), **kwargs)

    def test_minimum_boundaries(self) -> None:
        one = np.ones((1, 1), dtype=np.float32)
        models = (
            MAGE(1, 1, 1, model_dim=2, recur_num=2, topk=1, node_dim=1),
            MGSFformer(24, 1, 1, IE_dim=2, dropout=0, num_head=1),
            PCDCNet(1, 1, 1, one, cov_dim=0, d_model=2, dropout=0),
            STOP(1, 1, 1, model_dim=2, prompt_dim=1, num_layer=1, hid_dim=2, core=1, head=1),
            STTN(1, 1, 1, one, cov_dim=0, d_model=4, num_layers=1, dropout=0),
            STWave(1, 1, 1, one, hidden_size=2, layers=1),
        )
        for model in models:
            with self.subTest(model=type(model).__module__):
                self.assertEqual(model.eval()(torch.randn(1, model.seq_len, 1)).shape, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()


class FusionInvarianceTests(unittest.TestCase):
    def test_shared_logit_bias_does_not_change_fusion(self):
        from moderntsf.models.mgsfformer.model import DynamicFusion
        torch.manual_seed(47)
        fusion = DynamicFusion(4, 3).double()
        features = [torch.randn(2, 5, 3, 4, dtype=torch.float64) for _ in range(3)]
        forecasts = [torch.randn(2, 2, 3, dtype=torch.float64) for _ in range(3)]
        expected = fusion(features, forecasts)
        expected.square().mean().backward()
        self.assertLessEqual(fusion.score.bias.grad.abs().max().item(), 1e-12)
        with torch.no_grad():
            fusion.score.bias.add_(7.0)
        torch.testing.assert_close(fusion(features, forecasts), expected.detach(), rtol=1e-12, atol=1e-12)
