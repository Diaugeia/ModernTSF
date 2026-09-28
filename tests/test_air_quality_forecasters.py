"""Paper-equation and complete runtime tests for six air-quality implementations."""

from __future__ import annotations

import copy
import unittest

import torch
from pydantic import ValidationError

from moderntsf.models.aircade.model import DomainKnowledgeAttention, Model as AirCade
from moderntsf.models.aircade.spec import ModelParameterConfig as AirCadeParameters
from moderntsf.models.airdualode.model import BoundaryAwareDynamics, Model as AirDualODE
from moderntsf.models.airdualode.spec import ModelParameterConfig as AirDualODEParameters
from moderntsf.models.airformer.model import CausalTemporalAttention, Model as AirFormer, default_dartboard
from moderntsf.models.airformer.spec import ModelParameterConfig as AirFormerParameters
from moderntsf.models.airphynet.model import Model as AirPhyNet, PhysicsVectorField
from moderntsf.models.airphynet.spec import ModelParameterConfig as AirPhyNetParameters
from moderntsf.models.cauair.model import CacheAttention, Model as CauAir
from moderntsf.models.cauair.spec import ModelParameterConfig as CauAirParameters
from moderntsf.models.deepair.model import Model as DeepAir, default_spatial_projection
from moderntsf.models.deepair.spec import ModelParameterConfig as DeepAirParameters


def marks(batch: int, length: int, *, offset: float = 0.0) -> torch.Tensor:
    result = torch.zeros(batch, length, 6)
    result[..., 0] = 2024
    result[..., 1] = 1
    result[..., 2] = torch.arange(1, length + 1)
    result[..., 3] = (torch.arange(length) + offset) % 7
    result[..., 4] = (torch.arange(length) + offset) % 24
    return result


def factories(length: int = 6, horizon: int = 3, nodes: int = 3):
    return {
        "AirCade": lambda: AirCade(length, horizon, nodes, d_model=16, prompt_dim=2,
            adaptive_dim=3, num_heads=2, temporal_layers=1, spatial_layers=1),
        "AirDualODE": lambda: AirDualODE(length, horizon, nodes, phy_latent_dim=4,
            unk_latent_dim=4, gcn_hidden_dim=8, n_heads=2),
        "AirFormer": lambda: AirFormer(length, horizon, nodes, d_model=8, nhead=2,
            num_encoder_layers=2, spatial_regions=2, dropout=0.0),
        "AirPhyNet": lambda: AirPhyNet(length, horizon, nodes, latent_dim=4,
            rnn_units=8, ode_method="euler"),
        "CauAir": lambda: CauAir(length, horizon, nodes, dim=8, cache_count=2, heads=2),
        "DeepAir": lambda: DeepAir(length, horizon, nodes, hidden_dim=8, spatial_regions=2),
    }


class PaperEquationTests(unittest.TestCase):
    def test_airphynet_vector_field_is_equation_11(self) -> None:
        field = PhysicsVectorField(1)
        with torch.no_grad():
            field.diffusion_raw.fill_(0.0)
            field.gate_logits.fill_(0.0)
        state = torch.tensor([[[1.0], [3.0]]])
        distance = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        flow = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        laplacian = torch.eye(2) - distance
        flow_operator = torch.eye(2) - flow
        expected = -0.5 * torch.nn.functional.softplus(torch.tensor(0.0)) * torch.tanh(
            torch.einsum("nm,bmd->bnd", laplacian, state)
        ) - 0.5 * torch.tanh(torch.einsum("nm,bmd->bnd", flow_operator, state))
        torch.testing.assert_close(field(state, distance, flow), expected)

    def test_airdualode_boundary_aware_equation_6(self) -> None:
        dynamics = BoundaryAwareDynamics(2, 3)
        state = torch.tensor([[1.0, 3.0]])
        support = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        flow = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        gate = torch.full((1, 2, 1), 0.25)
        coefficient = torch.full((1, 2, 1), 2.0)
        correction = torch.tensor([[[-0.1], [0.2]]])
        expected = 0.25 * 2.0 * (-torch.einsum("nm,bm->bn", torch.eye(2) - support, state))
        expected += 0.75 * torch.einsum("nm,bm->bn", flow - torch.eye(2), state)
        expected += torch.tensor([[-0.1, 0.6]])
        torch.testing.assert_close(
            dynamics(state, support, flow, gate, coefficient, correction), expected
        )

    def test_airformer_causal_window_and_dartboard_projection(self) -> None:
        projection = default_dartboard(4, 2)
        torch.testing.assert_close(projection.sum(-1), torch.ones(4, 2))
        torch.manual_seed(17)
        attention = CausalTemporalAttention(4, 1, 4, 0.0).eval()
        original = torch.randn(1, 5, 4)
        changed = original.clone()
        changed[:, 4] += 100
        torch.testing.assert_close(attention(original)[:, :4], attention(changed)[:, :4])

    def test_aircade_has_four_paths_and_environment_intervention(self) -> None:
        attention = DomainKnowledgeAttention(8, 2, 4, 3, 3, True)
        self.assertEqual(tuple(attention.intervention_logits.shape), (3, 4, 4))
        self.assertEqual(attention.output.in_features, 32)
        values = torch.randn(2, 4, 8)
        self.assertEqual(tuple(attention(values, values, values).shape), (2, 4, 8))

    def test_cauair_cache_attention_is_station_permutation_equivariant(self) -> None:
        torch.manual_seed(19)
        attention = CacheAttention(8, 2, 3).eval()
        values = torch.randn(2, 4, 8)
        permutation = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(
            attention(values[:, permutation]), attention(values)[:, permutation]
        )

    def test_deepair_spatial_projection_is_target_relative_and_normalized(self) -> None:
        projection = default_spatial_projection(4, 2)
        self.assertEqual(tuple(projection.shape), (4, 2, 4))
        torch.testing.assert_close(projection.sum(-1), torch.ones(4, 2))
        self.assertFalse(torch.equal(projection[0], projection[1]))


class ParameterSchemaTests(unittest.TestCase):
    def test_invalid_widths_solvers_and_dimensions_are_rejected(self) -> None:
        invalid = (
            lambda: AirCadeParameters(enc_in=3, d_model=10, num_heads=3),
            lambda: AirDualODEParameters(enc_in=3, unk_latent_dim=5, n_heads=2),
            lambda: AirFormerParameters(enc_in=3, d_model=10, nhead=3),
            lambda: AirPhyNetParameters(enc_in=3, ode_method="dopri5"),
            lambda: CauAirParameters(enc_in=3, dim=10, heads=3),
            lambda: DeepAirParameters(enc_in=0),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory()


class CompleteRuntimeTests(unittest.TestCase):
    def test_forward_backward_gradients_serialization_marks_and_boundaries(self) -> None:
        torch.manual_seed(260827)
        for name, factory in factories().items():
            with self.subTest(model=name):
                model = factory().cpu()
                values = torch.randn(2, 6, 3, requires_grad=True)
                output = model(values, marks(2, 6), None, marks(2, 3))
                self.assertEqual(tuple(output.shape), (2, 3, 3))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(values.grad)
                self.assertGreater(values.grad.abs().max().item(), 0.0)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name}:{parameter_name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all(), f"{name}:{parameter_name}")
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, f"{name}:{parameter_name}")

                model.eval()
                expected = model(values.detach(), marks(2, 6), None, marks(2, 3))
                clone = factory().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(
                    clone(values.detach(), marks(2, 6), None, marks(2, 3)), expected
                )
                changed = model(values.detach(), marks(2, 6, offset=3), None, marks(2, 3, offset=5))
                self.assertFalse(torch.equal(expected, changed), name)
                self.assertEqual(tuple(model(torch.randn(1, 6, 3), marks(1, 6), None, marks(1, 3)).shape), (1, 3, 3))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 5, 3), marks(1, 5), None, marks(1, 3))

                structured = torch.randn(1, 6, 3, 2)
                future = torch.randn(1, 3, 3, 2)
                result = model(torch.randn(1, 6, 3), structured, None, future)
                self.assertEqual(tuple(result.shape), (1, 3, 3))

    def test_minimum_sequence_batch_and_node_boundary(self) -> None:
        for name, factory in factories(1, 1, 1).items():
            with self.subTest(model=name):
                model = factory().eval()
                result = model(torch.randn(1, 1, 1), marks(1, 1), None, marks(1, 1))
                self.assertEqual(tuple(result.shape), (1, 1, 1))
                self.assertTrue(torch.isfinite(result).all())

    def test_spatial_models_respond_to_graph_inputs(self) -> None:
        torch.manual_seed(71)
        identity = torch.eye(3)
        dense = torch.ones(3, 3)
        cases = (
            (lambda graph: AirDualODE(6, 3, 3, adj_mx=graph, flow_mx=graph,
                phy_latent_dim=4, unk_latent_dim=4, gcn_hidden_dim=8, n_heads=2)),
            (lambda graph: AirPhyNet(6, 3, 3, adj_mx=graph, flow_mx=graph,
                latent_dim=4, rnn_units=8, ode_method="euler")),
            (lambda graph: AirFormer(6, 3, 3, dartboard_mx=graph, d_model=8,
                nhead=2, num_encoder_layers=1, dropout=0.0)),
            (lambda graph: DeepAir(6, 3, 3, spatial_mx=graph, hidden_dim=8)),
        )
        values = torch.randn(2, 6, 3)
        for factory in cases:
            first = factory(identity).eval()
            second = factory(dense).eval()
            parameter_state = dict(first.named_parameters())
            with torch.no_grad():
                for name, parameter in second.named_parameters():
                    parameter.copy_(parameter_state[name])
            left = first(values, marks(2, 6), None, marks(2, 3))
            right = second(values, marks(2, 6), None, marks(2, 3))
            self.assertFalse(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
