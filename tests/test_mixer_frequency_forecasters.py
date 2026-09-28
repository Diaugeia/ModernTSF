"""Paper-structure and complete runtime tests for six mixer/frequency implementations."""

from __future__ import annotations

import copy
import unittest

import torch
from pydantic import ValidationError

from moderntsf.models.amplifier.model import Model as Amplifier
from moderntsf.models.amplifier.model import flipped_spectrum
from moderntsf.models.amplifier.spec import ModelParameterConfig as AmplifierParameters
from moderntsf.models.cmos.model import Model as CMoS
from moderntsf.models.cmos.model import periodic_correlation_initialization
from moderntsf.models.cmos.spec import ModelParameterConfig as CMoSParameters
from moderntsf.models.crib.model import Model as CRIB
from moderntsf.models.crib.model import observed_statistics
from moderntsf.models.crib.spec import ModelParameterConfig as CRIBParameters
from moderntsf.models.crossgnn.model import AdaptiveMultiScaleIdentifier
from moderntsf.models.crossgnn.model import Model as CrossGNN
from moderntsf.models.crossgnn.spec import ModelParameterConfig as CrossGNNParameters
from moderntsf.models.film.model import Model as FiLM
from moderntsf.models.film.model import bilinear_discretize, legendre_basis, legt_transition
from moderntsf.models.film.spec import ModelParameterConfig as FiLMParameters
from moderntsf.models.frets.model import ComplexFrequencyMLP
from moderntsf.models.frets.model import Model as FreTS
from moderntsf.models.frets.spec import ModelParameterConfig as FreTSParameters


class PaperStructureTests(unittest.TestCase):
    def test_amplifier_spectrum_flip_and_sci_structure(self):
        values = torch.arange(24.0).reshape(1, 8, 3)
        torch.testing.assert_close(
            flipped_spectrum(values), torch.flip(torch.fft.rfft(values, dim=1), (1,))
        )
        model = RuntimeTests.factories()["Amplifier"]()
        self.assertIsNotNone(model.sci)
        self.assertEqual(model.restoration.real.in_features, 5)
        self.assertEqual(model.restoration.real.out_features, 2)

    def test_cmos_periodic_peaks_and_channel_conditioned_mixture(self):
        initialized = periodic_correlation_initialization(4, 3, 2)
        expected = torch.tensor(
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]]
        )
        torch.testing.assert_close(initialized, expected)
        model = RuntimeTests.factories()["CMoS"]()
        self.assertEqual(len(model.mixer.aggregators), 3)
        self.assertEqual(tuple(model.mixer.correlations.shape), (3, 4, 2))

    def test_crib_observed_statistics_attention_and_auxiliary_objective(self):
        values = torch.tensor([[[1.0], [9.0], [3.0], [9.0]]])
        observed = torch.tensor([[[True], [False], [True], [False]]])
        mean, stdev = observed_statistics(values, observed)
        torch.testing.assert_close(mean, torch.tensor([[[2.0]]]))
        torch.testing.assert_close(stdev.square(), torch.tensor([[[1.00001]]]))
        model = RuntimeTests.factories()["CRIB"]().train()
        incomplete = torch.randn(2, 8, 3)
        incomplete[0, 2, 1] = torch.nan
        output = model(incomplete)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(model.aux_loss)
        self.assertTrue(torch.isfinite(model.aux_loss))

    def test_crossgnn_amsi_and_signed_sparse_graphs(self):
        time = torch.arange(8.0)
        periodic = torch.sin(2 * torch.pi * time / 2).reshape(1, 8, 1).repeat(2, 1, 4)
        amsi = AdaptiveMultiScaleIdentifier(2)
        multiscale, periods, lengths = amsi(periodic)
        self.assertEqual(multiscale.shape[1], sum(lengths))
        self.assertEqual(len(periods), 2)
        model = RuntimeTests.factories()["CrossGNN"]()
        encoded, periods, lengths = model.amsi(torch.randn(2, 8, 4))
        layer = model.layers[0]
        temporal = layer.temporal_adjacency(periods, lengths)
        variable = layer.variable_adjacency()
        torch.testing.assert_close(temporal.sum(-1), torch.ones(temporal.shape[0]))
        self.assertTrue((variable > 0).any())
        self.assertTrue((variable < 0).any())

    def test_film_legendre_bilinear_and_low_rank_frequency_path(self):
        matrix, vector = legt_transition(4)
        discrete_matrix, discrete_vector = bilinear_discretize(matrix, vector, 0.25)
        identity = torch.eye(4, dtype=matrix.dtype)
        expected = torch.linalg.solve(
            identity - 0.125 * matrix, identity + 0.125 * matrix
        ).float()
        torch.testing.assert_close(discrete_matrix, expected)
        self.assertEqual(tuple(discrete_vector.shape), (4,))
        basis = legendre_basis(3, 4)
        torch.testing.assert_close(basis[:, 0], torch.ones(3))
        model = RuntimeTests.factories()["FiLM"]()
        self.assertEqual(model.experts[0].frequency_layer.left_real.shape[-1], 2)

    def test_frets_uses_full_complex_matrix_not_only_diagonal(self):
        layer = ComplexFrequencyMLP(2, 0.0)
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
            layer.real_weight[0, 1] = 2.0
        values = torch.complex(torch.tensor([[[1.0, 3.0]]]), torch.zeros(1, 1, 2))
        output = layer(values)
        torch.testing.assert_close(output.real, torch.tensor([[[6.0, 0.0]]]))


class SchemaTests(unittest.TestCase):
    def test_invalid_architecture_constraints(self):
        invalid = (
            lambda: AmplifierParameters(enc_in=2, moving_average=0),
            lambda: CMoSParameters(enc_in=2, period=0),
            lambda: CRIBParameters(enc_in=2, model_dim=7, heads_num=2),
            lambda: CrossGNNParameters(enc_in=4, tk=3),
            lambda: FiLMParameters(enc_in=2, order=3, rank=4),
            lambda: FreTSParameters(enc_in=2, sparsity_threshold=-0.1),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory()


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def factories(length: int = 8, pred: int = 3):
        return {
            "Amplifier": lambda: Amplifier(length, pred, 3, 12, True, 3),
            "CMoS": lambda: CMoS(3, length, 4, 2, 3, 2, 4),
            "CRIB": lambda: CRIB(length, pred, 3, 2, 8, 2, 1, 0.0),
            "CrossGNN": lambda: CrossGNN(
                length, pred, 4, 1, True, 2, 2, True, True, 0.0, 4, 4, 6
            ),
            "FiLM": lambda: FiLM(
                length, pred, 3, ratio=0.5, multiscale=(1, 2), order=6, rank=2
            ),
            "FreTS": lambda: FreTS(
                length, pred, 3, embed_size=6, hidden_size=12, sparsity_threshold=0.0
            ),
        }

    @staticmethod
    def call(model, values, changed_marks: bool = False):
        marks = values.new_zeros(values.shape[0], values.shape[1], 6)
        if changed_marks:
            marks[..., 4] = 9
        return model(values, marks, None, None)

    def test_forward_backward_active_gradients_round_trip_and_contracts(self):
        torch.manual_seed(4811)
        channels = {"CrossGNN": 4}
        for name, factory in self.factories().items():
            with self.subTest(model=name):
                model = factory().cpu().train()
                values = torch.randn(2, 8, channels.get(name, 3), requires_grad=True)
                output = self.call(model, values)
                expected_length = 4 if name == "CMoS" else 3
                self.assertEqual(tuple(output.shape), (2, expected_length, values.shape[-1]))
                self.assertTrue(torch.isfinite(output).all())
                loss = output.square().mean()
                if getattr(model, "aux_loss", None) is not None:
                    loss = loss + model.aux_loss
                loss.backward()
                self.assertIsNotNone(values.grad)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name}:{parameter_name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)
                model.eval()
                expected = self.call(model, values.detach())
                clone = factory().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(self.call(clone, values.detach()), expected)
                batch_one = torch.randn(1, 8, values.shape[-1])
                self.assertEqual(self.call(model, batch_one).shape[0], 1)
                with self.assertRaises(ValueError):
                    self.call(model, torch.randn(1, 7, values.shape[-1]))
                torch.testing.assert_close(
                    self.call(model, values.detach(), changed_marks=True), expected
                )

    def test_minimum_sequence_boundaries(self):
        cases = (
            (Amplifier(1, 1, 1, 2, True, 1), torch.randn(1, 1, 1)),
            (CMoS(1, 1, 1, 1, 1, 1, 1), torch.randn(1, 1, 1)),
            (CRIB(1, 1, 1, 1, 2, 1, 1, 0.0), torch.randn(1, 1, 1)),
            (CrossGNN(2, 1, 4, 1, True, 2, 1, True, True, 0.0, 2, 2, 2), torch.randn(1, 2, 4)),
            (FiLM(2, 1, 1, multiscale=(1, 2), order=2, rank=1), torch.randn(1, 2, 1)),
            (FreTS(1, 1, 1, embed_size=2, hidden_size=2), torch.randn(1, 1, 1)),
        )
        for model, values in cases:
            with self.subTest(model=type(model).__module__):
                self.assertEqual(tuple(self.call(model.eval(), values).shape), (1, 1, values.shape[-1]))


if __name__ == "__main__":
    unittest.main()
