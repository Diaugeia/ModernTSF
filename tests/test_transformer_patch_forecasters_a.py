"""Paper-structure and complete runtime tests for six implementations."""

from __future__ import annotations

import copy
import math
import unittest

import torch
from pydantic import ValidationError

from moderntsf.models.autoformer.model import (
    AutoformerDecoderLayer,
    Model as Autoformer,
    fft_autocorrelation,
)
from moderntsf.models._components.series_decomposition import SeriesDecomposition
from moderntsf.models.autoformer.spec import ModelParameterConfig as AutoformerParameters
from moderntsf.models.fedformer.model import (
    FrequencyEnhancedBlock,
    Model as FEDformer,
    selected_modes,
)
from moderntsf.models.fedformer.spec import ModelParameterConfig as FEDformerParameters
from moderntsf.models.itransformer.model import InvertedEmbedding, Model as ITransformer
from moderntsf.models.itransformer.spec import ModelParameterConfig as ITransformerParameters
from moderntsf.models.patchtst.model import Model as PatchTST, patchify
from moderntsf.models.patchtst.spec import ModelParameterConfig as PatchTSTParameters
from moderntsf.models.timemixer.model import (
    DFTDecomposition,
    Model as TimeMixer,
    PastDecomposableMixing,
    multiscale_lengths,
)
from moderntsf.models.timemixer.spec import ModelParameterConfig as TimeMixerParameters
from moderntsf.models.timesnet.model import Model as TimesNet, dominant_periods
from moderntsf.models.timesnet.spec import ModelParameterConfig as TimesNetParameters


def marks(batch: int, length: int) -> torch.Tensor:
    values = torch.zeros(batch, length, 6)
    values[..., 0] = 2024
    values[..., 1] = 1
    values[..., 2] = torch.arange(1, length + 1)
    values[..., 3] = torch.arange(length) % 7
    values[..., 4] = torch.arange(length) % 24
    return values


class PaperEquationTests(unittest.TestCase):
    def test_autoformer_decomposition_and_fft_delay_equations(self) -> None:
        values = torch.tensor([[[1.0], [3.0], [5.0]]])
        seasonal, trend = SeriesDecomposition(3)(values)
        torch.testing.assert_close(
            trend, torch.tensor([[[5.0 / 3.0], [3.0], [13.0 / 3.0]]])
        )
        torch.testing.assert_close(seasonal + trend, values)

        periodic = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, 4, 1)
        correlation = fft_autocorrelation(periodic, periodic).flatten()
        self.assertEqual(set(torch.topk(correlation, 2).indices.tolist()), {0, 2})

        decoder = AutoformerDecoderLayer(8, 2, 16, 3, 2.0, 0.0, "gelu", 2)
        self.assertEqual(len(decoder.decompositions), 3)
        self.assertEqual(len(decoder.trend_projections), 3)

    def test_fedformer_selected_fourier_equations(self) -> None:
        self.assertEqual(selected_modes(8, 3, "low").tolist(), [0, 1, 2])
        self.assertTrue(torch.equal(selected_modes(8, 3, "random"), selected_modes(8, 3, "random")))
        block = FrequencyEnhancedBlock(2, 1, length=4, modes=1, mode_select="low")
        with torch.no_grad():
            block.input_projection.weight.copy_(torch.eye(2))
            block.input_projection.bias.zero_()
            block.output_projection.weight.copy_(torch.eye(2))
            block.output_projection.bias.zero_()
            block.weight_real.zero_()
            block.weight_real[0, 0].copy_(torch.eye(2))
            block.weight_imag.zero_()
        constant = torch.tensor([[[2.0, -1.0]]]).expand(1, 4, 2)
        torch.testing.assert_close(block(constant), constant)

    def test_patchtst_overlap_and_channel_independence(self) -> None:
        values = torch.arange(1.0, 9.0).reshape(1, 8, 1)
        patches = patchify(values, patch_len=4, stride=2, padding_patch="end")
        self.assertEqual(tuple(patches.shape), (1, 1, 4, 4))
        torch.testing.assert_close(patches[0, 0, -1], torch.tensor([7.0, 8.0, 8.0, 8.0]))

        torch.manual_seed(2023)
        model = PatchTST(2, 8, 3, 4, 2, "end", 1, 8, 2, d_ff=16, norm="LayerNorm", revin=False).eval()
        sample = torch.randn(2, 8, 2)
        torch.testing.assert_close(
            model(sample[:, :, [1, 0]]), model(sample)[:, :, [1, 0]]
        )

    def test_timesnet_period_discovery_equation(self) -> None:
        timeline = torch.arange(16, dtype=torch.float32)
        periodic = torch.sin(2 * math.pi * timeline / 4).reshape(1, 16, 1)
        periods, amplitudes = dominant_periods(periodic, 1)
        self.assertEqual(periods.tolist(), [4])
        self.assertEqual(tuple(amplitudes.shape), (1, 1))

    def test_itransformer_inverts_time_and_variate_axes(self) -> None:
        embedding = InvertedEmbedding(seq_len=8, d_model=4, dropout=0.0)
        tokens = embedding(torch.randn(2, 8, 3), marks(2, 8))
        self.assertEqual(tuple(tokens.shape), (2, 9, 4))

        torch.manual_seed(2024)
        model = ITransformer(8, 3, 2, 8, 2, 1, 16, 0.0, "gelu", False, False).eval()
        sample = torch.randn(2, 8, 2)
        torch.testing.assert_close(
            model(sample[:, :, [1, 0]]), model(sample)[:, :, [1, 0]]
        )

    def test_timemixer_pdm_and_fmm_directions(self) -> None:
        self.assertEqual(multiscale_lengths(8, 2, 2), (8, 4, 2))
        block = PastDecomposableMixing((8, 4, 2), 4, 8, 3, 1, "moving_avg", 0.0)
        self.assertEqual(block.seasonal_bottom_up[0].network[0].in_features, 8)
        self.assertEqual(block.seasonal_bottom_up[0].network[0].out_features, 4)
        self.assertEqual(block.trend_top_down[0].network[0].in_features, 4)
        self.assertEqual(block.trend_top_down[0].network[0].out_features, 8)
        fixture = torch.randn(2, 8, 3)
        seasonal, trend = DFTDecomposition(1)(fixture)
        torch.testing.assert_close(seasonal + trend, fixture)


class ParameterSchemaTests(unittest.TestCase):
    def test_cross_field_and_literal_constraints(self) -> None:
        invalid = (
            lambda: AutoformerParameters(enc_in=2, dec_in=3, c_out=2),
            lambda: FEDformerParameters(enc_in=2, dec_in=2, c_out=2, moving_avg=4),
            lambda: PatchTSTParameters(enc_in=2, d_model=10, n_heads=3),
            lambda: TimesNetParameters(enc_in=2, c_out=1),
            lambda: ITransformerParameters(enc_in=2, d_model=10, n_heads=3),
            lambda: TimeMixerParameters(enc_in=2, c_out=2, down_sampling_layers=0),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory()


class RewriteRuntimeTests(unittest.TestCase):
    @staticmethod
    def factories():
        return {
            "Autoformer": lambda length=8: Autoformer(length, 2, 3, 2, 2, 2, 8, 2, 1, 1, 16, 3, 2.0, 0.0),
            "FEDformer": lambda length=8: FEDformer(length, 2, 3, 2, 2, 2, 8, 2, 1, 1, 16, 3, 0.0, modes=2),
            "PatchTST": lambda length=8: PatchTST(2, length, 3, min(4, length), 2, "end", 1, 8, 2, d_ff=16, norm="LayerNorm"),
            "TimesNet": lambda length=8: TimesNet(length, 0, 3, 2, 2, 8, 1, 16, 0.0, 2, 2),
            "iTransformer": lambda length=8: ITransformer(length, 3, 2, 8, 2, 1, 16, 0.0, "gelu", False, True),
            "TimeMixer": lambda length=8: TimeMixer(length, 3, 2, 2, 1, 8, 16, 2, 2, 3, 1, 0.0, True, "moving_avg"),
        }

    @staticmethod
    def call(model: torch.nn.Module, name: str, x: torch.Tensor, changed: bool = False):
        encoder_marks = marks(x.shape[0], x.shape[1])
        if changed:
            encoder_marks[..., 4] += 6
        decoder_length = 5 if name in {"Autoformer", "FEDformer"} else 3
        decoder = torch.randn(x.shape[0], decoder_length, x.shape[2])
        decoder_marks = marks(x.shape[0], decoder_length)
        if changed:
            decoder_marks[..., 4] += 6
        output = model(x, encoder_marks, decoder, decoder_marks)
        return output[0] if isinstance(output, tuple) else output

    def test_complete_runtime_gradient_serialization_and_marks_contract(self) -> None:
        torch.manual_seed(8191)
        active_marks = {"Autoformer", "FEDformer", "TimesNet", "iTransformer"}
        for name, factory in self.factories().items():
            with self.subTest(model=name):
                model = factory().cpu()
                x = torch.randn(2, 8, 2, requires_grad=True)
                output = self.call(model, name, x)
                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name}:{parameter_name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all(), f"{name}:{parameter_name}")
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, f"{name}:{parameter_name}")

                model.eval()
                expected = self.call(model, name, x.detach())
                clone = factory().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(self.call(clone, name, x.detach()), expected)
                self.assertEqual(tuple(self.call(model, name, torch.randn(1, 8, 2)).shape), (1, 3, 2))
                with self.assertRaises(ValueError):
                    self.call(model, name, torch.randn(1, 7, 2))

                changed = self.call(model, name, x.detach(), changed=True)
                if name in active_marks:
                    self.assertFalse(torch.equal(expected, changed), name)
                else:
                    torch.testing.assert_close(expected, changed)

    def test_minimum_sequence_boundaries(self) -> None:
        cases = {
            "Autoformer": Autoformer(3, 1, 1, 1, 1, 1, 4, 1, 1, 1, 8, 3, 2.0, 0.0),
            "FEDformer": FEDformer(4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 8, 3, 0.0, modes=2),
            "PatchTST": PatchTST(1, 4, 1, 4, 1, "none", 1, 4, 1, d_ff=8, norm="LayerNorm"),
            "TimesNet": TimesNet(4, 0, 1, 1, 1, 4, 1, 8, 0.0, 1, 1),
            "iTransformer": ITransformer(1, 1, 1, 4, 1, 1, 8, 0.0, "gelu", False, True),
            "TimeMixer": TimeMixer(4, 1, 1, 1, 1, 4, 8, 2, 2, 3, 1, 0.0, True, "moving_avg"),
        }
        for name, model in cases.items():
            with self.subTest(model=name):
                length = model.seq_len if hasattr(model, "seq_len") else model.context_window
                x = torch.randn(1, length, 1)
                decoder_length = 2 if name in {"Autoformer", "FEDformer"} else 1
                output = model(x, marks(1, length), torch.zeros(1, decoder_length, 1), marks(1, decoder_length))
                if isinstance(output, tuple):
                    output = output[0]
                self.assertEqual(tuple(output.shape), (1, 1, 1))

    def test_itransformer_attention_return_contract(self) -> None:
        model = ITransformer(8, 3, 2, 8, 2, 2, 16, 0.0, "gelu", True, True)
        forecast, attention = model(torch.randn(2, 8, 2), marks(2, 8))
        self.assertEqual(tuple(forecast.shape), (2, 3, 2))
        self.assertEqual(len(attention), 2)
        self.assertEqual(tuple(attention[0].shape), (2, 2, 8, 8))


if __name__ == "__main__":
    unittest.main()
