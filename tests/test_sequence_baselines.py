"""Equation and complete runtime tests for six baseline sequence implementations."""

from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.bist.model import Model as BiST
from tsflab.models.deepar.model import Model as DeepAR
from tsflab.models.hl.model import Model as HL
from tsflab.models.lightts.model import Model as LightTS
from tsflab.models.lstm.model import Model as LSTM
from tsflab.models.wavenet.model import GatedCausalLayer, Model as WaveNet


def raw_marks(batch: int, steps: int, offset: int = 0) -> torch.Tensor:
    rows = [
        [2026, 8, 1 + index // 24, 5, (index + offset) % 24, 0]
        for index in range(steps)
    ]
    return torch.tensor([rows] * batch, dtype=torch.float32)


def factories(length: int = 8, horizon: int = 3, channels: int = 4):
    return {
        "BiST": lambda: BiST(
            length,
            horizon,
            channels,
            model_dim=8,
            prompt_dim=4,
            num_layers=1,
            kernel_size=1 if length == 1 else 3,
            residual_steps=1,
            graph_dim=4,
            virtual_clusters=2,
        ),
        "DeepAR": lambda: DeepAR(
            length,
            horizon,
            channels,
            embedding_size=4,
            hidden_size=8,
            num_layers=1,
            cov_feat_size=2,
            dropout=0,
        ),
        "HL": lambda: HL(length, horizon, channels),
        "LSTM": lambda: LSTM(
            length,
            horizon,
            channels,
            init_dim=4,
            hid_dim=8,
            end_dim=8,
            layer=1,
            dropout=0,
            cov_dim=2,
        ),
        "LightTS": lambda: LightTS(
            length,
            horizon,
            channels,
            hid_dim=16,
            chunk_size=1 if length == 1 else 2,
        ),
        "WaveNet": lambda: WaveNet(
            length,
            horizon,
            channels,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=4,
            end_channels=8,
            blocks=1,
            layers=2,
        ),
    }


def model_inputs(name: str, batch: int = 2, length: int = 8, channels: int = 4):
    x = torch.randn(batch, length, channels)
    if name in {"BiST", "LSTM"}:
        return x, {"x_mark_enc": raw_marks(batch, length)}
    if name == "DeepAR":
        return x, {
            "x_mark_enc": raw_marks(batch, length)[..., :2],
            "x_mark_dec": torch.randn(batch, 3, 2),
        }
    return x, {}


class PaperEquationTests(unittest.TestCase):
    def test_bist_decomposition_prompt_residual_and_diffusion(self) -> None:
        model = factories()["BiST"]()
        x = torch.randn(2, 8, 4)
        stable, trend = model.decomposition(x)
        torch.testing.assert_close(stable + trend, x)
        memberships = (
            model.node_queries @ model.cluster_keys.T
            / model.node_queries.shape[-1] ** 0.5
        ).softmax(-1)
        torch.testing.assert_close(memberships.sum(-1), torch.ones(4))
        self.assertEqual(model._adaptive_kernel().shape, (4, 4))
        self.assertEqual(len(model.forward_layers), len(model.residual_layers))

    def test_deepar_gaussian_factorization(self) -> None:
        model = factories()["DeepAR"]().eval()
        x, kwargs = model_inputs("DeepAR")
        output = model(x, **kwargs)
        self.assertEqual(output.shape, (2, 3, 4, 2))
        self.assertTrue((output[..., 1] > 0).all())
        changed = dict(kwargs)
        changed["x_mark_dec"] = kwargs["x_mark_dec"] + 5
        self.assertGreater((model(x, **changed) - output).abs().max().item(), 0)

    def test_historical_last_is_exact_persistence(self) -> None:
        x = torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4)
        expected = x[:, -1:].expand(-1, 3, -1)
        torch.testing.assert_close(factories()["HL"]()(x), expected)

    def test_lstm_uses_shared_per_node_recurrence(self) -> None:
        model = factories()["LSTM"]()
        self.assertIsInstance(model.recurrent, torch.nn.LSTM)
        self.assertEqual(model.recurrent.input_size, 4)
        self.assertEqual(model.forecast[-1].out_features, 3)

    def test_lightts_sampling_matches_equations_one_and_two(self) -> None:
        model = factories()["LightTS"]()
        series = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        torch.testing.assert_close(
            model.sample_continuous(series),
            torch.tensor([[[0, 2, 4, 6], [1, 3, 5, 7]]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            model.sample_interval(series),
            torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=torch.float32),
        )

    def test_wavenet_layer_is_strictly_causal(self) -> None:
        torch.manual_seed(22)
        layer = GatedCausalLayer(2, 2, 3, kernel_size=2, dilation=2).eval()
        first = torch.randn(1, 2, 8)
        changed = first.clone()
        changed[..., 6:] += 100
        first_residual, first_skip = layer(first)
        changed_residual, changed_skip = layer(changed)
        torch.testing.assert_close(first_residual[..., :6], changed_residual[..., :6])
        torch.testing.assert_close(first_skip[..., :6], changed_skip[..., :6])


class CompleteRuntimeTests(unittest.TestCase):
    def test_forward_backward_active_gradients_and_round_trip(self) -> None:
        torch.manual_seed(260827)
        for name, factory in factories().items():
            with self.subTest(model=name):
                model = factory().cpu().eval()
                x, kwargs = model_inputs(name)
                x.requires_grad_(True)
                output = model(x, **kwargs)
                expected = (2, 3, 4, 2) if name == "DeepAR" else (2, 3, 4)
                self.assertEqual(output.shape, expected)
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
                torch.testing.assert_close(clone(x.detach(), **kwargs), model(x.detach(), **kwargs))

    def test_batch_sequence_and_channel_boundaries(self) -> None:
        for name, factory in factories(length=1, horizon=1, channels=1).items():
            with self.subTest(model=name):
                model = factory().cpu().eval()
                x = torch.randn(1, 1, 1)
                kwargs = {}
                if name in {"BiST", "LSTM"}:
                    kwargs["x_mark_enc"] = raw_marks(1, 1)
                if name == "DeepAR":
                    kwargs = {
                        "x_mark_enc": raw_marks(1, 1)[..., :2],
                        "x_mark_dec": torch.randn(1, 1, 2),
                    }
                expected = (1, 1, 1, 2) if name == "DeepAR" else (1, 1, 1)
                self.assertEqual(model(x, **kwargs).shape, expected)

        for name, factory in factories().items():
            with self.subTest(model=name, case="wrong-length"):
                with self.assertRaises(ValueError):
                    factory()(torch.randn(1, 7, 4))
            with self.subTest(model=name, case="wrong-channel"):
                with self.assertRaises(ValueError):
                    factory()(torch.randn(1, 8, 3))

    def test_marks_contract_is_explicit(self) -> None:
        x = torch.randn(2, 8, 4)
        for name in ("BiST", "LSTM"):
            model = factories()[name]().eval()
            first = model(x, raw_marks(2, 8))
            second = model(x, raw_marks(2, 8, offset=9))
            self.assertGreater((first - second).abs().max().item(), 0, name)
        for name in ("HL", "LightTS", "WaveNet"):
            model = factories()[name]().eval()
            torch.testing.assert_close(
                model(x, raw_marks(2, 8)), model(x, raw_marks(2, 8, offset=9))
            )

    def test_invalid_configurations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LightTS(7, 3, 4, hid_dim=16, chunk_size=2)
        with self.assertRaises(ValueError):
            BiST(8, 3, 4, kernel_size=2)
        with self.assertRaises(ValueError):
            DeepAR(8, 3, 4, cov_feat_size=-1)


if __name__ == "__main__":
    unittest.main()
