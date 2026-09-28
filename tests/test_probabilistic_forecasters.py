"""Equation, probabilistic-output, and runtime tests for six implementations."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from moderntsf.models.gaussian_mlp.model import Model as GaussianMLP
from moderntsf.models.mqrnn.model import Model as MQRNN
from moderntsf.models.patchmlp.model import Model as PatchMLP
from moderntsf.models.pws.model import Model as PWS
from moderntsf.models.svtime.model import Model as SVTime
from moderntsf.models.timebase.model import Model as TimeBase, cal_orthogonal_loss
from moderntsf.models.gaussian_mlp.spec import ModelParameterConfig as GaussianSchema
from moderntsf.models.mqrnn.spec import ModelParameterConfig as MQRNNSchema
from moderntsf.models.patchmlp.spec import ModelParameterConfig as PatchMLPSchema
from moderntsf.models.pws.spec import ModelParameterConfig as PWSSchema
from moderntsf.models.svtime.spec import ModelParameterConfig as SVTimeSchema
from moderntsf.models.timebase.spec import ModelParameterConfig as TimeBaseSchema
from pydantic import ValidationError


def factory(name: str, length: int = 8, horizon: int = 3, channels: int = 2):
    if name == "GaussianMLP":
        return GaussianMLP(length, horizon, channels, hidden_size=16, num_layers=2, dropout=0.0)
    if name == "PWS":
        period = 4 if length >= 4 else 1
        return PWS(channels, period, length, horizon, min(2, period), False, False, False, "relu", [4])
    if name == "PatchMLP":
        patches = [4, 4, 2, 2] if length >= 4 else [2, 2, 2, 2]
        width = 32 if length >= 4 else 16
        return PatchMLP(length, horizon, channels, width, 1, False, 3 if length >= 3 else 1, patches)
    if name == "SVTime":
        period = 4 if length >= 4 else 1
        return SVTime(channels, period, length, horizon, min(3, period), False, False, False)
    if name == "TimeBase":
        period = 4 if length >= 4 else 1
        return TimeBase(length, horizon, channels, period, 2 if period > 1 else 1, False, 0.08, True)
    if name == "MQRNN":
        return MQRNN(
            length, horizon, channels, hidden_size=12, context_size=5,
            decoder_hidden=11, future_covariate_size=6, dropout=0.0,
        )
    raise KeyError(name)


NAMES = ("GaussianMLP", "PWS", "PatchMLP", "SVTime", "TimeBase", "MQRNN")


class NativeProbabilisticEquationTests(unittest.TestCase):
    def test_gaussian_parameter_equation_and_positive_scale(self) -> None:
        model = factory("GaussianMLP")
        head = model.parameter_head
        features = torch.randn(2, head.loc_layer.in_features)
        loc, scale = head(features)
        torch.testing.assert_close(loc, head.loc_layer(features))
        torch.testing.assert_close(scale, F.softplus(head.scale_layer(features)) + head.eps)
        self.assertTrue(bool((scale > 0).all()))

    def test_pws_patch_residual_analysis_then_period_map(self) -> None:
        model = PWS(1, 4, 8, 8, 2, False, False, False, "relu", [])
        core = model.model
        for analysis in core.analysis_layers:
            torch.nn.init.zeros_(analysis[0].weight)
            torch.nn.init.zeros_(analysis[0].bias)
        for weighted in core.weighted_sum_layers:
            weighted.weight.data.copy_(torch.eye(2))
            weighted.bias.data.zero_()
        x = torch.arange(8.0).reshape(1, 8, 1)
        torch.testing.assert_close(model(x), x)

    def test_patchmlp_latent_decomposition_and_mixing_equation(self) -> None:
        model = factory("PatchMLP").model.eval()
        latent = model.emb(torch.randn(2, 2, 8))
        residual, smooth = model.decomposition(latent)
        torch.testing.assert_close(residual + smooth, latent)
        residual_layer = model.residual_layers[0]
        torch.testing.assert_close(
            residual_layer(latent),
            residual_layer.norm1(residual_layer.ff1(latent) + latent),
        )
        layer = model.smooth_layers[0]
        temporal = layer.norm1(layer.ff1(latent) + latent)
        channels = layer.ff2(temporal.permute(0, 2, 1)).permute(0, 2, 1)
        expected = layer.norm2(channels * temporal + latent)
        torch.testing.assert_close(layer(latent), expected)

    def test_svtime_patch_map_and_backcast_residual_gate(self) -> None:
        model = SVTime(1, 4, 8, 4, 3, False, False, False).model
        periods = torch.arange(8.0).reshape(1, 2, 4)
        mapped = model.period_map(periods)
        expected = periods.new_empty(1, 3, 4)
        for patch in range(model.period_map.patch_count):
            start = patch * model.patch_size
            stop = min(start + model.patch_size, model.period)
            expected[:, :, start:stop] = (
                torch.einsum(
                    "bnp,no->bop",
                    periods[:, :, start:stop],
                    model.period_map.weight[patch],
                )
            )
        torch.testing.assert_close(mapped, expected)
        history = periods.reshape(1, 8)
        backcast, seasonal = model.period_backcast_forecast(history)
        trend = model.trend_projection(history - backcast.reshape(1, 8))
        combined = torch.sigmoid(model.trend_gate_logit) * trend
        combined = combined + (1 - torch.sigmoid(model.trend_gate_logit)) * seasonal.reshape(1, 4)
        torch.testing.assert_close(model(history.reshape(1, 8, 1)).squeeze(-1), combined)

    def test_timebase_basis_and_orthogonality_equations(self) -> None:
        basis = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        gram = basis.transpose(-2, -1) @ basis
        off_diagonal = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
        torch.testing.assert_close(cal_orthogonal_loss(basis), torch.linalg.matrix_norm(off_diagonal).mean())
        model = TimeBase(7, 5, 2, 4, 2, False, 0.08, True).train()
        output = model(torch.randn(2, 7, 2))
        self.assertEqual(tuple(output.shape), (2, 5, 2))
        self.assertIsNotNone(model.aux_loss)
        self.assertGreaterEqual(float(model.aux_loss), 0.0)

    def test_mqrnn_global_local_context_equations_and_future_covariates(self) -> None:
        model = factory("MQRNN").eval()
        x = torch.randn(2, 8, 2)
        history_marks = torch.randn(2, 8, 6)
        future_marks = torch.randn(2, 3, 6)
        targets = x.permute(0, 2, 1).reshape(4, 8, 1)
        history = history_marks.repeat_interleave(2, dim=0)
        _, (hidden, _) = model.encoder(torch.cat([targets, history], dim=-1))
        horizon_context, common_context = model.decode_contexts(hidden[-1], future_marks)
        local_future = future_marks.repeat_interleave(2, dim=0)
        local_input = torch.cat(
            [horizon_context, common_context[:, None].expand(-1, 3, -1), local_future],
            dim=-1,
        )
        expected = model.quantile_head(model.local_decoder(local_input))
        expected = expected.reshape(2, 2, 3, 9).permute(0, 2, 1, 3)
        actual = model(x, history_marks, None, future_marks)
        torch.testing.assert_close(actual, expected)
        changed = model(x, history_marks, None, future_marks + 1.0)
        self.assertGreater((changed - actual).abs().max().item(), 0.0)
        self.assertTrue(bool((actual[..., 1:] >= actual[..., :-1]).all()))


class NativeProbabilisticRuntimeTests(unittest.TestCase):
    def test_parameter_schemas_reject_invalid_dimensions_and_semantics(self) -> None:
        schemas = (GaussianSchema, PWSSchema, PatchMLPSchema, SVTimeSchema, TimeBaseSchema, MQRNNSchema)
        for schema in schemas:
            with self.subTest(schema=schema.__module__):
                with self.assertRaises(ValidationError):
                    schema.model_validate({"enc_in": 0})
        with self.assertRaises(ValidationError):
            PatchMLPSchema.model_validate({"enc_in": 2, "moving_avg": 4})
        with self.assertRaises(ValidationError):
            PWSSchema.model_validate({"enc_in": 2, "analysis_hidden": [4, 0]})
        with self.assertRaises(ValidationError):
            MQRNNSchema.model_validate({"enc_in": 2, "quantile_levels": [0.5, 0.1]})

    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(260827)
        for name in NAMES:
            with self.subTest(model=name):
                model = factory(name).cpu().eval()
                x = torch.randn(2, 8, 2, requires_grad=True)
                marks = torch.randn(2, 8, 6)
                future = torch.randn(2, 3, 6)
                output = (
                    model(x, marks, None, future)
                    if name == "MQRNN"
                    else model(x, marks, torch.eye(2))
                )
                expected = (2, 3, 2, 2) if name == "GaussianMLP" else (2, 3, 2, 9) if name == "MQRNN" else (2, 3, 2)
                self.assertEqual(tuple(output.shape), expected)
                self.assertTrue(bool(torch.isfinite(output).all()))
                loss = output.square().mean()
                if name == "TimeBase" and model.aux_loss is not None:
                    loss = loss + model.aux_loss
                loss.backward()
                self.assertIsNotNone(x.grad)
                self.assertGreater(x.grad.abs().max().item(), 0.0)
                for parameter_name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)

                clone = factory(name).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                clone_output = (
                    clone(x.detach(), marks, None, future)
                    if name == "MQRNN"
                    else clone(x.detach(), marks, torch.eye(2))
                )
                torch.testing.assert_close(clone_output, output.detach())
                self.assertEqual(factory(name)(torch.randn(1, 8, 2)).shape[0], 1)
                boundary = factory(name, 2 if name == "PatchMLP" else 1, 1, 2)
                boundary_output = boundary(torch.randn(1, 2 if name == "PatchMLP" else 1, 2))
                self.assertEqual(boundary_output.shape[:2], (1, 1))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 7, 2))


if __name__ == "__main__":
    unittest.main()
