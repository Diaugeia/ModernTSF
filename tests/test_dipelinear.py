"""DiPELinear: component contracts, paper equations, and official-code reference values."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from component_reference import assert_reference
from tsflab.catalog.registry.losses import get_loss
from tsflab.experiments.runner.trainer import _forward_training
from tsflab.models._components.fft_extrapolation_conv import FFTExtrapolationConv
from tsflab.models._components.weight_set_router import WeightSetRouter, mix_weight_sets
from tsflab.models.dipelinear.model import Model
from tsflab.models.dipelinear.spec import SPEC

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dipelinear_reference.pt"


class FFTExtrapolationConvTests(unittest.TestCase):
    def test_equals_circular_time_domain_convolution_of_the_padded_history(self) -> None:
        layer = FFTExtrapolationConv(10, 4)
        torch.manual_seed(0)
        with torch.no_grad():
            for p in layer.parameters():
                p.normal_()
        x = torch.randn(2, 3, 10)
        weight = torch.complex(layer.real_weight, layer.imag_weight)[0]
        bias = torch.complex(layer.real_bias, layer.imag_bias)[0]
        kernel = torch.fft.irfft(weight, n=layer.padded_length)
        offset = torch.fft.irfft(bias, n=layer.padded_length)
        padded = torch.nn.functional.pad(x, (layer.guard, 4 - 1 + layer.guard))
        expected = torch.zeros(2, 3, layer.padded_length)
        for t in range(layer.padded_length):  # circular convolution
            for s in range(layer.padded_length):
                expected[..., t] += kernel[(t - s) % layer.padded_length] * padded[..., s]
        expected = expected + offset
        torch.testing.assert_close(
            layer(x), expected[..., -4 - layer.guard : -layer.guard], atol=1e-4, rtol=1e-4
        )

    def test_default_initialization_is_the_dc_average_filter(self) -> None:
        layer = FFTExtrapolationConv(12, 5)
        x = torch.randn(2, 3, 12)
        padded_sum = x.sum(-1, keepdim=True)
        torch.testing.assert_close(layer(x), (padded_sum / layer.padded_length).expand(2, 3, 5))

    def test_mixing_combines_weight_sets_linearly_and_validates_shapes(self) -> None:
        layer = FFTExtrapolationConv(8, 3, num_sets=2)
        with torch.no_grad():
            layer.imag_weight.normal_()
            layer.real_bias.normal_()
        x = torch.randn(2, 4, 8)
        one_hot = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        out = layer(x, one_hot)
        for set_index in (0, 1):
            single = FFTExtrapolationConv(8, 3, num_sets=1)
            with torch.no_grad():
                for name in ("real_weight", "imag_weight", "real_bias", "imag_bias"):
                    getattr(single, name).copy_(getattr(layer, name)[set_index : set_index + 1])
            channels = [c for c in range(4) if one_hot[set_index, c] == 1]
            torch.testing.assert_close(out[:, channels], single(x[:, channels]))
        with self.assertRaises(ValueError):
            layer(x)
        with self.assertRaises(ValueError):
            layer(x, torch.ones(2, 3))
        assert_reference("fft_extrapolation_conv", {"output": out})


class WeightSetRouterTests(unittest.TestCase):
    def test_columns_are_convex_weights_and_temperature_flattens_them(self) -> None:
        router = WeightSetRouter(3, 5)
        sharp, flat = router(0.1), router(1e6)
        torch.testing.assert_close(sharp.sum(0), torch.ones(5))
        torch.testing.assert_close(flat, torch.full((3, 5), 1 / 3), atol=1e-4, rtol=0)
        self.assertGreater(sharp.max().item(), flat.max().item())
        with self.assertRaises(ValueError):
            router(0.0)

    def test_mix_weight_sets_matches_explicit_sum_and_is_differentiable(self) -> None:
        weights = torch.randn(3, 6, requires_grad=True)
        routing = torch.softmax(torch.randn(3, 4), dim=0)
        mixed = mix_weight_sets(weights, routing)
        expected = torch.stack([sum(routing[s, c] * weights[s] for s in range(3)) for c in range(4)])
        torch.testing.assert_close(mixed, expected)
        mixed.sum().backward()
        self.assertTrue(weights.grad.abs().sum() > 0)
        assert_reference("weight_set_router", {"mixed": mixed, "routing": routing})


class DiPELinearTests(unittest.TestCase):
    def test_matches_the_paper_equations_for_one_channel(self) -> None:
        torch.manual_seed(0)
        model = Model(16, 6, 2, use_revin=False, dropout=0.0).eval()
        with torch.no_grad():
            model.freq_weight.normal_(1.0, 0.3)
            model.time_weight.normal_(1.0, 0.3)
            for p in model.mapping.parameters():
                p.add_(0.1 * torch.randn_like(p))
        x = torch.randn(3, 16, 2)
        z = torch.fft.irfft(torch.fft.rfft(x.transpose(1, 2)) * model.freq_weight[0], n=16)  # Eq. 1
        z = z * model.time_weight[0]  # Eq. 2
        pad = model.mapping.guard
        padded = torch.nn.functional.pad(z, (pad, 6 - 1 + pad))
        spec = torch.fft.rfft(padded) * torch.complex(
            model.mapping.real_weight, model.mapping.imag_weight
        )[0] + torch.complex(model.mapping.real_bias, model.mapping.imag_bias)[0]
        y = torch.fft.irfft(spec, n=model.mapping.padded_length)[..., -6 - pad : -pad]  # Eqs. 3-5
        torch.testing.assert_close(model(x), y.transpose(1, 2), atol=1e-5, rtol=1e-5)

    def test_revin_is_equivariant_to_affine_input_changes(self) -> None:
        torch.manual_seed(1)
        model = Model(12, 4, 3).eval()
        with torch.no_grad():
            model.mapping.imag_weight.normal_(0, 0.1)
        x = torch.randn(2, 12, 3)
        torch.testing.assert_close(model(3.0 * x + 5.0), 3.0 * model(x) + 5.0, atol=1e-4, rtol=1e-4)

    def test_matches_official_reference_values(self) -> None:
        reference = torch.load(FIXTURE, map_location="cpu", weights_only=True)
        for case in reference.values():
            length, horizon, channels, experts = (int(v) for v in case["dims"])
            model = Model(length, horizon, channels, experts, loss_alpha=float(case["alpha"]))
            with torch.no_grad():
                model.freq_weight.copy_(case["freq_weight"])
                model.time_weight.copy_(case["time_weight"])
                for name in ("real_weight", "imag_weight", "real_bias", "imag_bias"):
                    getattr(model.mapping, name).copy_(case[name])
                if experts > 1:
                    model.router.logits.copy_(case["logits"])
            model.eval()
            forecast = model(case["x"])
            torch.testing.assert_close(forecast, case["forecast"], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                model.sfa_loss(forecast, case["y"]), case["loss"], atol=1e-5, rtol=1e-5
            )

    def test_sfa_loss_blends_weighted_frequency_l1_and_time_mse(self) -> None:
        model = Model(8, 8, 2, loss_alpha=0.25)
        with torch.no_grad():
            model.freq_weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 2.0]]))
        forecast, target = torch.randn(3, 8, 2), torch.randn(3, 8, 2)
        weight = model.freq_weight[0] / model.freq_weight.abs().mean()
        diff = (torch.fft.rfft(target.transpose(1, 2), norm="ortho")
                - torch.fft.rfft(forecast.transpose(1, 2), norm="ortho")).abs()
        expected = 0.25 * (weight * diff).mean() + 0.75 * (forecast - target).square().mean()
        torch.testing.assert_close(model.sfa_loss(forecast, target), expected)

    def test_frequency_loss_does_not_train_the_attention_weights(self) -> None:
        model = Model(8, 8, 2, loss_alpha=1.0, use_time_w=False, num_experts=2)
        forecast = torch.randn(3, 8, 2, requires_grad=True)
        model.sfa_loss(forecast, torch.randn(3, 8, 2)).backward()
        self.assertIsNone(model.freq_weight.grad)
        self.assertIsNone(model.router.logits.grad)

    def test_temperature_anneals_per_epoch_and_is_final_in_eval(self) -> None:
        model = Model(8, 4, 2, num_experts=2, temperature_start=30.0, anneal_epochs=10)
        self.assertEqual(model.temperature(), 1.0)  # no epoch length known yet
        model.steps_per_epoch.fill_(5)
        self.assertEqual(model.temperature(), 30.0)
        model.train_steps.fill_(5 * 5)
        self.assertAlmostEqual(model.temperature(), 30.0 - 29.0 * 5 / 10)
        model.train_steps.fill_(5 * 11)
        self.assertEqual(model.temperature(), 1.0)
        model.train_steps.zero_()
        self.assertEqual(model.eval().temperature(), 1.0)

    def test_runner_objective_updates_the_step_counter_and_trains_every_group(self) -> None:
        spec = SPEC
        cfg = SimpleNamespace(task=SimpleNamespace(seq_len=24, pred_len=8, label_len=0, features="M"))
        model = spec.build(cfg, {"enc_in": 3, "num_experts": 2}).train()
        spec.training_setup(model, [0] * 4, pred_len=8, features="M")
        self.assertEqual(int(model.steps_per_epoch), 4)
        x, y = torch.randn(4, 24, 3), torch.randn(4, 8, 3)
        outputs, loss = _forward_training(
            model, spec.training_objective, x, None, torch.zeros_like(y), None, y, 8, "M",
            get_loss("mse"),
        )
        loss.backward()
        self.assertEqual(int(model.train_steps), 1)
        self.assertEqual(tuple(outputs.shape), (4, 8, 3))
        for name in ("freq_weight", "time_weight"):
            self.assertGreater(getattr(model, name).grad.abs().sum().item(), 0)
        self.assertGreater(model.router.logits.grad.abs().sum().item(), 0)
        self.assertGreater(model.mapping.real_weight.grad.abs().sum().item(), 0)
        # ``MS`` keeps only the last channel: the loss uses that channel's weights.
        _, ms_loss = _forward_training(
            model, spec.training_objective, x, None, torch.zeros_like(y), None, y, 8, "MS",
            get_loss("mse"),
        )
        self.assertTrue(math.isfinite(float(ms_loss)))

    def test_rejects_wrong_input_and_bad_parameters(self) -> None:
        model = Model(8, 4, 2)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 9, 2))
        with self.assertRaises(ValueError):
            Model(8, 4, 2, loss_alpha=1.5)
        with self.assertRaises(ValueError):
            Model(8, 4, 2, temperature_start=0.5, temperature_end=1.0)


if __name__ == "__main__":
    unittest.main()
