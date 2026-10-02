"""CoRe equation checks and the SpectralDescriptor component contract."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from moderntsf.models._components.spectral_descriptor import SpectralDescriptor
from moderntsf.models.core.model import Model as CoRe


def _oracle_descriptor(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Independent float64 implementation of paper Eqs. 10-17 for one window."""
    x = window - window.mean(axis=0, keepdims=True)
    power = (np.abs(np.fft.rfft(x, axis=0)) ** 2).mean(axis=1)
    k = power.shape[0]
    p = (power + eps) / (power.sum() + k * eps)
    entropy = -(p * np.log(p + eps)).sum() / math.log(max(k, 2))
    a, b = k // 3, 2 * k // 3
    return np.array([entropy, p[: a + 1].sum(), p[a + 1 : b + 1].sum(), p[b + 1 :].sum()])


class SpectralDescriptorTests(unittest.TestCase):
    def test_matches_independent_oracle_for_odd_and_even_lengths(self) -> None:
        torch.manual_seed(0)
        for length in (1, 2, 5, 8, 33, 96):
            x = torch.randn(3, length, 4, dtype=torch.float64)
            got = SpectralDescriptor()(x).numpy()
            want = np.stack([_oracle_descriptor(w.numpy()) for w in x])
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, err_msg=str(length))

    def test_bands_partition_unit_mass_and_entropy_is_normalized(self) -> None:
        out = SpectralDescriptor()(torch.randn(5, 64, 3))
        torch.testing.assert_close(out[:, 1:].sum(dim=1), torch.ones(5))
        self.assertTrue(((out[:, 0] >= 0) & (out[:, 0] <= 1 + 1e-6)).all())

    def test_pure_tone_is_low_entropy_in_its_band(self) -> None:
        t = torch.arange(96, dtype=torch.float32)
        low = torch.sin(2 * math.pi * 2 * t / 96)[None, :, None]
        high = torch.sin(2 * math.pi * 40 * t / 96)[None, :, None]
        d_low, d_high = SpectralDescriptor()(low)[0], SpectralDescriptor()(high)[0]
        self.assertLess(d_low[0].item(), 0.3)
        self.assertGreater(d_low[1].item(), 0.99)
        self.assertGreater(d_high[3].item(), 0.99)

    def test_invariance_and_contract_errors(self) -> None:
        x = torch.randn(2, 16, 3)
        module = SpectralDescriptor()
        torch.testing.assert_close(module(x + 5.0), module(x), atol=1e-4, rtol=0)
        self.assertEqual(len(list(module.parameters())), 0)
        with self.assertRaises(ValueError):
            module(torch.randn(16, 3))
        with self.assertRaises(ValueError):
            SpectralDescriptor(eps=0.0)


class CoReEquationTests(unittest.TestCase):
    def _model(self, **kw) -> CoRe:
        return CoRe(8, 3, 2, context_len=2, **kw)

    def test_initialization_matches_paper(self) -> None:
        model = self._model()
        self.assertTrue((model.gate == 0).all())
        torch.testing.assert_close(model.scr_gate.weight, torch.zeros(2, 4))
        torch.testing.assert_close(model.scr_gate.bias, torch.full((2,), -1.0))
        self.assertEqual(model.scr_down.weight.shape, (2, 6))  # rank r = C
        self.assertEqual(model.scr_up.weight.shape, (3, 2))
        torch.testing.assert_close(model.scr_down.bias, torch.zeros(2))
        # Gate is zero at init, so the adapted forecast equals the frozen base.
        x = torch.randn(2, 8, 2)
        torch.testing.assert_close(model(x), x[:, -1:, :].expand(-1, 3, -1))

    def test_exact_correction_space_equations(self) -> None:
        torch.manual_seed(3)
        model = self._model(var_wise_gating=False, scr_rank=3)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.copy_(torch.randn_like(p) * 0.5)
        base = torch.randn(2, 3, 2)
        context = torch.randn(2, 2)
        spectral = torch.rand(2, 4)
        got = model.correct(base, context, spectral)
        # Paper equations written out per variate.
        delta = torch.zeros(2, 3, 2)
        for c in range(2):
            inp = torch.cat((base[:, :, c], context), dim=1)
            lin = model.adapter.linear
            delta[:, :, c] = model.gate.tanh() * (inp @ lin.weight.T + lin.bias)
        alpha = delta.mean(dim=2)  # [B, H]
        want = torch.zeros_like(base)
        for c in range(2):
            z = torch.cat((delta[:, :, c], alpha), dim=1)
            refine = torch.tanh(z @ model.scr_down.weight.T + model.scr_down.bias)
            refine = refine @ model.scr_up.weight.T + model.scr_up.bias
            g = torch.tanh(spectral @ model.scr_gate.weight[c] + model.scr_gate.bias[c])
            want[:, :, c] = base[:, :, c] + delta[:, :, c] + g[:, None] * refine
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)

    def test_cross_variate_mixing_acts_on_corrections_not_predictions(self) -> None:
        torch.manual_seed(5)
        model = self._model(gate_init=0.5)
        with torch.no_grad():
            model.adapter.linears[0].weight[:, :3] = 0  # adapters ignore the base forecast
            model.adapter.linears[1].weight[:, :3] = 0
            model.scr_gate.bias.fill_(0.7)
            model.scr_up.weight.normal_()
        context, spectral = torch.randn(2, 2), torch.rand(2, 4)
        base = torch.randn(2, 3, 2)
        out_a = model.correct(base, context, spectral) - base
        other = base + 10 * torch.randn(2, 3, 2)
        out_b = model.correct(other, context, spectral) - other
        # Backbone errors stay outside the cross-variate operator: the added
        # correction is unchanged when the base forecast changes.
        torch.testing.assert_close(out_b, out_a)
        delta = model.base_correction(base, context)
        torch.testing.assert_close(model.refine(delta, spectral), out_a)

    def test_adaptable_parameters_exclude_frozen_base_and_train(self) -> None:
        model = self._model(gate_init=0.3)
        frozen = {id(p) for p in model.base.parameters()}
        adaptable = model.adaptable_parameters()
        self.assertFalse(frozen & {id(p) for p in adaptable})
        self.assertEqual({id(p) for p in adaptable}, {id(p) for p in model.parameters() if p.requires_grad})
        model(torch.randn(3, 8, 2)).square().sum().backward()
        self.assertGreater(model.gate.grad.abs().max().item(), 0)
        self.assertGreater(model.scr_gate.bias.grad.abs().max().item(), 0)

    def test_external_base_is_detached_and_contract_errors(self) -> None:
        model = self._model()
        base = torch.randn(2, 3, 2, requires_grad=True)
        model.forecast_with_context(torch.randn(2, 8, 2), base_forecast=base).sum().backward()
        self.assertIsNone(base.grad)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 7, 2))
        with self.assertRaises(ValueError):
            model.correct(torch.randn(2, 4, 2), torch.randn(2, 2), torch.rand(2, 4))


if __name__ == "__main__":
    unittest.main()
