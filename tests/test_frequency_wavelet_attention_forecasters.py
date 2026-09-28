"""Paper-structure and runtime tests for ReFocus, FreqMoE, SWIFT, and Sensorformer."""

from __future__ import annotations

import copy
import unittest

import torch

from moderntsf.models._components.energy_frequency_pooling import EnergyBasedFrequencyPooling
from moderntsf.models._components.freq_band_moe import FrequencyBandMixtureOfExperts
from moderntsf.models._components.global_patch_compression_attention import (
    GlobalPatchCompressionAttention,
)
from moderntsf.models._components.haar_dwt1d import HaarDWT1D, HaarIDWT1D
from moderntsf.models.freqmoe.model import Model as FreqMoE
from moderntsf.models.freqmoe.spec import ModelParameterConfig as FreqMoEParameters
from moderntsf.models.refocus.model import Model as ReFocus
from moderntsf.models.refocus.spec import ModelParameterConfig as ReFocusParameters
from moderntsf.models.sensorformer.model import Model as Sensorformer
from moderntsf.models.sensorformer.spec import ModelParameterConfig as SensorformerParameters
from moderntsf.models.swift.model import Model as SWIFT
from moderntsf.models.swift.spec import ModelParameterConfig as SWIFTParameters


class ComponentEquationTests(unittest.TestCase):
    def test_energy_based_frequency_pooling_picks_the_higher_energy_token(self):
        pooling = EnergyBasedFrequencyPooling().eval()
        low = torch.complex(torch.tensor([0.1]), torch.tensor([0.0]))
        high = torch.complex(torch.tensor([5.0]), torch.tensor([0.0]))
        spectrum = torch.stack([low, high], dim=0).view(1, 2, 1)
        pooled = pooling(spectrum)
        torch.testing.assert_close(pooled[:, 0, :], high.view(1, 1))
        torch.testing.assert_close(pooled[:, 1, :], high.view(1, 1))

    def test_frequency_band_moe_partitions_are_contiguous_and_gates_sum_to_one(self):
        moe = FrequencyBandMixtureOfExperts(expert_num=3, seq_len=16)
        x = torch.randn(2, 4, 16)
        combined, boundaries, gating = moe(x)
        self.assertEqual(tuple(combined.shape), (2, 4, 16))
        self.assertEqual(boundaries.shape[0], 4)
        self.assertTrue(torch.all(boundaries[:-1] <= boundaries[1:]))
        torch.testing.assert_close(gating.sum(dim=-1), torch.ones(2))

    def test_haar_dwt_round_trip_is_lossless_for_even_and_odd_lengths(self):
        dwt, idwt = HaarDWT1D(), HaarIDWT1D()
        for length in (8, 9):
            x = torch.randn(2, 3, length)
            approx, detail = dwt(x)
            reconstructed = idwt(approx, detail, length=length)
            torch.testing.assert_close(reconstructed, x)

    def test_haar_dwt_matches_closed_form_on_a_known_pair(self):
        dwt = HaarDWT1D()
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])
        approx, detail = dwt(x)
        inv_sqrt2 = 1.0 / (2.0 ** 0.5)
        torch.testing.assert_close(approx, torch.tensor([[[6.0 * inv_sqrt2, 14.0 * inv_sqrt2]]]))
        torch.testing.assert_close(detail, torch.tensor([[[-2.0 * inv_sqrt2, -2.0 * inv_sqrt2]]]))

    def test_global_patch_compression_attention_shapes_and_last_patch_query(self):
        torch.manual_seed(0)
        block = GlobalPatchCompressionAttention(d_model=8, n_heads=2, d_ff=16)
        patches = torch.randn(2, 3, 5, 8)
        out = block(patches)
        self.assertEqual(tuple(out.shape), tuple(patches.shape))
        changed = patches.clone()
        changed[:, :, :-1, :] = torch.randn_like(changed[:, :, :-1, :])
        out_changed = block(changed)
        # Only the non-last patches were perturbed; the compression query (the
        # last patch of each group) is unchanged but stage 2 still lets every
        # patch see the perturbation through the shared key/value, so outputs
        # must differ while shape stays identical.
        self.assertFalse(torch.allclose(out, out_changed))
        self.assertEqual(tuple(out_changed.shape), tuple(patches.shape))


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def factories(length: int = 16, pred: int = 8):
        return {
            "ReFocus": lambda: ReFocus(
                length, pred, 3, d_model=12, d_pick=4, layers=2, dropout=0.0, beta=0.5, kernel_size=3
            ),
            "FreqMoE": lambda: FreqMoE(
                length, pred, 3, expert_num=3, freq_num_blocks=2, dropout_freq=0.0
            ),
            "SWIFT": lambda: SWIFT(
                length, pred, 3, hidden_size=0, conv_kernel=3
            ),
            "Sensorformer": lambda: Sensorformer(
                length, pred, 3, d_model=8, n_heads=2, d_ff=16, layers=2, patch_len=4, stride=2, dropout=0.0
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
        # ReFocus's EKPB uses a stochastic energy-weighted pick during training
        # (matching the official code), so it is excluded from the eval-mode
        # determinism/state-dict-clone comparison below and checked separately.
        # FreqMoE's band_boundaries never receives a gradient: the official
        # code casts it through `.long()` to build integer slice indices,
        # which blocks autograd (see the model card's Differences section).
        no_grad_exceptions = {"FreqMoE": {"moe.band_boundaries"}}
        for name, factory in self.factories().items():
            with self.subTest(model=name):
                model = factory().cpu().train()
                values = torch.randn(2, 16, 3, requires_grad=True)
                output = self.call(model, values)
                self.assertEqual(tuple(output.shape), (2, 8, 3))
                self.assertTrue(torch.isfinite(output).all())
                loss = output.square().mean()
                loss.backward()
                self.assertIsNotNone(values.grad)
                exceptions = no_grad_exceptions.get(name, set())
                for parameter_name, parameter in model.named_parameters():
                    if parameter_name in exceptions:
                        continue
                    self.assertIsNotNone(parameter.grad, f"{name}:{parameter_name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                model.eval()
                expected = self.call(model, values.detach())
                clone = factory().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(self.call(clone, values.detach()), expected)
                batch_one = torch.randn(1, 16, 3)
                self.assertEqual(self.call(model, batch_one).shape[0], 1)
                with self.assertRaises(ValueError):
                    self.call(model, torch.randn(1, 15, 3))
                torch.testing.assert_close(
                    self.call(model, values.detach(), changed_marks=True), expected
                )

    def test_minimum_sequence_boundaries(self):
        cases = (
            ReFocus(3, 2, 1, d_model=2, d_pick=2, layers=1, kernel_size=1),
            FreqMoE(2, 1, 1, expert_num=1, freq_num_blocks=1),
            SWIFT(2, 2, 1, conv_kernel=1),
            Sensorformer(2, 1, 1, d_model=2, n_heads=1, d_ff=2, layers=1, patch_len=1, stride=1),
        )
        for model in cases:
            with self.subTest(model=type(model).__module__):
                values = torch.randn(1, model.seq_len, model.enc_in)
                output = self.call(model.eval(), values)
                self.assertEqual(tuple(output.shape), (1, model.pred_len, model.enc_in))


class SchemaTests(unittest.TestCase):
    def test_parameter_schemas_accept_declared_defaults(self):
        for schema, kwargs in (
            (ReFocusParameters, {"enc_in": 2}),
            (FreqMoEParameters, {"enc_in": 2}),
            (SWIFTParameters, {"enc_in": 2}),
            (SensorformerParameters, {"enc_in": 2}),
        ):
            with self.subTest(schema=schema.__name__):
                schema(**kwargs)

    def test_invalid_architecture_constraints_raise_in_the_model(self):
        invalid = (
            lambda: ReFocus(16, 8, 2, kernel_size=4),
            lambda: FreqMoE(16, 8, 2, expert_num=0),
            lambda: SWIFT(2, 2, 2, conv_kernel=4),
            lambda: Sensorformer(16, 8, 2, d_model=6, n_heads=4),
        )
        for factory in invalid:
            with self.subTest(factory=factory):
                with self.assertRaises(ValueError):
                    factory()


if __name__ == "__main__":
    unittest.main()
