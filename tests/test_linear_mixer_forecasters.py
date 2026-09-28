"""Equation, structure, and runtime tests for linear/mixer implementations."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from moderntsf.models.crosslinear.model import Model as CrossLinear
from moderntsf.models.mixlinear.model import LowRankSpectralPath, Model as MixLinear
from moderntsf.models.mtsmixer.model import Model as MTSMixer, TemporalSubsequenceMixer
from moderntsf.models.rlinear.model import Model as RLinear
from moderntsf.models.rpmixer.model import Model as RPMixer
from moderntsf.models.tsmixer.model import Model as TSMixer


def _factories():
    return {
        "CrossLinear": (
            lambda: CrossLinear(8, 3, 3, 3, 4, 8, 0.4, 0.6),
            lambda: CrossLinear(1, 1, 2, 1, 2, 2, 0.4, 0.6),
        ),
        "MixLinear": (
            lambda: MixLinear(8, 3, 3, 2, 2, 2, 2),
            lambda: MixLinear(1, 1, 2, 1, 1, 1, 1),
        ),
        "RLinear": (
            lambda: RLinear(3, 8, 3, dropout=0.0),
            lambda: RLinear(2, 1, 1, dropout=0.0),
        ),
        "MTSMixer": (
            lambda: MTSMixer(8, 3, 3, d_model=5, d_ff=2, e_layers=1),
            lambda: MTSMixer(
                1, 1, 2, d_model=2, d_ff=1, e_layers=1, sampling=1
            ),
        ),
        "TSMixer": (
            lambda: TSMixer(8, 3, 3, d_model=5, e_layers=1, dropout=0.0),
            lambda: TSMixer(1, 1, 2, d_model=2, e_layers=1, dropout=0.0),
        ),
        "RPMixer": (
            lambda: RPMixer(8, 3, 3, random_dim=2, e_layers=2),
            lambda: RPMixer(1, 1, 2, random_dim=1, e_layers=1),
        ),
    }


class PaperEquationTests(unittest.TestCase):
    def test_crosslinear_alpha_blends_direct_cross_correlation(self) -> None:
        model = CrossLinear(4, 2, 2, 2, 3, 4, 0.25, 0.5)
        x = torch.randn(1, 2, 4)
        embedding = model.cross_embedding
        direct = embedding.direct_map(x)
        torch.testing.assert_close(embedding(x), 0.25 * x + 0.75 * direct)
        self.assertEqual(model.head.patch_count, 2)

    def test_mixlinear_frequency_path_is_rank_constrained_uvf(self) -> None:
        path = LowRankSpectralPath(4, 2)
        x = torch.randn(2, 3, 4)
        spectrum = torch.fft.fft(x, dim=-1)
        analysis = torch.complex(path.analysis_real, path.analysis_imag)
        synthesis = torch.complex(path.synthesis_real, path.synthesis_imag)
        expected = torch.fft.ifft(
            torch.einsum(
                "nr,bcr->bcn",
                synthesis,
                torch.einsum("rn,bcn->bcr", analysis, spectrum),
            ),
            dim=-1,
        ).real
        torch.testing.assert_close(path(x), expected)

    def test_mixlinear_output_adds_segment_and_frequency_paths(self) -> None:
        model = MixLinear(8, 3, 2, 2, 2, 2, 2)
        x = torch.randn(2, 8, 2)
        center = x.mean(1, keepdim=True)
        reduced = F.avg_pool1d((x - center).transpose(1, 2), 2, 2)
        expected = F.interpolate(
            model.segment_path(reduced) + model.spectral_path(reduced),
            size=3,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2) + center
        torch.testing.assert_close(model(x), expected)

    def test_rlinear_is_revin_plus_one_affine_projection(self) -> None:
        model = RLinear(2, 4, 3)
        x = torch.randn(2, 4, 2)
        normalized = model.normalization(x, "norm")
        expected = model.projection(normalized.transpose(1, 2)).transpose(1, 2)
        expected = model.normalization(expected, "denorm")
        torch.testing.assert_close(model(x), expected)

    def test_mtsmixer_interleaved_subsequences_are_independent(self) -> None:
        mixer = TemporalSubsequenceMixer(6, 2, 4, factorized=True)
        base = torch.randn(1, 2, 6)
        changed = base.clone()
        changed[:, :, 0::2] += 5
        original_output = mixer(base)
        changed_output = mixer(changed)
        torch.testing.assert_close(original_output[:, :, 1::2], changed_output[:, :, 1::2])
        self.assertEqual(len(mixer.paths), 2)

    def test_tsmixer_residual_blocks_retain_linear_path(self) -> None:
        model = TSMixer(4, 2, 2, d_model=3, e_layers=1, dropout=0.0)
        block = model.blocks[0]
        with torch.no_grad():
            block.time_projection.weight.zero_()
            block.time_projection.bias.zero_()
            block.feature_in.weight.zero_()
            block.feature_in.bias.zero_()
            block.feature_out.weight.zero_()
            block.feature_out.bias.zero_()
        x = torch.randn(2, 4, 2)
        torch.testing.assert_close(block(x), x)
        expected = model.projection(x.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(model(x), expected)

    def test_rpmixer_has_fixed_distinct_random_projections_and_identity(self) -> None:
        model = RPMixer(4, 2, 3, random_dim=2, e_layers=2)
        named_parameters = dict(model.named_parameters())
        self.assertFalse(any("random_projection.weight" in name for name in named_parameters))
        first = model.blocks[0].random_projection.weight
        second = model.blocks[1].random_projection.weight
        self.assertFalse(torch.equal(first, second))
        block = model.blocks[0]
        with torch.no_grad():
            block.temporal.real_weight.zero_()
            block.temporal.imag_weight.zero_()
            block.spatial_reconstruction.weight.zero_()
            block.spatial_reconstruction.bias.zero_()
        x = torch.randn(2, 3, 4)
        torch.testing.assert_close(block(x), x)


class RuntimeContractTests(unittest.TestCase):
    def test_all_models_complete_runtime_contract(self) -> None:
        torch.manual_seed(260827)
        for name, (factory, boundary_factory) in _factories().items():
            with self.subTest(model=name):
                model = factory().cpu().eval()
                x = torch.randn(2, 8, 3, requires_grad=True)
                marks = torch.randn(2, 8, 4)
                adjacency = torch.eye(3)
                future_marks = torch.randn(2, 3, 4)
                output = model(x, marks, adjacency, future_marks)
                self.assertEqual(tuple(output.shape), (2, 3, 3))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().max().item(), 0.0)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)
                clone = factory().cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach()), output.detach())
                self.assertEqual(tuple(model(torch.randn(1, 8, 3)).shape), (1, 3, 3))
                boundary = boundary_factory().cpu().eval()
                self.assertEqual(tuple(boundary(torch.randn(1, 1, 2)).shape), (1, 1, 2))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 7, 3))
                torch.testing.assert_close(
                    model(x.detach(), marks, adjacency, future_marks), model(x.detach())
                )

    def test_invalid_architecture_constraints_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CrossLinear(4, 2, 2, 2, 3, 4, 1.1, 0.5)
        with self.assertRaises(ValueError):
            MixLinear(7, 2, 2, downsample=2, segments=1)
        with self.assertRaises(ValueError):
            RLinear(2, 4, 2, dropout=1.0)
        with self.assertRaises(ValueError):
            MTSMixer(4, 2, 2, d_model=3, d_ff=2, fac_C=True)
        with self.assertRaises(ValueError):
            TSMixer(4, 2, 2, 3, 1, dropout=1.0)
        with self.assertRaises(ValueError):
            RPMixer(4, 2, 2, random_dim=0)


if __name__ == "__main__":
    unittest.main()
