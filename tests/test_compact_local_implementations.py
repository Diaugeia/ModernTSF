"""Paper-equation checks for the compact local forecasting implementations."""

from __future__ import annotations

import unittest

import torch

from tsflab.models.cyclenet.model import Model as CycleNet
from tsflab.models.dlinear.model import Model as DLinear
from tsflab.models.fits.model import ComplexFrequencyInterpolation, Model as FITS
from tsflab.models.linear.model import Model as Linear
from tsflab.models.nlinear.model import Model as NLinear
from tsflab.models.segrnn.model import Model as SegRNN
from tsflab.models.sparsetsf.model import Model as SparseTSF


class CompactLocalEquationTests(unittest.TestCase):
    def test_linear_is_the_paper_temporal_affine_equation(self) -> None:
        model = Linear(c_in=2, seq_len=3, pred_len=2)
        with torch.no_grad():
            model.projection.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
            model.projection.linear.bias.copy_(torch.tensor([0.5, -0.5]))
        values = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])
        expected = torch.einsum("btc,ot->boc", values, model.projection.linear.weight)
        expected = expected + model.projection.linear.bias[None, :, None]
        torch.testing.assert_close(model(values), expected)

    def test_nlinear_restores_the_last_observation(self) -> None:
        model = NLinear(c_in=2, seq_len=4, pred_len=3)
        with torch.no_grad():
            model.projection.linear.weight.zero_()
            model.projection.linear.bias.zero_()
        values = torch.randn(2, 4, 2)
        expected = values[:, -1:, :].expand(-1, 3, -1)
        torch.testing.assert_close(model(values), expected)

    def test_dlinear_is_seasonal_plus_trend_forecasting(self) -> None:
        model = DLinear(c_in=1, seq_len=5, pred_len=2, kernel_size=3)
        values = torch.arange(5.0).reshape(1, 5, 1)
        seasonal, trend = model.backbone.decomposition(values)
        expected = model.backbone.seasonal_projection(seasonal.transpose(1, 2))
        expected = expected + model.backbone.trend_projection(trend.transpose(1, 2))
        torch.testing.assert_close(model(values), expected.transpose(1, 2))

    def test_fits_complex_interpolation_obeys_complex_affine_algebra(self) -> None:
        layer = ComplexFrequencyInterpolation(1, 1, channels=1, individual=False)
        with torch.no_grad():
            layer.real_weight.fill_(2.0)
            layer.imag_weight.fill_(3.0)
            layer.real_bias.fill_(5.0)
            layer.imag_bias.fill_(7.0)
        value = torch.tensor([[[complex(11.0, 13.0)]]])
        expected = torch.tensor([[[complex(11 * 2 - 13 * 3 + 5, 11 * 3 + 13 * 2 + 7)]]])
        torch.testing.assert_close(layer(value), expected)
        model = FITS(seq_len=8, pred_len=4, enc_in=1, cut_freq=2)
        self.assertEqual(model.input_bins, 2)
        self.assertLessEqual(model.output_bins, model.total_len // 2 + 1)

    def test_sparsetsf_interleaves_phase_aligned_subsequences(self) -> None:
        model = SparseTSF(seq_len=8, pred_len=8, enc_in=1, period=2)
        with torch.no_grad():
            model.aggregation.weight.zero_()
            model.aggregation.bias.zero_()
            model.forecaster.weight.copy_(torch.eye(4))
            model.forecaster.bias.zero_()
        values = torch.arange(8.0).reshape(1, 8, 1)
        torch.testing.assert_close(model(values), values)

    def test_segrnn_parallel_decoder_has_no_cross_segment_recursion(self) -> None:
        torch.manual_seed(7)
        model = SegRNN(seq_len=8, pred_len=8, enc_in=2, d_model=8, seg_len=2, dropout=0.0)
        model.eval()
        values = torch.randn(1, 8, 2)
        original = model(values)
        with torch.no_grad():
            model.relative_position[0].add_(1.0)
        changed = model(values)
        self.assertFalse(torch.equal(original[:, :2], changed[:, :2]))
        torch.testing.assert_close(original[:, 2:], changed[:, 2:])

    def test_cyclenet_restores_the_aligned_future_cycle(self) -> None:
        model = CycleNet(seq_len=4, pred_len=4, enc_in=1, cycle=4, use_revin=False)
        with torch.no_grad():
            model.cycle_pattern[:, 0].copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
            model.backbone.linear.weight.zero_()
            model.backbone.linear.bias.zero_()
        marks = torch.zeros(1, 4, 6)
        marks[:, -1, 4] = 3
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 4, 1)
        torch.testing.assert_close(model(torch.zeros(1, 4, 1), marks), expected)


if __name__ == "__main__":
    unittest.main()
