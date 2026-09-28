"""Paper-equation checks for local N-BEATS and N-HiTS."""

from __future__ import annotations

import unittest

import torch

from moderntsf.models.nbeats.model import Model as NBeats, seasonality_basis, trend_basis
from moderntsf.models.nhits.model import Model as NHiTS


class LocalBasisForecasterTests(unittest.TestCase):
    def test_nbeats_trend_basis_is_polynomial(self) -> None:
        basis = trend_basis(4, 3)
        time = torch.arange(4) / 4
        torch.testing.assert_close(basis, torch.stack([torch.ones(4), time, time.square()]))

    def test_nbeats_seasonality_basis_is_fourier(self) -> None:
        basis = seasonality_basis(8, 4)
        torch.testing.assert_close(basis[0], torch.ones(8))
        torch.testing.assert_close(basis[1], torch.zeros(8), atol=1e-6, rtol=0)
        torch.testing.assert_close(basis[2, :4], torch.tensor([1.0, 2**-0.5, 0.0, -(2**-0.5)]),
                                   atol=1e-5, rtol=1e-5)

    def test_nbeats_has_doubly_residual_basis_stack(self) -> None:
        model = NBeats(12, 5, 0, "M", 2, stack_types=("trend", "seasonality", "generic"),
                       nb_blocks_per_stack=1, thetas_dim=(3, 4, 4), hidden_layer_units=16)
        self.assertEqual([block.basis for block in model.blocks],
                         ["trend", "seasonality", "generic"])
        self.assertEqual(model(torch.randn(2, 12, 2)).shape, (2, 5, 2))

    def test_nhits_coarse_coefficients_follow_frequency_schedule(self) -> None:
        model = NHiTS(12, 8, 0, "M", 2,
                      stack_types=["identity"] * 3, n_blocks=[1, 1, 1],
                      mlp_units=[[16, 16]], n_pool_kernel_size=[4, 2, 1],
                      n_freq_downsample=[4, 2, 1], use_norm=True)
        self.assertEqual([block.coefficient_count for block in model.blocks], [2, 4, 8])
        self.assertTrue(all(
            block.backcast_coefficients.out_features == model.seq_len
            for block in model.blocks
        ))
        self.assertEqual(model(torch.randn(2, 12, 2)).shape, (2, 8, 2))


if __name__ == "__main__":
    unittest.main()
