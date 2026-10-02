"""Equivalence tests for the batch-7 component extraction.

Compares TimeMixer, NLinear, SegRNN, and CrossGNN against reference outputs
captured from the unmodified models (before ``MovingAverageDecomposition``
was migrated to ``series_decomposition`` and last-value centering was
migrated to ``last_value_center``), stored in
``tests/fixtures/component_extraction_batch7.pt``. Behavior must be exactly
unchanged: identical state_dict keys/values, forward outputs, and input
gradients.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tsflab.models.crossgnn.model import Model as CrossGNNModel
from tsflab.models.nlinear.model import Model as NLinearModel
from tsflab.models.segrnn.model import Model as SegRNNModel
from tsflab.models.timemixer.model import Model as TimeMixerModel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "component_extraction_batch7.pt"


class ComponentExtractionEquivalenceTests(unittest.TestCase):
    """Every migrated consumer must reproduce its pre-refactor reference exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = torch.load(FIXTURES, weights_only=False)

    def _assert_matches(self, name: str, model: torch.nn.Module) -> None:
        reference = self.fixtures[name]
        model.eval()
        state = model.state_dict()
        self.assertEqual(list(state.keys()), reference["state_keys"])
        for key, value in state.items():
            self.assertTrue(
                torch.equal(value, reference["state_dict"][key]),
                f"{name}: parameter {key!r} changed after component extraction",
            )
        x = reference["input"].clone().requires_grad_(True)
        output = model(x)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, reference["output"], atol=1e-6, rtol=0)
        (grad,) = torch.autograd.grad(output.sum(), x)
        torch.testing.assert_close(grad, reference["grad_input"], atol=1e-6, rtol=0)

    def test_timemixer_moving_average_decomposition_matches_reference(self) -> None:
        torch.manual_seed(0)
        model = TimeMixerModel(
            seq_len=16, pred_len=8, enc_in=3, c_out=3, e_layers=1,
            d_model=8, d_ff=16, down_sampling_window=2, down_sampling_layers=2,
            moving_avg=3, top_k=2, dropout=0.0, use_norm=True,
            decomp_method="moving_avg",
        )
        self._assert_matches("timemixer", model)

    def test_nlinear_last_value_center_matches_reference(self) -> None:
        torch.manual_seed(0)
        model = NLinearModel(c_in=3, seq_len=16, pred_len=8, individual=False)
        self._assert_matches("nlinear", model)

    def test_segrnn_last_value_center_matches_reference(self) -> None:
        torch.manual_seed(0)
        model = SegRNNModel(seq_len=16, pred_len=8, enc_in=3, d_model=4, dropout=0.0, seg_len=4)
        self._assert_matches("segrnn", model)

    def test_crossgnn_last_value_center_anti_ood_true_matches_reference(self) -> None:
        torch.manual_seed(0)
        model = CrossGNNModel(
            seq_len=16, pred_len=8, enc_in=4, e_layers=1, anti_ood=True,
            tk=2, scale_number=2, use_tgcn=True, use_ngcn=True, dropout=0.0,
            tvechidden=4, nvechidden=4, hidden=4,
        )
        self._assert_matches("crossgnn_anti_ood_true", model)

    def test_crossgnn_disabled_anti_ood_path_stays_baseline_free_matches_reference(self) -> None:
        torch.manual_seed(0)
        model = CrossGNNModel(
            seq_len=16, pred_len=8, enc_in=4, e_layers=1, anti_ood=False,
            tk=2, scale_number=2, use_tgcn=True, use_ngcn=True, dropout=0.0,
            tvechidden=4, nvechidden=4, hidden=4,
        )
        self._assert_matches("crossgnn_anti_ood_false", model)


if __name__ == "__main__":
    unittest.main()
