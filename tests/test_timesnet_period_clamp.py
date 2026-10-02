"""TimesNet clamps detected periods to at least 1."""

from __future__ import annotations

from unittest import mock

import numpy as np
import torch

from tsflab.models.timesnet import model as timesnet_model


def test_timesblock_clamps_zero_period() -> None:
    block = timesnet_model.TimesBlock(8, 2, 4, 8, 2)
    values = torch.randn(2, 8, 4)
    fake = (np.array([0, 4]), torch.ones(2, 2))
    with mock.patch.object(timesnet_model, "dominant_periods", return_value=fake):
        out = block(values)
    assert out.shape == values.shape
    assert torch.isfinite(out).all()
    assert block.last_periods.tolist() == [0, 4]
