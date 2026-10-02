"""The custom CSV loader resolves the target by name in every feature mode."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsflab.data.datasets.custom import Dataset_Custom


def _csv(tmp_path):
    n = 200
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
        "OT": np.arange(n, dtype=float),          # target not in the last position
        "a": np.ones(n), "b": np.full(n, 2.0),
    })
    frame.to_csv(tmp_path / "data.csv", index=False)
    return str(tmp_path)


def _load(root, features, target="OT"):
    return Dataset_Custom(root, "data.csv", (8, 0, 4), features=features, target=target, scale=False)


def test_ms_moves_target_to_last_channel(tmp_path) -> None:
    root = _csv(tmp_path)
    ms = _load(root, "MS")
    s = _load(root, "S")
    assert np.allclose(ms.data[:, -1], s.data[:, 0])
    assert ms.data.shape[1] == 3


def test_missing_target_fails_with_a_clear_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="target column"):
        _load(_csv(tmp_path), "S", target="missing")
