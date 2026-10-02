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


def test_missing_sentinels_are_imputed_causally_before_scaling(tmp_path) -> None:
    n = 100
    ot = np.arange(1, n + 1, dtype=float)
    ot[[0, 1, 50, 51, 99]] = -9999  # leading, interior and trailing gaps
    pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
                  "OT": ot, "a": np.ones(n)}).to_csv(tmp_path / "data.csv", index=False)
    kw = dict(root_path=str(tmp_path), data_path="data.csv", size=(8, 0, 4), features="S", target="OT",
              split_ratio=(1.0, 0.0, 0.0), scale=False)
    raw = Dataset_Custom(**kw)
    clean = Dataset_Custom(**kw, missing_sentinels=[-9999])
    assert raw.data.min() == -9999
    out = clean.data[:, 0]
    assert out[0] == out[1] == 3.0  # back fill only at the series start
    assert out[50] == out[51] == 50.0  # forward fill from the past, never from the future
    assert out[99] == 99.0
    assert out.min() == 3.0
    scaled = Dataset_Custom(**{**kw, "scale": True}, missing_sentinels=[-9999])
    assert abs(scaled.data).max() < 3
