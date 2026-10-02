"""Solar loader: opt-in synthetic timestamps."""

import numpy as np

from tsflab.data.datasets.solar import Dataset_Solar


def _write(tmp_path, rows=300):
    path = tmp_path / "s.txt"
    np.savetxt(path, np.random.rand(rows, 2), delimiter=",")
    return str(tmp_path), "s.txt"


def test_default_marks_are_constant(tmp_path):
    root, name = _write(tmp_path)
    ds = Dataset_Solar(root, name, (8, 4, 4), "train", "M", "0")
    assert np.unique(ds.time_stamp[:, 4]).size == 1


def test_start_freq_synthesises_calendar(tmp_path):
    root, name = _write(tmp_path)
    ds = Dataset_Solar(root, name, (8, 4, 4), "train", "M", "0",
                       start="2006-01-01 00:00", freq="10min")
    assert ds.time_stamp[0].tolist() == [2006, 1, 1, 6, 0, 0]
    assert ds.time_stamp[7].tolist()[-2:] == [1, 10]
