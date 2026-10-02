"""Calendar marks helper shared by the UltraTraffic time-series loader."""

import pandas as pd

from tsflab.data.calendar import time_marks


def test_time_marks_layout():
    index = pd.date_range("2023-01-02 05:00", periods=3, freq="h")  # a Monday
    marks = time_marks(index)
    assert marks.shape == (3, 6)
    assert marks[0].tolist() == [2023, 1, 2, 0, 5, 0]
    assert marks[2, 4] == 7
