"""Solar dataset implementation."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import pandas as pd

from tsflab.data.schemas.datasets.solar import DatasetParameterConfig
from tsflab.benchmark.registry import DATASET_REGISTRY
from tsflab.data.datasets.base import ForecastingDataset


class Dataset_Solar(ForecastingDataset):
    """Solar dataset for CSV-like text files without a date column."""

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "S",
        target: str = "OT",
        split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
        scale: bool = True,
        target_channel: int | None = None,
        norm_each_channel: bool = False,
    ):
        super().__init__(
            root_path,
            data_path,
            size,
            flag,
            features,
            target,
            split_ratio,
            scale,
            target_channel,
            norm_each_channel,
        )

    def _read_data(
        self,
        flag: str,
        features: str,
        target: str,
        split_ratio: tuple[float, float, float],
        scale: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read the Solar data and return split series and timestamps."""
        df_rows = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip("\n").split(",")
                data_line = np.stack([float(i) for i in line])
                df_rows.append(data_line)
        df_raw = np.stack(df_rows, 0)
        df_raw = pd.DataFrame(df_raw)
        df_raw.columns = df_raw.columns.map(str)

        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)

        if features in {"M", "MS"}:
            df_data = cast(pd.DataFrame, df_raw.copy())
        else:
            df_data = cast(pd.DataFrame, df_raw.loc[:, [str(target)]].copy())

        if scale:
            train_len = int(split_ratio[0] / sum(split_ratio) * num_samples)
            data = self._apply_scaling(df_data.to_numpy(), train_len)
        else:
            data = df_data.to_numpy()

        data = np.asarray(data)

        df_stamp = df_raw.copy()
        df_stamp["date"] = 0
        time_stamp = np.asarray(self._build_time_stamp(df_stamp))

        series_data = np.asarray(data[border1:border2])
        time_stamp = time_stamp[border1:border2]
        return cast(np.ndarray, series_data), cast(np.ndarray, time_stamp)


def register() -> None:
    """Register the Solar dataset."""
    DATASET_REGISTRY.register(
        "solar", Dataset_Solar, DatasetParameterConfig,
        task_modes=frozenset({"time_series"}),
    )
