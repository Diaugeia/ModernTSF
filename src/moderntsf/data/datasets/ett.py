"""ETT dataset implementations."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import pandas as pd

from moderntsf.data.schemas.datasets.ett import DatasetParameterConfig
from moderntsf.benchmark.registry import DATASET_REGISTRY
from moderntsf.data.datasets.base import ForecastingDataset


class Dataset_ETT_hour(ForecastingDataset):
    """ETT hourly dataset split following the original paper setup."""

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "S",
        target: str = "OT",
        split_ratio: tuple[float, float, float] = (12, 4, 4),
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
        """Read the ETT hourly data and return split series and timestamps."""
        df_raw = pd.read_csv(self.file_path)
        df_raw = df_raw.iloc[:14400, :]  # Follow the original ETT paper setting
        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)

        if features in {"M", "MS"}:
            df_data = cast(pd.DataFrame, df_raw.iloc[:, 1:].copy())
        else:
            df_data = cast(pd.DataFrame, df_raw.loc[:, [target]].copy())

        if scale:
            train_len = int(split_ratio[0] / sum(split_ratio) * num_samples)
            data = self._apply_scaling(df_data.to_numpy(), train_len)
        else:
            data = df_data.to_numpy()

        data = np.asarray(data)

        time_stamp = np.asarray(self._build_time_stamp(df_raw))
        series_data = np.asarray(data[border1:border2])
        time_stamp = time_stamp[border1:border2]
        return cast(np.ndarray, series_data), cast(np.ndarray, time_stamp)


class Dataset_ETT_minute(ForecastingDataset):
    """ETT minute dataset split following the original paper setup."""

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "S",
        target: str = "OT",
        split_ratio: tuple[float, float, float] = (12, 4, 4),
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
        """Read the ETT minute data and return split series and timestamps."""
        df_raw = pd.read_csv(self.file_path)
        df_raw = df_raw.iloc[:57600, :]  # Follow the original ETT paper setting

        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)

        if features in {"M", "MS"}:
            df_data = cast(pd.DataFrame, df_raw.iloc[:, 1:].copy())
        else:
            df_data = cast(pd.DataFrame, df_raw.loc[:, [target]].copy())

        if scale:
            train_len = int(split_ratio[0] / sum(split_ratio) * num_samples)
            data = self._apply_scaling(df_data.to_numpy(), train_len)
        else:
            data = df_data.to_numpy()

        data = np.asarray(data)

        time_stamp = np.asarray(self._build_time_stamp(df_raw))
        series_data = np.asarray(data[border1:border2])
        time_stamp = time_stamp[border1:border2]
        return cast(np.ndarray, series_data), cast(np.ndarray, time_stamp)


def register() -> None:
    """Register ETT datasets by name."""
    modes = frozenset({"time_series"})
    DATASET_REGISTRY.register("ETTh1", Dataset_ETT_hour, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("ETTh2", Dataset_ETT_hour, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("ETTm1", Dataset_ETT_minute, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("ETTm2", Dataset_ETT_minute, DatasetParameterConfig, task_modes=modes)
