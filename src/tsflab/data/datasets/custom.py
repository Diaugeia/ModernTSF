"""Custom CSV dataset implementation."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import pandas as pd

from tsflab.data.schemas.datasets.custom import DatasetParameterConfig
from tsflab.benchmark.registry import DATASET_REGISTRY
from tsflab.data.datasets.base import ForecastingDataset


class Dataset_Custom(ForecastingDataset):
    """Custom dataset for CSV files with a date column."""

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
        """Read custom CSV data and return split series and timestamps."""
        df_raw = pd.read_csv(self.file_path)
        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)

        if target not in df_raw.columns[1:]:
            raise ValueError(
                f"target column {target!r} not found in {self.file_path}; "
                f"available: {list(df_raw.columns[1:6])}... (set dataset.params.target)"
            )
        if features == "M":
            df_data = cast(pd.DataFrame, df_raw.iloc[:, 1:].copy())
        elif features == "MS":
            # The trainer reads the target from the last channel, so place it there.
            covariates = [column for column in df_raw.columns[1:] if column != target]
            df_data = cast(pd.DataFrame, df_raw.loc[:, [*covariates, target]].copy())
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
    """Register the custom dataset."""
    modes = frozenset({"time_series"})
    DATASET_REGISTRY.register("traffic", Dataset_Custom, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("weather", Dataset_Custom, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("electricity", Dataset_Custom, DatasetParameterConfig, task_modes=modes)
    # Generic name so any flat-multivariate CSV config can use name = "custom".
    DATASET_REGISTRY.register("custom", Dataset_Custom, DatasetParameterConfig, task_modes=modes)
