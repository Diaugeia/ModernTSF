"""UltraTraffic PeMS datasets: hourly flow per station from a local parquet store.

The store is produced by :mod:`tsflab.data.prepare.ultratraffic` from the
UltraTraffic_CL archive (four Caltrans districts, 2003-2023). Two registered
names share this class, mirroring the CauAir pair:

* ``ultratraffic_st`` — spatiotemporal layout: value ``(T, N)`` plus calendar
  covariates ``(T, N, 2)`` in the stamp slots;
* ``ultratraffic_ts`` — plain time-series layout: stations become channels.

The station set is fixed at load time, so windows never change width.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.calendar import node_calendar, time_marks
from tsflab.data.ultratraffic_store import load_panel
from tsflab.data.schemas.datasets.ultratraffic import DatasetParameterConfig

class _UltraTrafficBase(Dataset):
    spatiotemporal = True

    def __init__(self, root_path: str, data_path: str, size: tuple[int, int, int],
                 flag: str = "train", features: str = "M", region: str = "PEMS_SB",
                 years: list[int] | None = None, variant: str = "static", stations: str = "last",
                 split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2), scale: bool = True,
                 calendar: bool = True, max_windows: int | None = None) -> None:
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size
        panel = load_panel(root_path, region, list(years or [2023]), variant, stations)
        values = panel.interpolate(limit_direction="both").fillna(0.0).to_numpy(np.float32)
        total = len(values)
        cuts = np.cumsum(split_ratio) / sum(split_ratio)
        train_end = int(cuts[0] * total)
        if scale:
            train = values[:train_end]
            self.value_mean = float(train.mean())
            self.value_std = float(train.std()) or 1.0
            values = (values - self.value_mean) / self.value_std
        else:
            self.value_mean = self.value_std = None
        self.values = values
        self.covariates = node_calendar(panel.index, values.shape[1]) if calendar else None
        # (T, 6) year/month/day/weekday/hour/minute, as the custom CSV loader emits.
        self.marks = (time_marks(panel.index) if calendar
                      else np.zeros((len(values), 6), np.float32))
        self.num_nodes = values.shape[1]
        self.adj_mx = None  # the archive carries no station coordinates
        start = {"train": 0, "val": train_end, "test": int(cuts[1] * total)}[flag]
        end = {"train": train_end, "val": int(cuts[1] * total), "test": total}[flag]
        first = max(start, self.seq_len - 1) if flag == "train" else start
        centers = np.arange(first, end - self.pred_len)
        if flag != "train":
            centers = centers[centers >= self.seq_len - 1]
        if max_windows is not None and len(centers) > max_windows:
            centers = centers[np.linspace(0, len(centers) - 1, max_windows).astype(np.int64)]
        self.idx = centers

    def __len__(self) -> int:
        return len(self.idx)

    def _window(self, center: int):
        h0 = center - self.seq_len + 1
        return (self.values[h0:center + 1], self.values[center + 1:center + 1 + self.pred_len],
                h0, center)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.value_mean is None:
            return data
        return data * self.value_std + self.value_mean


class Dataset_UltraTraffic_ST(_UltraTrafficBase):
    """Spatiotemporal layout: ``(value_hist, value_fut, cov_hist, cov_fut)``."""

    spatiotemporal = True

    def __getitem__(self, index: int) -> Tuple:
        center = int(self.idx[index])
        hist, fut, h0, _ = self._window(center)
        if self.covariates is None:
            cov = np.zeros((len(self.values), self.num_nodes, 0), np.float32)
        else:
            cov = self.covariates
        return (np.ascontiguousarray(hist), np.ascontiguousarray(fut),
                np.ascontiguousarray(cov[h0:center + 1]),
                np.ascontiguousarray(cov[center + 1:center + 1 + self.pred_len]))


class Dataset_UltraTraffic_TS(_UltraTrafficBase):
    """Plain time-series layout: stations are channels, real calendar marks.

    Marks are ``(year, month, day, weekday, hour, minute)`` per step, built from
    the panel timestamps exactly like the custom CSV loader's marks.
    """

    spatiotemporal = False

    def __getitem__(self, index: int) -> Tuple:
        center = int(self.idx[index])
        hist, fut, h0, _ = self._window(center)
        return (np.ascontiguousarray(hist), np.ascontiguousarray(fut),
                np.ascontiguousarray(self.marks[h0:center + 1]),
                np.ascontiguousarray(self.marks[center + 1:center + 1 + self.pred_len]))


def register() -> None:
    """Register the UltraTraffic spatiotemporal and time-series datasets."""
    DATASET_REGISTRY.register(
        "ultratraffic_st", Dataset_UltraTraffic_ST, DatasetParameterConfig,
        task_modes=frozenset({"spatiotemporal", "covariate"}), storage="directory",
    )
    DATASET_REGISTRY.register(
        "ultratraffic_ts", Dataset_UltraTraffic_TS, DatasetParameterConfig,
        task_modes=frozenset({"time_series"}), storage="directory",
    )
