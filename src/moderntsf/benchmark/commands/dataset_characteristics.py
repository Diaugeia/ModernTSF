"""Extract TFB-style dataset characteristics from a TOML config.

Standalone analysis layer: loads a dataset split (the same way ``visual_data.py``
does) and computes a handful of per-dataset (and optionally per-channel) time
series characteristics, then prints a table and optionally writes a CSV. This
tool never touches training and has zero runtime dependency for the benchmark.

Characteristics (numpy/scipy only; no statsmodels required):

- ``trend_strength``     : STL-style ``1 - Var(resid) / Var(resid + trend)``
                           using a moving-average trend.
- ``seasonality_strength``: STL-style ``1 - Var(resid) / Var(resid + seasonal)``
                           using a period-averaged seasonal component. Period is
                           taken from ``--period`` or the dominant FFT frequency.
- ``stationarity``       : ADF p-value if statsmodels is installed, otherwise a
                           lightweight rolling mean/variance stability ratio in
                           ``[0, 1]`` (1.0 = perfectly stationary moments).
- ``shifting``           : absolute mean shift between the first and second half,
                           normalised by the series std.
- ``transition``         : lag-1 autocorrelation.
- ``correlation``        : mean absolute pairwise channel correlation (dataset
                           level only; ``n/a`` per channel).
"""

from __future__ import annotations

import argparse
import csv
import os
import tomllib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import ValidationError

from moderntsf.benchmark.config import load_config
from moderntsf.benchmark.registry.datasets import DATASET_REGISTRY, register_dataset_by_name

try:  # optional; never a hard dependency
    from statsmodels.tsa.stattools import adfuller as _adfuller

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - import guard
    _adfuller = None
    _HAS_STATSMODELS = False


# --------------------------------------------------------------------------- #
# Config loading shared with the dataset plotting command.
# --------------------------------------------------------------------------- #
def _params_to_dict(params) -> dict:
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return dict(params)


@dataclass(frozen=True)
class _DatasetConfig:
    name: str
    alias: Optional[str]
    path: str
    id: str | None
    params: dict


@dataclass(frozen=True)
class _TaskConfig:
    seq_len: int
    label_len: int
    pred_len: int
    features: str


@dataclass(frozen=True)
class _PartialConfig:
    dataset: _DatasetConfig
    task: _TaskConfig


def _load_task_defaults() -> dict:
    base_path = os.path.join("configs", "base.toml")
    if os.path.exists(base_path):
        with open(base_path, "rb") as handle:
            base_cfg = tomllib.load(handle)
        return base_cfg.get("task", {})
    return {"seq_len": 96, "label_len": 0, "pred_len": 24, "features": "M"}


def _load_partial_config(path: str) -> _PartialConfig:
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)
    if "dataset" not in cfg:
        raise RuntimeError("Dataset config must include a [dataset] section")
    dataset_cfg = cfg["dataset"]
    dataset = _DatasetConfig(
        name=dataset_cfg["name"],
        alias=dataset_cfg.get("alias"),
        path=dataset_cfg.get("path", ""),
        id=dataset_cfg.get("id"),
        params=dict(dataset_cfg.get("params", {})),
    )
    task_defaults = _load_task_defaults()
    task_cfg = cfg.get("task", {})
    task = _TaskConfig(
        seq_len=int(task_cfg.get("seq_len", task_defaults.get("seq_len", 96))),
        label_len=int(task_cfg.get("label_len", task_defaults.get("label_len", 0))),
        pred_len=int(task_cfg.get("pred_len", task_defaults.get("pred_len", 24))),
        features=task_cfg.get("features", task_defaults.get("features", "M")),
    )
    return _PartialConfig(dataset=dataset, task=task)


def _build_dataset(config, split: str):
    register_dataset_by_name(config.dataset.name)
    spec = DATASET_REGISTRY.get(config.dataset.name)
    dataset_cls = spec.dataset_class
    root_path, data_path = spec.resolve_location(
        config.dataset.path, config.dataset.id
    )
    params = _params_to_dict(config.dataset.params)
    size = (config.task.seq_len, config.task.label_len, config.task.pred_len)
    return dataset_cls(
        root_path=root_path,
        data_path=data_path,
        size=size,
        flag=split,
        features=config.task.features,
        **params,
    )


def _extract_series(dataset) -> np.ndarray:
    """Return the underlying split series as a ``(T, C)`` float64 array."""
    if not hasattr(dataset, "data"):
        raise RuntimeError(
            "Dataset has no `.data` attribute; cannot extract raw series."
        )
    arr = np.asarray(dataset.data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


# --------------------------------------------------------------------------- #
# Characteristic computations (numpy / scipy only)
# --------------------------------------------------------------------------- #
def _dominant_period(x: np.ndarray, max_period: Optional[int] = None) -> int:
    """Estimate the dominant seasonal period via the FFT power spectrum.

    Returns an integer period >= 2. Falls back to 2 when no clear peak exists.
    """
    n = x.size
    if n < 4:
        return 2
    x = x - x.mean()
    if not np.any(x):
        return 2
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Ignore the zero-frequency (DC) component.
    spectrum[0] = 0.0
    cap = max_period if max_period is not None else n // 2
    # Drop ultra-low frequencies whose period exceeds the cap (treated as trend).
    for i in range(1, freqs.size):
        if freqs[i] > 0 and (1.0 / freqs[i]) > cap:
            spectrum[i] = 0.0
    if not np.any(spectrum):
        return 2
    peak = int(np.argmax(spectrum))
    if freqs[peak] <= 0:
        return 2
    period = int(round(1.0 / freqs[peak]))
    return max(2, period)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding (returns same length)."""
    if window < 2:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _seasonal_component(detrended: np.ndarray, period: int) -> np.ndarray:
    """Period-averaged seasonal estimate, broadcast back over the series."""
    n = detrended.size
    if period < 2 or period >= n:
        return np.zeros_like(detrended)
    means = np.zeros(period, dtype=np.float64)
    counts = np.zeros(period, dtype=np.float64)
    idx = np.arange(n) % period
    np.add.at(means, idx, detrended)
    np.add.at(counts, idx, 1.0)
    counts = np.where(counts == 0, 1.0, counts)
    means = means / counts
    means = means - means.mean()  # center so the seasonal sums to ~0
    return means[idx]


def _strength(var_resid: float, var_combined: float) -> float:
    """STL-style strength: ``max(0, 1 - Var(resid) / Var(resid + comp))``."""
    if var_combined <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - var_resid / var_combined)))


def _trend_and_seasonal_strength(x: np.ndarray, period: int) -> tuple[float, float]:
    n = x.size
    if n < 4:
        return 0.0, 0.0
    # Trend via moving average over (roughly) one seasonal cycle.
    trend_window = max(3, period)
    trend = _moving_average(x, trend_window)
    detrended = x - trend
    seasonal = _seasonal_component(detrended, period)
    resid = detrended - seasonal

    # Trend strength: how much variance the moving-average trend removes from x
    # (resid+seasonal == detrended vs the original series).
    trend_strength = _strength(np.var(detrended), np.var(x))
    # Seasonality strength: how much variance the seasonal component removes from
    # the detrended series (resid vs resid+seasonal).
    seasonal_strength = _strength(np.var(resid), np.var(resid + seasonal))
    return trend_strength, seasonal_strength


def _stationarity(x: np.ndarray) -> tuple[float, str]:
    """Stationarity score in ``[0, 1]`` plus a label describing the method.

    Uses the ADF test p-value (statsmodels) when available, otherwise a
    lightweight rolling mean/variance stability ratio. Higher is more stationary.
    """
    n = x.size
    if n < 8:
        return 0.0, "n/a"
    if _HAS_STATSMODELS:
        try:
            pvalue = float(_adfuller(x, autolag="AIC")[1])
            # Map p-value to a 0..1 "stationarity" score (1 - p): small p => stationary.
            return float(max(0.0, min(1.0, 1.0 - pvalue))), "adf(1-pvalue)"
        except Exception:
            pass
    # Lightweight fallback: compare rolling means/vars across windows. A series
    # whose windowed moments barely move scores near 1.0.
    n_windows = min(10, max(2, n // 20))
    splits = np.array_split(x, n_windows)
    means = np.array([s.mean() for s in splits])
    variances = np.array([s.var() for s in splits])
    global_std = x.std()
    if global_std <= 1e-12:
        return 1.0, "rolling(n/a-statsmodels)"
    mean_drift = means.std() / (abs(x.mean()) + global_std + 1e-12)
    var_global = x.var()
    var_drift = variances.std() / (var_global + 1e-12)
    score = 1.0 / (1.0 + mean_drift + var_drift)
    return float(max(0.0, min(1.0, score))), "rolling(n/a-statsmodels)"


def _shifting(x: np.ndarray) -> float:
    """Absolute mean shift between halves, normalised by the series std."""
    n = x.size
    if n < 4:
        return 0.0
    half = n // 2
    first, second = x[:half], x[half:]
    std = x.std()
    if std <= 1e-12:
        return 0.0
    return float(abs(second.mean() - first.mean()) / std)


def _transition(x: np.ndarray) -> float:
    """Lag-1 autocorrelation."""
    n = x.size
    if n < 2:
        return 0.0
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc[:-1], xc[1:]) / denom)


def _channel_correlation(series: np.ndarray) -> float:
    """Mean absolute off-diagonal pairwise channel correlation."""
    if series.shape[1] < 2:
        return float("nan")
    # Guard against zero-variance channels (corrcoef yields NaN rows).
    stds = series.std(axis=0)
    keep = stds > 1e-12
    if keep.sum() < 2:
        return float("nan")
    corr = np.corrcoef(series[:, keep], rowvar=False)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = np.abs(corr[mask])
    off = off[np.isfinite(off)]
    if off.size == 0:
        return float("nan")
    return float(off.mean())


@dataclass(frozen=True)
class Characteristics:
    scope: str  # "dataset" or "ch<i>"
    length: int
    channels: int
    period: int
    trend_strength: float
    seasonality_strength: float
    stationarity: float
    stationarity_method: str
    shifting: float
    transition: float
    correlation: float


def _per_series(x: np.ndarray, period: int, scope: str) -> Characteristics:
    trend_s, seas_s = _trend_and_seasonal_strength(x, period)
    stat_score, stat_method = _stationarity(x)
    return Characteristics(
        scope=scope,
        length=int(x.size),
        channels=1,
        period=int(period),
        trend_strength=trend_s,
        seasonality_strength=seas_s,
        stationarity=stat_score,
        stationarity_method=stat_method,
        shifting=_shifting(x),
        transition=_transition(x),
        correlation=float("nan"),
    )


def compute_characteristics(
    series: np.ndarray, period: Optional[int], per_channel: bool
) -> list[Characteristics]:
    """Compute dataset-level (and optionally per-channel) characteristics."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    n, c = series.shape

    # Dataset-level: average each channel's series for the univariate stats, but
    # report period from the mean-across-channels signal for a stable estimate.
    pooled = series.mean(axis=1)
    ds_period = period if period is not None else _dominant_period(pooled)

    rows: list[Characteristics] = []

    # Aggregate dataset row: average per-channel univariate stats.
    per_ch = [_per_series(series[:, j], ds_period, f"ch{j}") for j in range(c)]
    agg = Characteristics(
        scope="dataset",
        length=n,
        channels=c,
        period=ds_period,
        trend_strength=float(np.mean([r.trend_strength for r in per_ch])),
        seasonality_strength=float(np.mean([r.seasonality_strength for r in per_ch])),
        stationarity=float(np.mean([r.stationarity for r in per_ch])),
        stationarity_method=per_ch[0].stationarity_method if per_ch else "n/a",
        shifting=float(np.mean([r.shifting for r in per_ch])),
        transition=float(np.mean([r.transition for r in per_ch])),
        correlation=_channel_correlation(series),
    )
    rows.append(agg)

    if per_channel:
        for r in per_ch:
            ch_period = period if period is not None else _dominant_period(
                series[:, int(r.scope[2:])]
            )
            if ch_period != r.period:
                r = _per_series(series[:, int(r.scope[2:])], ch_period, r.scope)
            rows.append(r)

    return rows


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
_COLUMNS = [
    "scope",
    "length",
    "channels",
    "period",
    "trend_strength",
    "seasonality_strength",
    "stationarity",
    "stationarity_method",
    "shifting",
    "transition",
    "correlation",
]


def _fmt(value) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def _print_table(rows: list[Characteristics]) -> None:
    table = [[_fmt(getattr(r, col)) for col in _COLUMNS] for r in rows]
    widths = [
        max(len(col), *(len(row[i]) for row in table)) for i, col in enumerate(_COLUMNS)
    ]
    header = "  ".join(col.ljust(widths[i]) for i, col in enumerate(_COLUMNS))
    print(header)
    print("  ".join("-" * widths[i] for i in range(len(_COLUMNS))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(_COLUMNS))))


def _write_csv(path: str, rows: list[Characteristics]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_COLUMNS)
        for r in rows:
            writer.writerow([_fmt(getattr(r, col)) for col in _COLUMNS])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TFB-style dataset characteristics from a TOML config"
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to dataset or run config TOML"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test"),
        help="Dataset split to analyse (default: train)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="Seasonal period; if unset, estimated from the dominant FFT frequency",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Also emit one row per channel (otherwise dataset-level only)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output CSV path (default: "
            "work_dirs/<dataset>/characteristics_<split>.csv)"
        ),
    )
    args = parser.parse_args()

    try:
        configs = load_config(args.config)
        if not configs:
            raise RuntimeError("No configs loaded from the provided path")
        if len(configs) > 1:
            print(f"Loaded {len(configs)} configs from sweep; using the first one")
        config = configs[0].config
    except ValidationError:
        print("Config is dataset-only; using task defaults for analysis")
        config = _load_partial_config(args.config)

    dataset = _build_dataset(config, args.split)
    series = _extract_series(dataset)
    if series.shape[0] < 4:
        raise SystemExit(
            f"Split '{args.split}' has only {series.shape[0]} time steps; "
            "too short to analyse."
        )

    rows = compute_characteristics(series, args.period, args.per_channel)

    name = config.dataset.alias or config.dataset.name
    print(f"Dataset: {name} | split: {args.split}")
    if not _HAS_STATSMODELS:
        print("statsmodels not installed; stationarity uses rolling-moment fallback")
    _print_table(rows)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            "work_dirs", config.dataset.name, f"characteristics_{args.split}.csv"
        )
    _write_csv(out_path, rows)
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
