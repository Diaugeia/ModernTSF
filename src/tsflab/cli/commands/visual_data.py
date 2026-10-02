"""Visualize dataset samples from a TOML config."""

from __future__ import annotations

import argparse
import os
import tomllib
from dataclasses import dataclass
from typing import Iterable

import matplotlib
import numpy as np

from pydantic import ValidationError

from tsflab.experiments.config import load_config
from tsflab.catalog.registry.datasets import DATASET_REGISTRY, register_dataset_by_name


def _params_to_dict(params) -> dict:
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return dict(params)


def _select_indices(
    dataset_len: int, num_samples: int, index: int | None, seed: int | None
) -> list[int]:
    if dataset_len <= 0:
        return []
    if index is not None:
        if index < 0 or index >= dataset_len:
            raise ValueError(f"index {index} is out of range (0-{dataset_len - 1})")
        return [index]
    count = min(num_samples, dataset_len)
    if seed is None:
        return list(range(count))
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_len, size=count, replace=False).tolist()


def _normalize_series(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _parse_channels(channels: str | None, total_channels: int) -> list[int]:
    if channels is None or channels == "all":
        return list(range(total_channels))
    tokens = [token.strip() for token in channels.split(",") if token.strip()]
    indices = [int(token) for token in tokens]
    for idx in indices:
        if idx < 0 or idx >= total_channels:
            raise ValueError(
                f"channel index {idx} is out of range (0-{total_channels - 1})"
            )
    return indices


@dataclass(frozen=True)
class VisualDatasetConfig:
    name: str
    alias: str | None
    path: str
    id: str | None
    params: dict


@dataclass(frozen=True)
class VisualTaskConfig:
    seq_len: int
    label_len: int
    pred_len: int
    features: str


@dataclass(frozen=True)
class VisualConfig:
    dataset: VisualDatasetConfig
    task: VisualTaskConfig


def _load_task_defaults() -> dict:
    base_path = os.path.join("configs", "base.toml")
    if os.path.exists(base_path):
        with open(base_path, "rb") as handle:
            base_cfg = tomllib.load(handle)
        return base_cfg.get("task", {})
    return {"seq_len": 96, "label_len": 0, "pred_len": 24, "features": "M"}


def _load_partial_config(path: str) -> VisualConfig:
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)

    if "dataset" not in cfg:
        raise RuntimeError("Dataset config must include a [dataset] section")

    dataset_cfg = cfg["dataset"]
    params = dict(dataset_cfg.get("params", {}))
    dataset = VisualDatasetConfig(
        name=dataset_cfg["name"],
        alias=dataset_cfg.get("alias"),
        path=dataset_cfg.get("path", ""),
        id=dataset_cfg.get("id"),
        params=params,
    )

    task_defaults = _load_task_defaults()
    task_cfg = cfg.get("task", {})
    task = VisualTaskConfig(
        seq_len=int(task_cfg.get("seq_len", task_defaults.get("seq_len", 96))),
        label_len=int(task_cfg.get("label_len", task_defaults.get("label_len", 0))),
        pred_len=int(task_cfg.get("pred_len", task_defaults.get("pred_len", 24))),
        features=task_cfg.get("features", task_defaults.get("features", "M")),
    )
    return VisualConfig(dataset=dataset, task=task)


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


def _title_parts(config, split: str, sample_index: int, dataset) -> Iterable[str]:
    yield f"{config.dataset.alias or config.dataset.name}"
    yield f"{split}"
    yield f"idx={sample_index}"
    if hasattr(dataset, "period"):
        yield f"period={getattr(dataset, 'period')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--config", required=True, type=str, help="Path to config TOML")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test"),
        help="Dataset split",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to plot when --index is not set",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Specific sample index to plot",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="all",
        help="Comma-separated channel indices or 'all'",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output image path (default: work_dirs/plots/<dataset>_<split>.png)",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for sampling"
    )
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        configs = load_config(args.config)
        if not configs:
            raise RuntimeError("No configs loaded from the provided path")
        if len(configs) > 1:
            print(f"Loaded {len(configs)} configs from sweep; using the first one")
        config = configs[0].config
    except ValidationError:
        print("Config is dataset-only; using task defaults for visualization")
        config = _load_partial_config(args.config)

    dataset = _build_dataset(config, args.split)
    indices = _select_indices(len(dataset), args.num_samples, args.index, args.seed)
    if not indices:
        raise RuntimeError("Dataset is empty; no samples to visualize")

    fig, axes = plt.subplots(
        len(indices),
        1,
        figsize=(10, max(3, 3 * len(indices))),
        sharex=False,
    )
    if len(indices) == 1:
        axes = [axes]

    for ax, sample_index in zip(axes, indices):
        input_series, output_series, _, _ = dataset[sample_index]
        input_arr = _normalize_series(input_series)
        output_arr = _normalize_series(output_series)
        pred_arr = output_arr[dataset.label_len :]
        full_arr = np.concatenate([input_arr, pred_arr], axis=0)

        channel_indices = _parse_channels(args.channels, full_arr.shape[1])
        t = np.arange(full_arr.shape[0])
        for ch in channel_indices:
            ax.plot(t, full_arr[:, ch], label=f"ch{ch}")

        ax.axvline(
            dataset.seq_len - 1,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )
        ax.set_title(
            " | ".join(_title_parts(config, args.split, sample_index, dataset))
        )
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.legend(ncol=4, fontsize=8)

    fig.tight_layout()

    save_path = args.save
    if save_path is None:
        save_path = os.path.join(
            "work_dirs", "plots", f"{config.dataset.name}_{args.split}.png"
        )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
