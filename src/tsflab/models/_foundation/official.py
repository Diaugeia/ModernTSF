"""Offline adapters around official Chronos, TimesFM, and Moirai APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from tsflab.models._foundation.runtime import (
    FoundationDependencyError,
    FoundationForecast,
    FoundationSource,
)


def _local_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"foundation checkpoint path does not exist: {resolved}")
    return resolved


def _optional_import(message: str, importer: Callable[[], Any]) -> Any:
    try:
        return importer()
    except (ImportError, ModuleNotFoundError) as exc:
        raise FoundationDependencyError(message) from exc


class ChronosRuntime:
    """Normalize an official ``BaseChronosPipeline`` without network access."""

    def __init__(self, source: FoundationSource, pipeline: Any) -> None:
        self.source = source
        self.pipeline = pipeline

    @classmethod
    def from_local(
        cls,
        source: FoundationSource,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        loader: Any | None = None,
    ) -> "ChronosRuntime":
        if loader is None:
            def import_loader():
                from chronos import BaseChronosPipeline

                return BaseChronosPipeline

            loader = _optional_import(
                "Install the official chronos-forecasting package to use Chronos.",
                import_loader,
            )
        pipeline = loader.from_pretrained(
            str(_local_path(checkpoint)),
            revision=source.revision,
            local_files_only=True,
            device_map=device,
            torch_dtype=torch_dtype,
        )
        return cls(source, pipeline)

    def predict(self, context, prediction_length, quantile_levels):
        quantiles, mean = self.pipeline.predict_quantiles(
            inputs=context,
            prediction_length=prediction_length,
            quantile_levels=list(quantile_levels),
        )
        if isinstance(quantiles, list):
            quantiles = torch.stack(quantiles)
        if isinstance(mean, list):
            mean = torch.stack(mean)
        quantiles = torch.as_tensor(quantiles)
        mean = torch.as_tensor(mean)
        if quantiles.ndim == 4 and quantiles.shape[1] == 1:
            quantiles = quantiles[:, 0]
        if mean.ndim == 3 and mean.shape[1] == 1:
            mean = mean[:, 0]
        normalized_quantiles = (
            quantiles if quantile_levels else None
        )
        return FoundationForecast(mean, normalized_quantiles)


class TimesFMRuntime:
    """Normalize the official TimesFM 2.5 PyTorch forecasting API."""

    _LEVELS = tuple(index / 10 for index in range(1, 10))

    def __init__(self, source: FoundationSource, model: Any) -> None:
        self.source = source
        self.model = model

    @classmethod
    def from_local(
        cls,
        source: FoundationSource,
        checkpoint: str | Path,
        *,
        max_context: int,
        max_horizon: int,
        loader: Any | None = None,
        config_factory: Callable[..., Any] | None = None,
    ) -> "TimesFMRuntime":
        if loader is None or config_factory is None:
            def import_timesfm():
                import timesfm

                return timesfm

            timesfm = _optional_import(
                "Install the official timesfm[torch] package to use TimesFM.",
                import_timesfm,
            )
            loader = loader or timesfm.TimesFM_2p5_200M_torch
            config_factory = config_factory or timesfm.ForecastConfig
        model = loader.from_pretrained(
            str(_local_path(checkpoint)),
            revision=source.revision,
            local_files_only=True,
            torch_compile=False,
        )
        model.compile(
            config_factory(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return cls(source, model)

    def predict(self, context, prediction_length, quantile_levels):
        missing = sorted(set(quantile_levels) - set(self._LEVELS))
        if missing:
            raise ValueError(f"TimesFM supports decile quantiles only; unsupported: {missing}")
        point, all_quantiles = self.model.forecast(
            horizon=prediction_length,
            inputs=[row for row in context.detach().cpu().numpy()],
        )
        # Official TimesFM output stores the mean first, followed by deciles.
        indices = [self._LEVELS.index(level) + 1 for level in quantile_levels]
        selected = np.asarray(all_quantiles)[..., indices]
        normalized_quantiles = torch.as_tensor(selected) if quantile_levels else None
        return FoundationForecast(torch.as_tensor(point), normalized_quantiles)


class MoiraiRuntime:
    """Normalize an already constructed official Moirai 2 forecast object.

    Uni2TS currently constrains PyTorch below the version used by TSFLab.
    Construct this object in a compatible provider environment or inject a
    tested official forecast object; the core package deliberately does not
    install or reimplement Uni2TS.
    """

    def __init__(self, source: FoundationSource, model: Any) -> None:
        self.source = source
        self.model = model

    def predict(self, context, prediction_length, quantile_levels):
        configured = int(self.model.hparams.prediction_length)
        if prediction_length != configured:
            raise ValueError(
                f"Moirai runtime horizon is {configured}, requested {prediction_length}"
            )
        raw = np.asarray(
            self.model.predict([row for row in context.detach().cpu().numpy()])
        )
        if raw.ndim == 4 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        if raw.ndim != 3:
            raise ValueError(f"unexpected Moirai forecast shape: {raw.shape}")
        available = tuple(float(level) for level in self.model.module.quantile_levels)
        missing = sorted(set(quantile_levels) - set(available))
        if missing:
            raise ValueError(f"Moirai checkpoint lacks quantiles: {missing}")
        median_index = min(range(len(available)), key=lambda i: abs(available[i] - 0.5))
        mean = torch.as_tensor(raw[:, median_index, :])
        if not quantile_levels:
            return FoundationForecast(mean)
        selected = raw[:, [available.index(level) for level in quantile_levels], :]
        quantiles = torch.as_tensor(selected).permute(0, 2, 1)
        return FoundationForecast(mean, quantiles)
