"""Thin interfaces for official time-series foundation-model runtimes."""

from moderntsf.models._foundation.official import (
    ChronosRuntime,
    MoiraiRuntime,
    TimesFMRuntime,
)
from moderntsf.models._foundation.runtime import (
    FoundationDependencyError,
    FoundationForecast,
    FoundationModel,
    FoundationRuntime,
    FoundationSource,
)

__all__ = [
    "ChronosRuntime",
    "FoundationDependencyError",
    "FoundationForecast",
    "FoundationModel",
    "FoundationRuntime",
    "FoundationSource",
    "MoiraiRuntime",
    "TimesFMRuntime",
]
