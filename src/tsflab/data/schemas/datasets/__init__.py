"""Dataset parameter schemas."""

from tsflab.data.schemas.datasets.custom import DatasetParameterConfig as CustomConfig
from tsflab.data.schemas.datasets.ett import DatasetParameterConfig as ETTConfig
from tsflab.data.schemas.datasets.periodic import DatasetParameterConfig as PeriodicConfig
from tsflab.data.schemas.datasets.solar import DatasetParameterConfig as SolarConfig
from tsflab.data.schemas.datasets.trend import DatasetParameterConfig as TrendConfig

__all__ = [
    "CustomConfig",
    "ETTConfig",
    "PeriodicConfig",
    "SolarConfig",
    "TrendConfig",
]
