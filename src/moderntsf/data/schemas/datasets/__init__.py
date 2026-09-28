"""Dataset parameter schemas."""

from moderntsf.data.schemas.datasets.custom import DatasetParameterConfig as CustomConfig
from moderntsf.data.schemas.datasets.ett import DatasetParameterConfig as ETTConfig
from moderntsf.data.schemas.datasets.periodic import DatasetParameterConfig as PeriodicConfig
from moderntsf.data.schemas.datasets.solar import DatasetParameterConfig as SolarConfig
from moderntsf.data.schemas.datasets.trend import DatasetParameterConfig as TrendConfig

__all__ = [
    "CustomConfig",
    "ETTConfig",
    "PeriodicConfig",
    "SolarConfig",
    "TrendConfig",
]
