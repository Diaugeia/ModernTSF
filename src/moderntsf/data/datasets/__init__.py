"""Built-in forecasting dataset implementations."""

from moderntsf.data.datasets.base import ForecastingDataset
from moderntsf.data.datasets.custom import Dataset_Custom
from moderntsf.data.datasets.ett import Dataset_ETT_hour, Dataset_ETT_minute
from moderntsf.data.datasets.periodic_data import Dataset_periodic
from moderntsf.data.datasets.solar import Dataset_Solar
from moderntsf.data.datasets.trend_data import Dataset_trend

__all__ = [
    "ForecastingDataset",
    "Dataset_Custom",
    "Dataset_ETT_hour",
    "Dataset_ETT_minute",
    "Dataset_periodic",
    "Dataset_Solar",
    "Dataset_trend",
]
