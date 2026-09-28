"""Parameter contract for pre-windowed NumPy forecasting datasets."""

from moderntsf.data.schemas.base import DatasetParameters


class PreProcessedParameterConfig(DatasetParameters):
    """No dataset-level params; preprocessing is done by `tsf dataset prepare`."""
