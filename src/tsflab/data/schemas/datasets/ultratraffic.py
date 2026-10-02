"""Parameter schema for the UltraTraffic PeMS datasets."""

from typing import Literal

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    """Validated parameters for ``ultratraffic_st`` / ``ultratraffic_ts``.

    Parameters
    ----------
    region : str
        Region directory in the store: ``PEMS_BA`` (District 4), ``PEMS_LA``
        (7), ``PEMS_SAC`` (3), or ``PEMS_SB`` (8).
    years : list[int]
        Years concatenated in order.
    variant : str
        ``static`` (every station observed in a year), or the continual-learning
        splits ``cl_common`` (stations retained from the previous year) and
        ``cl_added`` (stations new that year).
    stations : str
        Channel set across years: ``last`` (stations of the final year, absent
        cells filled) or ``intersection`` (stations present in every year).
    split_ratio : tuple[float, float, float]
        Chronological train/val/test fractions.
    scale : bool
        Z-score with statistics of the training portion.
    calendar : bool
        Attach time-of-day and day-of-week covariates (spatiotemporal layout).
    max_windows : int | None
        Evenly subsample windows per split (smoke runs).
    """

    region: Literal["PEMS_BA", "PEMS_LA", "PEMS_SAC", "PEMS_SB"] = "PEMS_SB"
    years: list[int] = Field(default_factory=lambda: [2023])
    variant: Literal["static", "cl_common", "cl_added"] = "static"
    stations: Literal["last", "intersection"] = "last"
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2)
    scale: bool = True
    calendar: bool = True
    max_windows: int | None = None
