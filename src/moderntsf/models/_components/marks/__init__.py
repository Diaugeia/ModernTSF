"""Shared temporal-mark and spatiotemporal input-contract helpers.

ModernTSF feeds models a 4-tuple ``(x_enc, x_mark_enc, x_dec, x_mark_dec)``:

* ``x_enc``      : ``(B, seq_len, N)``  value tensor (N channels / nodes)
* ``x_mark_enc`` : ``(B, seq_len, 6)``  raw integer time features
                   ``[year, month, day, weekday, hour, minute]``
* ``x_dec``      : ``(B, label_len + pred_len, N)`` decoder value input
* ``x_mark_dec`` : ``(B, label_len + pred_len, 6)`` raw decoder time features

Several spatiotemporal and air-quality methods use the BasicTS / LargeST
tensor convention, where each node carries ``1 + F`` channels:
the measured value plus ``F`` *normalized* calendar features. Following the
user's specification we use the two coarsest-useful features (up to
day-of-week), so ``F = 2``:

* ``time_in_day`` = ``(hour * 60 + minute) / 1440`` in ``[0, 1)``
* ``day_in_week`` = ``weekday / 7``                 in ``[0, 1)``

These helpers convert framework marks into the layout each model family
expects, keeping the tensor contract explicit and consistent.
"""

from __future__ import annotations

import torch


# Index positions inside the raw 6-column time-stamp produced by
# ``ForecastingDataset._build_time_stamp``.
_WEEKDAY = 3
_HOUR = 4
_MINUTE = 5

# Number of stacked calendar features (time-of-day, day-of-week).
TIME_FEATURES = 2

# Widths used by ``Time-Series-Library/layers/Embed.py`` for its continuous
# ``timeF`` embeddings.  These are intentionally separate from ModernTSF's
# six-column raw timestamp contract.
TSLIB_TIME_FEATURE_DIMS = {
    "h": 4,
    "t": 5,
    "s": 6,
    "m": 1,
    "a": 1,
    "w": 2,
    "d": 3,
    "b": 3,
}


def tslib_time_feature_dimension(freq: str) -> int:
    """Return the pinned Time-Series-Library ``timeF`` width for ``freq``."""
    try:
        return TSLIB_TIME_FEATURE_DIMS[freq.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(TSLIB_TIME_FEATURE_DIMS))
        raise ValueError(
            f"unsupported Time-Series-Library frequency {freq!r}; "
            f"expected one of: {supported}"
        ) from error


def _hourly_tslib_time_features(marks: torch.Tensor) -> torch.Tensor:
    """Reproduce the pinned TSLib hourly ``utils.timefeatures`` pipeline."""
    year, month, day = marks[..., 0], marks[..., 1], marks[..., 2]
    weekday, hour = marks[..., 3], marks[..., 4]
    month_offsets = marks.new_tensor(
        [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    )
    month_index = month.long().clamp(1, 12)
    leap = (year.long() % 4 == 0) & (
        (year.long() % 100 != 0) | (year.long() % 400 == 0)
    )
    day_of_year = month_offsets[month_index] + day
    day_of_year = day_of_year + (leap & (month_index > 2)).to(day.dtype)
    return torch.stack(
        (
            hour / 23.0 - 0.5,
            weekday / 6.0 - 0.5,
            (day - 1.0) / 30.0 - 0.5,
            (day_of_year - 1.0) / 365.0 - 0.5,
        ),
        dim=-1,
    )


def adapt_tslib_marks(
    marks: torch.Tensor | None,
    *,
    embed_type: str,
    freq: str,
) -> torch.Tensor | None:
    """Adapt ModernTSF marks to the pinned Time-Series-Library contract.

    The repository runner provides raw ``[year, month, day, weekday, hour,
    minute]`` timestamps.  TSLib's categorical embedding omits ``year`` and
    its default hourly ``timeF`` data pipeline projects four normalized
    features. Already-adapted reference tensors are accepted unchanged so the
    exact backbone remains directly testable.

    Only the hourly continuous transform can be reconstructed exactly from
    the repository's six columns.  Other continuous frequencies must be
    supplied in their already-preprocessed reference width; in particular the
    raw contract has no seconds or ISO-week column.
    """
    if marks is None:
        return None
    if marks.ndim != 3:
        raise ValueError(
            "Time-Series-Library marks must have shape [batch, time, features]"
        )

    if embed_type != "timeF":
        if marks.shape[-1] == 6:
            return marks[..., 1:]
        if marks.shape[-1] == 5:
            return marks
        raise ValueError(
            "categorical Time-Series-Library embeddings expect raw six-column "
            "calendar marks or five preprocessed calendar indices"
        )

    expected = tslib_time_feature_dimension(freq)
    if marks.shape[-1] == expected:
        return marks
    if marks.shape[-1] == 6 and freq.lower() == "h":
        return _hourly_tslib_time_features(marks)
    raise ValueError(
        f"timeF/{freq} expects {expected} preprocessed features; exact conversion "
        "from raw six-column marks is currently defined only for hourly data"
    )


def normalized_time_features(marks: torch.Tensor) -> torch.Tensor:
    """Convert raw integer marks to normalized calendar features.

    Parameters
    ----------
    marks : torch.Tensor
        Raw time features of shape ``(B, T, 6)`` with columns
        ``[year, month, day, weekday, hour, minute]``.

    Returns
    -------
    torch.Tensor
        Normalized features of shape ``(B, T, 2)`` ordered as
        ``[time_in_day, day_in_week]``, both in ``[0, 1)``.
    """
    hour = marks[..., _HOUR]
    minute = marks[..., _MINUTE]
    weekday = marks[..., _WEEKDAY]

    time_in_day = (hour * 60.0 + minute) / 1440.0
    day_in_week = weekday / 7.0
    return torch.stack([time_in_day, day_in_week], dim=-1)


def to_spatiotemporal(values: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
    """Build a ``(B, T, N, 1 + F)`` spatiotemporal tensor.

    The value tensor becomes channel 0; the ``F`` covariate / calendar
    features fill the remaining channels. The covariate source depends on the
    rank of ``marks``:

    * ``marks`` is ``(B, T, N, F)`` — node-structured covariates supplied by a
      spatiotemporal / air-quality dataset; used as-is.
    * ``marks`` is ``(B, T, 6)`` — raw calendar stamps from a forecasting-style
      dataset; converted to ``[time_in_day, day_in_week]`` and broadcast across
      the ``N`` nodes.

    Parameters
    ----------
    values : torch.Tensor
        Value tensor of shape ``(B, T, N)``.
    marks : torch.Tensor
        Either node-structured covariates ``(B, T, N, F)`` or raw calendar
        stamps ``(B, T, 6)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, T, N, 1 + F)``.
    """
    b, t, n = values.shape
    value_channel = values.unsqueeze(-1)  # (B, T, N, 1)
    if marks is not None and marks.dim() == 4:
        # Already node-structured covariates (B, T, N, F).
        feats = marks
    else:
        if marks is None:
            marks = values.new_zeros((b, t, 6))
        feats = normalized_time_features(marks)  # (B, T, F)
        feats = feats.unsqueeze(2).expand(b, t, n, feats.shape[-1])  # (B, T, N, F)
    return torch.cat([value_channel, feats], dim=-1)


def to_calendar_spatiotemporal(
    values: torch.Tensor, marks: torch.Tensor
) -> torch.Tensor:
    """Build ``(B, T, N, 1 + 2)`` = ``[value, time_in_day, day_in_week]``.

    For the *calendar-embedding* models (``BiST`` / ``MAGE`` / ``STOP``), which
    index time-of-day and day-of-week embedding **tables** by these two
    normalized channels. Raw ``(B, T, 6)`` stamps are converted via
    :func:`to_spatiotemporal`; node-structured 4-D covariates are accepted only
    when they already carry exactly ``TIME_FEATURES`` channels (assumed
    ``[time_in_day, day_in_week]``).

    Arbitrary meteorology-style covariates (e.g. ``cauair_st`` with ``F > 2``)
    are **rejected** with a clear error: feeding them as embedding indices is
    undefined and would index out of range. These models therefore support the
    ``spatiotemporal`` mode only with *calendar* covariates (``synthetic_st``)
    or the default ``time_series`` mode — not arbitrary node covariates.

    Raises
    ------
    ValueError
        If ``marks`` is a 4-D node-covariate tensor whose last dim is not
        ``TIME_FEATURES``.
    """
    if marks is not None and marks.dim() == 4 and marks.shape[-1] != TIME_FEATURES:
        raise ValueError(
            "Calendar-embedding models (BiST/MAGE/STOP) support spatiotemporal "
            f"mode with exactly {TIME_FEATURES} calendar covariates "
            "[time_in_day, day_in_week], but received "
            f"{marks.shape[-1]} node covariates. Use a calendar-covariate "
            "dataset (e.g. synthetic_st) or the default time_series mode; "
            "arbitrary covariates (e.g. cauair_st meteorology) are not valid "
            "embedding indices for these models."
        )
    return to_spatiotemporal(values, marks)


def future_time_features(marks: torch.Tensor, n: int) -> torch.Tensor:
    """Build a ``(B, T, N, F)`` tensor of future covariate features.

    Used by air-quality models that consume future covariates on the decoder
    side. The source depends on the rank of ``marks``:

    * ``(B, T, N, F)`` — node-structured future covariates from an
      air-quality dataset; used as-is.
    * ``(B, T, 6)`` — raw future calendar stamps; converted to
      ``[time_in_day, day_in_week]`` and broadcast across ``n`` nodes.

    Parameters
    ----------
    marks : torch.Tensor
        Future covariates ``(B, T, N, F)`` or raw stamps ``(B, T, 6)``.
    n : int
        Number of nodes to broadcast across (raw-stamp case only).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, T, N, F)``.
    """
    if marks is not None and marks.dim() == 4:
        return marks
    feats = normalized_time_features(marks)  # (B, T, F)
    b, t, f = feats.shape
    return feats.unsqueeze(2).expand(b, t, n, f)


def coerce_time_length(marks: torch.Tensor, length: int) -> torch.Tensor:
    """Coerce a mark tensor to an exact temporal length.

    The air-quality models tie their future-covariate block to a fixed
    length (``seq_len`` / ``time_step``). The benchmark's decoder marks have
    length ``label_len + pred_len``, so we take the most recent ``length``
    future steps, repeating the last step if there are too few.

    Parameters
    ----------
    marks : torch.Tensor
        Marks of shape ``(B, L, ...)`` (raw stamps or node-structured
        covariates); only the time axis (dim 1) is adjusted.
    length : int
        Desired temporal length.

    Returns
    -------
    torch.Tensor
        Marks with the time axis coerced to ``length``.
    """
    have = marks.shape[1]
    if have == length:
        return marks
    if have > length:
        return marks[:, -length:]
    pad = marks[:, -1:].expand(
        marks.shape[0], length - have, *marks.shape[2:]
    )
    return torch.cat([marks, pad], dim=1)
