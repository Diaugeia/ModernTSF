"""Forecast metrics and model-resource profiling."""

from tsflab.benchmark.evaluation.metrics import collect_metrics
from tsflab.benchmark.evaluation.profile import profile_model

__all__ = ["collect_metrics", "profile_model"]
