"""TSFLab — Agent-first time-series forecasting framework.

Subpackages:
    tsflab.tsf_core   lightweight contract layer (pydantic only)
    tsflab.benchmark  CLI, registries, runner, evaluation, verification
    tsflab.data       dataset providers and loaders
    tsflab.models     flat model catalog (one package per model)
    tsflab.assets     read-only resources bundled into wheels
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tsflab")
except PackageNotFoundError:  # running from a checkout without installation
    __version__ = "0.0.0"
