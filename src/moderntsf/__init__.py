"""ModernTSF — Agent-first time-series forecasting framework.

Subpackages:
    moderntsf.tsf_core   lightweight contract layer (pydantic only)
    moderntsf.benchmark  CLI, registries, runner, evaluation, verification
    moderntsf.data       dataset providers and loaders
    moderntsf.models     flat model catalog (one package per model)
    moderntsf.assets     read-only resources bundled into wheels
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("modern-tsf")
except PackageNotFoundError:  # running from a checkout without installation
    __version__ = "0.0.0"
