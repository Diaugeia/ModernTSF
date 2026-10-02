"""Model specification for Sonnet."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.sonnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 16
    num_wavelets: int = 4
    dropout: float = 0.0


def build_model(cfg, params):
    """Construct Sonnet from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 16),
        num_wavelets=params.get("num_wavelets", 4),
        dropout=params.get("dropout", 0.0),
    )


SPEC = ModelSpec(
    name='Sonnet',
    module='tsflab.models.sonnet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/Sonnet.toml',
    model_card='src/tsflab/models/sonnet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
