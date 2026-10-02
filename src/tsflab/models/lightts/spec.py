"""Model specification for LightTS."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.lightts.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    hid_dim: int = Field(default=128, ge=16)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    chunk_size: int = Field(default=24, ge=1)


def build_model(cfg, params):
    """Construct LightTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], hid_dim=params.get('hid_dim', 128), dropout=params.get('dropout', 0.0), chunk_size=params.get('chunk_size', 24))
    )


SPEC = ModelSpec(
    name='LightTS',
    module='tsflab.models.lightts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/LightTS.toml',
    model_card='src/tsflab/models/lightts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
