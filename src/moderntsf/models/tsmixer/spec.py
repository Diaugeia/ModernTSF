"""Model specification for TSMixer."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.tsmixer.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


def build_model(cfg, params):
    """Construct TSMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        e_layers=params.get("e_layers", 2),
        dropout=params.get("dropout", 0.1),
    )


SPEC = ModelSpec(
    name='TSMixer',
    module='moderntsf.models.tsmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TSMixer.toml',
    model_card='src/moderntsf/models/tsmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('channel_wise_linear', 'mixer_block'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
