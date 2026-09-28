"""Model specification for LSTM."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.lstm.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    init_dim: int = Field(default=32, ge=1)
    hid_dim: int = Field(default=64, ge=1)
    end_dim: int = Field(default=128, ge=1)
    layer: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    cov_dim: int = Field(default=2, ge=0)


def build_model(cfg, params):
    """Construct LSTM from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], init_dim=params.get('init_dim', 32), hid_dim=params.get('hid_dim', 64), end_dim=params.get('end_dim', 128), layer=params.get('layer', 2), dropout=params.get('dropout', 0.1), cov_dim=params.get('cov_dim', 2))
    )


SPEC = ModelSpec(
    name='LSTM',
    module='moderntsf.models.lstm',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/LSTM.toml',
    model_card='src/moderntsf/models/lstm/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
