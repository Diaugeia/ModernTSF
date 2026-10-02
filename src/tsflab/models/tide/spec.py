"""Model specification for TiDE."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.tide.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    d_model: int = Field(default=512, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_layers: int = Field(default=1, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    decoder_output_dim: int = Field(default=7, gt=0)
    time_feat_dim: int = Field(default=6, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    bias: bool = True
    feature_encode_dim: int = Field(default=2, gt=0)


def build_model(cfg, params):
    """Construct TiDE from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        d_model=params.get('d_model', 512),
        e_layers=params.get('e_layers', 2),
        d_layers=params.get('d_layers', 1),
        d_ff=params.get('d_ff', 2048),
        decoder_output_dim=params.get('decoder_output_dim', 7),
        time_feat_dim=params.get('time_feat_dim', 6),
        dropout=params.get('dropout', 0.1),
        bias=bool(params.get('bias', True)),
        feature_encode_dim=params.get('feature_encode_dim', 2),
    )


SPEC = ModelSpec(
    name='TiDE',
    module='tsflab.models.tide',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TiDE.toml',
    model_card='src/tsflab/models/tide/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
