"""Model specification for OccamVTS."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.occamvts.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    patch_len: int = 16
    stride: int = 8
    period: int = 24
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct OccamVTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params["enc_in"],
        d_model=params.get("d_model", 32), patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8), period=params.get("period", 24),
        num_heads=params.get("num_heads", 4), num_layers=params.get("num_layers", 1),
        dropout=params.get("dropout", 0.0), use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name='OccamVTS',
    module='tsflab.models.occamvts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/OccamVTS.toml',
    model_card='src/tsflab/models/occamvts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
