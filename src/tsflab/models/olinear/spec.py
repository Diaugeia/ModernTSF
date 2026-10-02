"""Model specification for OLinear."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.olinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct OLinear from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 32),
        dropout=params.get("dropout", 0.0),
        use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name='OLinear',
    module='tsflab.models.olinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/OLinear.toml',
    model_card='src/tsflab/models/olinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
