"""Model specification for ImplicitForecaster."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.implicitforecaster.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    frequency_pool: int | None = None
    dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct ImplicitForecaster from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params["enc_in"],
        d_model=params.get("d_model", 64), frequency_pool=params.get("frequency_pool"),
        dropout=params.get("dropout", 0.0), use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name='ImplicitForecaster',
    module='tsflab.models.implicitforecaster',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/ImplicitForecaster.toml',
    model_card='src/tsflab/models/implicitforecaster/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
