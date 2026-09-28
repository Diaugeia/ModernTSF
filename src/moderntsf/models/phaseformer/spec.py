"""Model specification for PhaseFormer."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.phaseformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 16
    dropout: float = 0.0
    period: int = 24
    num_routers: int = 4
    num_layers: int = 1
    num_heads: int = 1
    use_revin: bool = True


def build_model(cfg, params):
    """Construct PhaseFormer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 16),
        dropout=params.get("dropout", 0.0),
        period=params.get("period", 24),
        num_routers=params.get("num_routers", 4),
        num_layers=params.get("num_layers", 1),
        num_heads=params.get("num_heads", 1),
        use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name='PhaseFormer',
    module='moderntsf.models.phaseformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/PhaseFormer.toml',
    model_card='src/moderntsf/models/phaseformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
