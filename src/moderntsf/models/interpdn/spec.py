"""Model specification for InterPDN."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.interpdn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    support_size: int = 31
    support_bound: float = 4.0
    ema_decay: float = 0.8
    use_revin: bool = True


def build_model(cfg, params):
    """Construct InterPDN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        support_size=params.get("support_size", 31),
        support_bound=params.get("support_bound", 4.0),
        ema_decay=params.get("ema_decay", 0.8),
        use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name='InterPDN',
    module='moderntsf.models.interpdn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/InterPDN.toml',
    model_card='src/moderntsf/models/interpdn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
