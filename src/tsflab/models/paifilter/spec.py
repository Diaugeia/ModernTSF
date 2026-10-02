"""Model specification for PaiFilter."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.paifilter.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 256


def build_model(cfg, params):
    """Construct PaiFilter from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], hidden_size=params.get('hidden_size', 256))
    )


SPEC = ModelSpec(
    name='PaiFilter',
    module='tsflab.models.paifilter',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/PaiFilter.toml',
    model_card='src/tsflab/models/paifilter/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
