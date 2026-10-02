"""Model specification for HL."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.hl.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """HL has no tunable parameters beyond enc_in (num nodes)."""

    enc_in: int = Field(default=207, ge=1)


def build_model(cfg, params):
    """Construct HL from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'])
    )


SPEC = ModelSpec(
    name='HL',
    module='tsflab.models.hl',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/HL.toml',
    model_card='src/tsflab/models/hl/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=(),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
