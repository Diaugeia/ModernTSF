"""Model specification for RPMixer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.rpmixer.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    random_dim: int = Field(default=4, gt=0)
    e_layers: int = Field(default=3, gt=0)


def build_model(cfg, params):
    """Construct RPMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        random_dim=params.get("random_dim", 4),
        e_layers=params.get("e_layers", 3),
    )


SPEC = ModelSpec(
    name='RPMixer',
    module='tsflab.models.rpmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/RPMixer.toml',
    model_card='src/tsflab/models/rpmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('channel_wise_linear',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
