"""Model specification for MTSMixer."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.mtsmixer.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    d_ff: int = Field(default=4, gt=0)
    e_layers: int = Field(default=2, gt=0)
    fac_T: bool = True
    fac_C: bool = True
    sampling: int = Field(default=2, gt=0)
    norm: bool = True
    individual: bool = False
    rev: bool = True


def build_model(cfg, params):
    """Construct MTSMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        d_ff=params.get("d_ff", 4),
        e_layers=params.get("e_layers", 2),
        fac_T=bool(params.get("fac_T", True)),
        fac_C=bool(params.get("fac_C", True)),
        sampling=params.get("sampling", 2),
        norm=bool(params.get("norm", True)),
        individual=bool(params.get("individual", False)),
        rev=bool(params.get("rev", True)),
    )


SPEC = ModelSpec(
    name='MTSMixer',
    module='moderntsf.models.mtsmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MTSMixer.toml',
    model_card='src/moderntsf/models/mtsmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('channel_wise_linear', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
