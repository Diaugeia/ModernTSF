"""Model specification for UMixer."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.umixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct UMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 64), e_layers=params.get('e_layers', 2), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='UMixer',
    module='moderntsf.models.umixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/UMixer.toml',
    model_card='src/moderntsf/models/umixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
