"""Model specification for TimeFilter."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.timefilter.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    d_ff: int = 128
    e_layers: int = 2
    patch_len: int = 16
    dropout: float = 0.1
    top_p: float = 0.5
    pos: bool = True
    num_experts: int = 4


def build_model(cfg, params):
    """Construct TimeFilter from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 64), d_ff=params.get('d_ff', 128), e_layers=params.get('e_layers', 2), patch_len=params.get('patch_len', 16), dropout=params.get('dropout', 0.1), top_p=params.get('top_p', 0.5), pos=bool(params.get('pos', True)), num_experts=params.get('num_experts', 4))
    )


SPEC = ModelSpec(
    name='TimeFilter',
    module='moderntsf.models.timefilter',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TimeFilter.toml',
    model_card='src/moderntsf/models/timefilter/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
