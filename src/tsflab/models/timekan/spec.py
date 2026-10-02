"""Model specification for TimeKAN."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.timekan.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 16
    e_layers: int = 1
    down_sampling_window: int = 2
    down_sampling_layers: int = 1
    begin_order: int = 0
    moving_avg: int = 25
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"
    use_norm: int = 1


def build_model(cfg, params):
    """Construct TimeKAN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out', None), d_model=params.get('d_model', 16), e_layers=params.get('e_layers', 1), down_sampling_window=params.get('down_sampling_window', 2), down_sampling_layers=params.get('down_sampling_layers', 1), begin_order=params.get('begin_order', 0), moving_avg=params.get('moving_avg', 25), dropout=params.get('dropout', 0.1), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'), use_norm=params.get('use_norm', 1))
    )


SPEC = ModelSpec(
    name='TimeKAN',
    module='tsflab.models.timekan',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TimeKAN.toml',
    model_card='src/tsflab/models/timekan/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
