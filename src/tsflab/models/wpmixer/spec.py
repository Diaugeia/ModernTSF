"""Model specification for WPMixer."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.wpmixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    dropout: float = 0.1
    tfactor: int = 5
    dfactor: int = 5
    wavelet: str = "db2"
    level: int = 1
    patch_len: int = 16
    stride: int = 8
    no_decomposition: bool = False


def build_model(cfg, params):
    """Construct WPMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), dropout=params.get('dropout', 0.1), tfactor=params.get('tfactor', 5), dfactor=params.get('dfactor', 5), wavelet=params.get('wavelet', 'db2'), level=params.get('level', 1), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), no_decomposition=bool(params.get('no_decomposition', False)))
    )


SPEC = ModelSpec(
    name='WPMixer',
    module='tsflab.models.wpmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/WPMixer.toml',
    model_card='src/tsflab/models/wpmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
