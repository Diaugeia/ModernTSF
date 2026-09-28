"""Model specification for TexFilter."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.texfilter.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    embed_size: int = 128
    hidden_size: int = 256
    dropout: float = 0.0


def build_model(cfg, params):
    """Construct TexFilter from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], embed_size=params.get('embed_size', 128), hidden_size=params.get('hidden_size', 256), dropout=params.get('dropout', 0.0))
    )


SPEC = ModelSpec(
    name='TexFilter',
    module='moderntsf.models.texfilter',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TexFilter.toml',
    model_card='src/moderntsf/models/texfilter/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
