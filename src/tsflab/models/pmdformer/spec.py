"""Model specification for PMDformer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pmdformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    patch_len: int = 16
    num_heads: int = 4
    dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct PMDformer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params["enc_in"],
        d_model=params.get("d_model", 64), patch_len=params.get("patch_len", 16),
        num_heads=params.get("num_heads", 4), dropout=params.get("dropout", 0.0),
        use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name='PMDformer',
    module='tsflab.models.pmdformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/PMDformer.toml',
    model_card='src/tsflab/models/pmdformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
