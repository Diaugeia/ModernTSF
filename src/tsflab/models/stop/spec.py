"""Model specification for STOP."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stop.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STOP parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 16
    prompt_dim: int = 16
    num_layer: int = 2
    hid_dim: int = 64
    tod_size: int = 24
    kernel_size: int = 3
    core: int = 4
    head: int = 4


def build_model(cfg, params):
    """Construct STOP from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"], model_dim=params.get("model_dim", 16),
        prompt_dim=params.get("prompt_dim", 16), num_layer=params.get("num_layer", 2),
        hid_dim=params.get("hid_dim", 64), tod_size=params.get("tod_size", 24),
        kernel_size=params.get("kernel_size", 3), core=params.get("core", 4),
        head=params.get("head", 4),
    )


SPEC = ModelSpec(
    name='STOP',
    module='tsflab.models.stop',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/STOP.toml',
    model_card='src/tsflab/models/stop/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks', 'series_decomposition'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
