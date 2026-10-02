"""Model specification for BiST."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.bist.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated BiST parameters supplied via ``model.params``."""

    enc_in: int = Field(ge=1)
    model_dim: int = Field(default=32, ge=1)
    prompt_dim: int = Field(default=16, ge=1)
    num_layers: int = Field(default=3, ge=1)
    tod_size: int = Field(default=24, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    residual_steps: int = Field(default=2, ge=0)
    graph_dim: int = Field(default=8, ge=1)
    virtual_clusters: int = Field(default=8, ge=1)


def build_model(cfg, params):
    """Construct BiST from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        model_dim=params.get("model_dim", 32),
        prompt_dim=params.get("prompt_dim", 16),
        num_layers=params.get("num_layers", 3),
        tod_size=params.get("tod_size", 24),
        kernel_size=params.get("kernel_size", 3),
        residual_steps=params.get("residual_steps", 2),
        graph_dim=params.get("graph_dim", 8),
        virtual_clusters=params.get("virtual_clusters", 8),
    )


SPEC = ModelSpec(
    name='BiST',
    module='tsflab.models.bist',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/BiST.toml',
    model_card='src/tsflab/models/bist/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks', 'series_decomposition'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
