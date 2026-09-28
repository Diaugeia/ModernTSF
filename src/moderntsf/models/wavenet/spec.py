"""Model specification for WaveNet."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.wavenet.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    residual_channels: int = Field(default=16, ge=1)
    dilation_channels: int = Field(default=16, ge=1)
    skip_channels: int = Field(default=64, ge=1)
    end_channels: int = Field(default=128, ge=1)
    kernel_size: int = Field(default=2, ge=1)
    blocks: int = Field(default=2, ge=1)
    layers: int = Field(default=2, ge=1)
    use_norm: bool = True


def build_model(cfg, params):
    """Construct WaveNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], residual_channels=params.get('residual_channels', 16), dilation_channels=params.get('dilation_channels', 16), skip_channels=params.get('skip_channels', 64), end_channels=params.get('end_channels', 128), kernel_size=params.get('kernel_size', 2), blocks=params.get('blocks', 2), layers=params.get('layers', 2), use_norm=bool(params.get('use_norm', True)))
    )


SPEC = ModelSpec(
    name='WaveNet',
    module='moderntsf.models.wavenet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/WaveNet.toml',
    model_card='src/moderntsf/models/wavenet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('gated_dilated_conv', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
