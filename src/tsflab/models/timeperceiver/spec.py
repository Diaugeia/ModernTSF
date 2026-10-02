"""Model specification for TimePerceiver."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.timeperceiver.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    n_heads: int = 2
    patch_len: int = 16
    dropout: float = 0.1
    num_latents: int = 8
    latent_dim: int = 128
    latent_d_ff: int = 256
    num_latent_blocks: int = 1
    query_share: bool = True


def build_model(cfg, params):
    """Construct TimePerceiver from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 32), n_heads=params.get('n_heads', 2), patch_len=params.get('patch_len', 16), dropout=params.get('dropout', 0.1), num_latents=params.get('num_latents', 8), latent_dim=params.get('latent_dim', 128), latent_d_ff=params.get('latent_d_ff', 256), num_latent_blocks=params.get('num_latent_blocks', 1), query_share=bool(params.get('query_share', True)))
    )


SPEC = ModelSpec(
    name='TimePerceiver',
    module='tsflab.models.timeperceiver',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TimePerceiver.toml',
    model_card='src/tsflab/models/timeperceiver/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
