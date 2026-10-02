"""Model specification for GTS."""

from __future__ import annotations

from tsflab.benchmark.runner.objective import TrainingBatch

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.gts.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated GTS parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    need not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    only required field and serves as the fallback for ``num_nodes``.
    """

    enc_in: int = Field(ge=1)
    input_dim: int = Field(default=3, ge=1)
    rnn_units: int = Field(default=16, ge=1)
    num_rnn_layers: int = Field(default=1, ge=1)
    max_diffusion_step: int = Field(default=2, ge=1)
    embedding_dim: int = Field(default=16, ge=1)
    temp: float = Field(default=0.5, gt=0)
    prior_strength: float = Field(default=0.1, ge=0)


def build_model(cfg, params):
    """Construct GTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), rnn_units=params.get('rnn_units', 16), num_rnn_layers=params.get('num_rnn_layers', 1), max_diffusion_step=params.get('max_diffusion_step', 2), embedding_dim=params.get('embedding_dim', 16), temp=params.get('temp', 0.5), prior_strength=params.get('prior_strength', 0.1))
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Forecast criterion plus the graph prior BCE when a prior adjacency exists."""
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    if model.has_prior:
        loss = loss + model.graph_prior_loss(batch.x)
    return forecast, loss


SPEC = ModelSpec(
    name='GTS',
    module='tsflab.models.gts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GTS.toml',
    model_card='src/tsflab/models/gts/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('channel_alignment', 'marks'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
    training_objective=training_objective,
)
