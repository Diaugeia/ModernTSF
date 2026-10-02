"""Runtime specification for MAFS."""

from __future__ import annotations

from tsflab.experiments.runner.objective import TrainingBatch

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mafs.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    num_agents: int = 4
    num_layers: int = 2
    num_heads: int = 4
    topology: str = "star"


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        dropout=params.get("dropout", 0.1),
        num_agents=params.get("num_agents", 4),
        num_layers=params.get("num_layers", 2),
        num_heads=params.get("num_heads", 4),
        topology=params.get("topology", "star"),
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Forecast criterion plus the multi-scale prefix specialization loss."""
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    return forecast, loss + model.specialization_loss(batch.x, batch.target)


SPEC = ModelSpec(
    name="MAFS",
    module="tsflab.models.mafs",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/MAFS.toml",
    model_card="src/tsflab/models/mafs/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
