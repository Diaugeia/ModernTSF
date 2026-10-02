"""Runtime specification for Kronos."""

from __future__ import annotations

from tsflab.benchmark.runner.objective import TrainingBatch

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.kronos.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    code_bits: int = 8
    num_layers: int = 2
    num_heads: int = 4


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        dropout=params.get("dropout", 0.1),
        code_bits=params.get("code_bits", 8),
        num_layers=params.get("num_layers", 2),
        num_heads=params.get("num_heads", 4),
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Forecast criterion plus the hierarchical tokenizer reconstruction loss."""
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    return forecast, loss + model.tokenizer_loss(batch.x)


SPEC = ModelSpec(
    name="Kronos",
    module="tsflab.models.kronos",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/Kronos.toml",
    model_card="src/tsflab/models/kronos/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
