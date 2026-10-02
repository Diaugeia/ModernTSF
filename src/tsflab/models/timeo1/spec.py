"""Runtime specification for TimeO1."""

import torch

from tsflab.benchmark.runner.objective import TrainingBatch

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.timeo1.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    alpha: float = 0.8
    rank_ratio: float = 0.5


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params.get("d_model", 64),
        alpha=params.get("alpha", 0.8),
        rank_ratio=params.get("rank_ratio", 0.5),
    )


def training_setup(model, train_loader, *, pred_len, features):
    """Fit the per-variate label basis on the training split (Time-o1 Sec. 4)."""
    start = -1 if features == "MS" else 0
    labels = [batch_y[:, -pred_len:, start:].float() for _, batch_y, *_ in train_loader]
    model.fit_projection(torch.cat(labels).to(model.projection.device))


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (5) with the configured criterion as the temporal term."""
    forecast = batch.forecast(model)
    loss = model.transformed_alignment_loss(
        batch.align(forecast), batch.target, reduction="mean", temporal=criterion
    )
    return forecast, loss


SPEC = ModelSpec(
    name="TimeO1",
    module="tsflab.models.timeo1",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TimeO1.toml",
    model_card="src/tsflab/models/timeo1/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
