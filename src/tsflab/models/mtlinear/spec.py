"""Runtime specification for MTLinear."""

import torch
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.mtlinear.model import LAYER_TYPES, Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    layer_type: str = "DLinear"
    kernel_size: int = Field(default=25, ge=1)
    cluster_dist: float = Field(default=0.5, ge=0)
    max_clusters: int = Field(default=8, ge=1)
    penalty_param: float = Field(default=2.0, ge=0)
    use_horizon_penalty: bool = True
    use_variates_penalty: bool = True


def build_model(cfg, params):
    if params.get("layer_type", "DLinear") not in LAYER_TYPES:
        raise ValueError(f"layer_type must be one of {LAYER_TYPES}")
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        layer_type=params.get("layer_type", "DLinear"),
        kernel_size=params.get("kernel_size", 25),
        cluster_dist=params.get("cluster_dist", 0.5),
        max_clusters=params.get("max_clusters", 8),
        penalty_param=params.get("penalty_param", 2.0),
        use_horizon_penalty=params.get("use_horizon_penalty", True),
        use_variates_penalty=params.get("use_variates_penalty", True),
    )


def training_setup(model, train_loader, *, pred_len, features):
    """Fit the variate grouping on the training split (MTLinear Sec. 4.2).

    The newest step of every stride-1 window visits each training timestamp once, so
    their stack is the training series (minus the first lookback).
    """
    steps = [batch_x[:, -1, :].float() for batch_x, *_ in train_loader]
    model.fit_clusters(torch.cat(steps).to(model.group_of.device))


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (7)-(8) replaces the observation loss with the penalized squared error."""
    forecast = batch.forecast(model)
    if not (model.use_horizon_penalty or model.use_variates_penalty):
        return forecast, criterion(batch.align(forecast), batch.target)
    return forecast, model.penalized_loss(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="MTLinear",
    module="tsflab.models.mtlinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/MTLinear.toml",
    model_card="src/tsflab/models/mtlinear/README.md",
    smoke_config="configs/runs/smoke_mtlinear.toml",
    capabilities=frozenset({"time-series"}),
    components=("channel_wise_linear", "dlinear","last_value_center", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
