"""Runtime specification for GOTSF."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.gotsf.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    num_intervals: int = 4
    interval_min: float = -2.0
    interval_max: float = 2.0
    decay_rate: float = 50.0
    classification_weight: float = 0.1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        dropout=params.get("dropout", 0.1),
        num_intervals=params.get("num_intervals", 4),
        interval_min=params.get("interval_min", -2.0),
        interval_max=params.get("interval_max", 2.0),
        decay_rate=params.get("decay_rate", 50.0),
        classification_weight=params.get("classification_weight", 0.1),
    )


SPEC = ModelSpec(
    name="GOTSF",
    module="tsflab.models.gotsf",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/GOTSF.toml",
    model_card="src/tsflab/models/gotsf/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
