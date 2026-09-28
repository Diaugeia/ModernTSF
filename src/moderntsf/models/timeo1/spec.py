"""Runtime specification for TimeO1."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.timeo1.model import Model
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


SPEC = ModelSpec(
    name="TimeO1",
    module="moderntsf.models.timeo1",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TimeO1.toml",
    model_card="src/moderntsf/models/timeo1/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
