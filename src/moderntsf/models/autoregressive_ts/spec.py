"""Runtime specification for AutoRegressiveTS."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.autoregressive_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"])


SPEC = ModelSpec(
    name="AutoRegressiveTS",
    module="moderntsf.models.autoregressive_ts",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/AutoRegressiveTS.toml",
    model_card="src/moderntsf/models/autoregressive_ts/README.md",
    capabilities=frozenset(["time-series"]),
    components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
