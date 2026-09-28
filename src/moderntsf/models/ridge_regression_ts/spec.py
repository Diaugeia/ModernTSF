"""Runtime specification for RidgeRegressionTS."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.ridge_regression_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    l2_penalty: float = 1e-4


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        l2_penalty=params.get("l2_penalty", 1e-4),
    )


SPEC = ModelSpec(
    name="RidgeRegressionTS",
    module="moderntsf.models.ridge_regression_ts",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/RidgeRegressionTS.toml",
    model_card="src/moderntsf/models/ridge_regression_ts/README.md",
    capabilities=frozenset(["time-series"]),
    components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
