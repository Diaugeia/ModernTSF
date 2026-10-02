"""Runtime specification for PolynomialRegressionTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.polynomial_regression_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    degree: int = 2


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        degree=params.get("degree", 2),
    )


SPEC = ModelSpec(
    name="PolynomialRegressionTS",
    module="tsflab.models.polynomial_regression_ts",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/PolynomialRegressionTS.toml",
    model_card="src/tsflab/models/polynomial_regression_ts/README.md",
    capabilities=frozenset(["time-series"]),
    components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
