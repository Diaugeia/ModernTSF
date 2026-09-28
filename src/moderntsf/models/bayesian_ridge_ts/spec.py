"""Runtime specification for BayesianRidgeTS."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.bayesian_ridge_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    initial_weight_precision: float = 1e-3


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params.get("initial_weight_precision", 1e-3))


SPEC = ModelSpec(name="BayesianRidgeTS", module="moderntsf.models.bayesian_ridge_ts", model_class=Model, factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/BayesianRidgeTS.toml", model_card="src/moderntsf/models/bayesian_ridge_ts/README.md", capabilities=frozenset(["time-series"]), components=(), contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0})
