"""Runtime specification for GaussianProcessTS."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.gaussian_process_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_inducing: int = 16
    length_scale: float = 1.0
    noise: float = 1e-3


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params.get("num_inducing", 16), params.get("length_scale", 1.0), params.get("noise", 1e-3))


SPEC = ModelSpec(name="GaussianProcessTS", module="moderntsf.models.gaussian_process_ts", model_class=Model, factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/GaussianProcessTS.toml", model_card="src/moderntsf/models/gaussian_process_ts/README.md", capabilities=frozenset(["time-series"]), components=(), contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0})
