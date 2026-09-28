"""Runtime specification for KNNForecasterTS."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.knn_forecaster_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_prototypes: int = 32
    kernel_gamma: float = 0.08


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        num_prototypes=params.get("num_prototypes", 32),
        kernel_gamma=params.get("kernel_gamma", 0.08),
    )


SPEC = ModelSpec(
    name="KNNForecasterTS",
    module="moderntsf.models.knn_forecaster_ts",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/KNNForecasterTS.toml",
    model_card="src/moderntsf/models/knn_forecaster_ts/README.md",
    capabilities=frozenset(["time-series"]),
    components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
