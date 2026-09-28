"""Model specification for COSA."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.cosa.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    context_len: int = Field(default=10, gt=0)
    gate_init: float = 0.1


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params.get("context_len", 10),
        params.get("gate_init", 0.1),
    )


SPEC = ModelSpec(
    name="COSA",
    module="moderntsf.models.cosa",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/COSA.toml",
    model_card="src/moderntsf/models/cosa/README.md",
    capabilities=frozenset(["time-series", "test-time-adaptation"]),
        components=("channel_wise_linear",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
