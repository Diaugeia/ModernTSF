"""Runtime specification for LIFT."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.lift.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    leader_num: int = Field(default=4, ge=1)
    state_num: int = Field(default=8, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    kernel_size: int = Field(default=25, ge=1)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        leader_num=params.get("leader_num", 4),
        state_num=params.get("state_num", 8),
        temperature=params.get("temperature", 1.0),
        kernel_size=params.get("kernel_size", 25),
    )


SPEC = ModelSpec(
    name="LIFT",
    module="tsflab.models.lift",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/LIFT.toml",
    model_card="src/tsflab/models/lift/README.md",
    smoke_config="configs/runs/smoke_lift.toml",
    capabilities=frozenset({"time-series"}),
    components=("dlinear",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
