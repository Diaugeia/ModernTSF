"""Runtime specification for CMoS."""

from pydantic import BaseModel, Field

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.cmos.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    seg_size: int = Field(default=4, gt=0)
    num_map: int = Field(default=3, gt=0)
    kernel_size: int = Field(default=4, gt=0)
    period: int | None = Field(default=None, gt=0)


def build_model(cfg, params):
    return Model(
        c_in=params["enc_in"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        seg_size=params.get("seg_size", 4),
        num_map=params.get("num_map", 3),
        kernel_size=params.get("kernel_size", 4),
        period=params.get("period"),
    )


SPEC = ModelSpec(
    name="CMoS", module="moderntsf.models.cmos", model_class=Model, factory=build_model,
    params_schema=ModelParameterConfig, config_path="configs/models/CMoS.toml",
    model_card="src/moderntsf/models/cmos/README.md", smoke_config=None,
    capabilities=frozenset(["time-series"]), components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
