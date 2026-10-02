"""Runtime specification for TimeCAP."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.timecap.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    patch_len: int = 16
    group_size: int = 4
    group_stride: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    fusion_alpha: float = 0.1
    fusion_midpoint: float | None = None


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params.get("d_model", 32),
        patch_len=params.get("patch_len", 16),
        group_size=params.get("group_size", 4),
        group_stride=params.get("group_stride", 2),
        num_heads=params.get("num_heads", 4),
        dropout=params.get("dropout", 0.1),
        fusion_alpha=params.get("fusion_alpha", 0.1),
        fusion_midpoint=params.get("fusion_midpoint"),
    )


SPEC = ModelSpec(
    name="TimeCAP",
    module="tsflab.models.timecap",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TimeCAP.toml",
    model_card="src/tsflab/models/timecap/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
