"""Runtime specification for SymTime."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.symtime.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    patch_len: int = 16
    num_layers: int = 2
    num_heads: int = 4
    trend_kernel: int = 25
    dropout: float = 0.1


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params.get("d_model", 64),
        patch_len=params.get("patch_len", 16),
        num_layers=params.get("num_layers", 2),
        num_heads=params.get("num_heads", 4),
        trend_kernel=params.get("trend_kernel", 25),
        dropout=params.get("dropout", 0.1),
    )


SPEC = ModelSpec(
    name="SymTime",
    module="moderntsf.models.symtime",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/SymTime.toml",
    model_card="src/moderntsf/models/symtime/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=("revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
