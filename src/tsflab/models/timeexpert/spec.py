"""Runtime specification for TimeExpert."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timeexpert.model import Model

"""Validated parameters for TimeExpert."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    topk: int = 4
    shared: bool = False


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 128),
        n_heads=params.get("n_heads", 8),
        e_layers=params.get("e_layers", 2),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        dropout=params.get("dropout", 0.1),
        topk=params.get("topk", 4),
        shared=params.get("shared", False),
    )


SPEC = ModelSpec(
    name="TimeExpert",
    module="tsflab.models.timeexpert",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TimeExpert.toml",
    model_card="src/tsflab/models/timeexpert/README.md",
    smoke_config="configs/runs/smoke_timeexpert.toml",
    capabilities=frozenset({'time-series'}),
    components=('embed', 'flatten_forecast_head', 'topk_expert_attention'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
