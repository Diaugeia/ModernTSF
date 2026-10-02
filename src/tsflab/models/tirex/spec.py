"""Runtime specification for TiRex."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tirex.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    patch_len: int = 16
    num_layers: int = 2
    dropout: float = 0.1
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        features=cfg.task.features,
        d_model=params.get("d_model", 64),
        patch_len=params.get("patch_len", 16),
        num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout", 0.1),
        quantile_levels=params.get("quantile_levels") or list(cfg.evaluation.quantile_levels),
    )


SPEC = ModelSpec(
    name="TiRex",
    module="tsflab.models.tirex",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TiRex.toml",
    model_card="src/tsflab/models/tirex/README.md",
    smoke_config=None,
    capabilities=frozenset(["quantile-output", "time-series"]),
        components=("quantile_head",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
