"""Runtime specification for PENGUIN."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.penguin.model import Model

"""Validated parameters for PENGUIN."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 8
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    periods: list[int] = [24]
    activation: Literal["relu", "gelu"] = "relu"
    use_rmsnorm: bool = False
    alibi: bool = True
    causal: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 128),
        d_ff=params.get("d_ff", 256),
        n_heads=params.get("n_heads", 8),
        e_layers=params.get("e_layers", 2),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        dropout=params.get("dropout", 0.1),
        periods=params.get("periods", [24]),
        activation=params.get("activation", "relu"),
        use_rmsnorm=params.get("use_rmsnorm", False),
        alibi=params.get("alibi", True),
        causal=params.get("causal", True),
    )


SPEC = ModelSpec(
    name="PENGUIN",
    module="tsflab.models.penguin",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/PENGUIN.toml",
    model_card="src/tsflab/models/penguin/README.md",
    smoke_config="configs/runs/smoke_penguin.toml",
    capabilities=frozenset({'time-series'}),
    components=('embed', 'flatten_forecast_head', 'mamba', 'periodic_alibi_bias', 'revin'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
