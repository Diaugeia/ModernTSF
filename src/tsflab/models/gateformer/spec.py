"""Runtime specification for Gateformer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.gateformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    """Construct Gateformer from a validated run configuration."""
    if params["patch_len"] > cfg.task.seq_len:
        raise ValueError("patch_len must not exceed seq_len")
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        d_model=params.get("d_model", 128),
        n_heads=params.get("n_heads", 8),
        e_layers=params.get("e_layers", 2),
        d_ff=params.get("d_ff", 256),
        dropout=params.get("dropout", 0.1),
    )


SPEC = ModelSpec(
    name="Gateformer",
    module="tsflab.models.gateformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/Gateformer.toml",
    model_card="src/tsflab/models/gateformer/README.md",
    smoke_config="configs/runs/smoke_gateformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "gated_fusion", "positional_encoding"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
