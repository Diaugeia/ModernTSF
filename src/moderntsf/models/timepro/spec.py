"""Runtime specification for TimePro."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.timepro.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    d_model: int = Field(default=16, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_state: int = Field(default=1, gt=0)
    d_conv: int = Field(default=5, gt=0)
    expand: float = Field(default=1.0, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_norm: bool = True

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.d_state != 1:
            raise ValueError("this local implementation supports only d_state=1")
        return self


def build_model(cfg, params):
    """Construct TimePro from a validated run configuration."""
    patch_len = params.get("patch_len", 16)
    stride = params.get("stride", 8)
    d_model = params.get("d_model", 16)
    expand = params.get("expand", 1.0)
    if patch_len > cfg.task.seq_len:
        raise ValueError("patch_len must not exceed seq_len")
    patch_num = (cfg.task.seq_len + stride - patch_len) // stride + 1
    d_inner = int(expand * d_model * patch_num)
    if d_inner % patch_num:
        raise ValueError("expand * d_model * patch_num must be an integer multiple of patch_num")
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        e_layers=params.get("e_layers", 2),
        d_state=params.get("d_state", 1),
        d_conv=params.get("d_conv", 5),
        expand=expand,
        dropout=params.get("dropout", 0.1),
        use_norm=bool(params.get("use_norm", True)),
    )


SPEC = ModelSpec(
    name="TimePro",
    module="moderntsf.models.timepro",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TimePro.toml",
    model_card="src/moderntsf/models/timepro/README.md",
    smoke_config="configs/runs/smoke_timepro.toml",
    capabilities=frozenset({"time-series"}),
    components=("hyper_state_scan",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
