"""Model specification for xPatch."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.xpatch.model import Model

from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    padding_patch: Literal["end", "none"] = "end"
    ma_type: Literal["ema", "dema"] = "ema"
    alpha: float = Field(default=0.3, gt=0.0, lt=1.0)
    beta: float = Field(default=0.3, gt=0.0, lt=1.0)
    revin: bool = True
    hidden_dim: int = Field(default=64, gt=0)


def build_model(cfg, params):
    """Construct xPatch from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        padding_patch=params.get("padding_patch", "end"),
        ma_type=params.get("ma_type", "ema"),
        alpha=params.get("alpha", 0.3),
        beta=params.get("beta", 0.3),
        revin=bool(params.get("revin", True)),
        hidden_dim=params.get("hidden_dim", 64),
    )


SPEC = ModelSpec(
    name="xPatch",
    module="tsflab.models.xpatch",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/xPatch.toml",
    model_card="src/tsflab/models/xpatch/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
