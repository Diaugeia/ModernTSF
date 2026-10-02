"""Runtime specification for HMformer."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.hmformer.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    num_scales: int = 3
    depth: int = 1
    num_heads: int = 4


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        dropout=params.get("dropout", 0.1),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        num_scales=params.get("num_scales", 3),
        depth=params.get("depth", 1),
        num_heads=params.get("num_heads", 4),
    )


SPEC = ModelSpec(
    name="HMformer",
    module="tsflab.models.hmformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/HMformer.toml",
    model_card="src/tsflab/models/hmformer/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
