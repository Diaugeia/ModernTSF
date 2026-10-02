"""Runtime specification for MMPD."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.mmpd.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    patch_len: int = 8
    num_heads: int = 4
    adjacent_range: int = 1
    diffusion_steps: int = 100
    denoiser_depth: int = 2
    diffusion_weight: float = 0.99


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        dropout=params.get("dropout", 0.1),
        patch_len=params.get("patch_len", 8),
        num_heads=params.get("num_heads", 4),
        adjacent_range=params.get("adjacent_range", 1),
        diffusion_steps=params.get("diffusion_steps", 100),
        denoiser_depth=params.get("denoiser_depth", 2),
        diffusion_weight=params.get("diffusion_weight", 0.99),
    )


SPEC = ModelSpec(
    name="MMPD",
    module="tsflab.models.mmpd",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/MMPD.toml",
    model_card="src/tsflab/models/mmpd/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
