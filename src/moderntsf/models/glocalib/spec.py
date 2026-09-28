"""Model specification for GlocalIB."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.glocalib.model import Model

from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Parameters for the Glocal-IB forecasting model."""

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    # Alignment regularizer weight and augmentation strength.
    align_weight: float = Field(default=0.5, ge=0.0)
    mask_ratio: float = Field(default=0.25, ge=0.0, lt=1.0)
    # "cos_align" (robust) or "contrastive" (InfoNCE over time steps).
    align_loss_type: Literal["cos_align", "contrastive"] = "cos_align"
    kl_weight: float = Field(default=0.01, ge=0.0)


def build_model(cfg, params):
    """Construct GlocalIB from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        align_weight=params.get("align_weight", 0.5),
        mask_ratio=params.get("mask_ratio", 0.25),
        align_loss_type=params.get("align_loss_type", "cos_align"),
        kl_weight=params.get("kl_weight", 0.01),
    )


SPEC = ModelSpec(
    name='GlocalIB',
    module='moderntsf.models.glocalib',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GlocalIB.toml',
    model_card='src/moderntsf/models/glocalib/README.md',
    smoke_config='configs/runs/smoke_glocalib.toml',
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
