"""Runtime specification for PatchTSMixer."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.patchtsmixer.model import Model

"""Validated parameters for PatchTSMixer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 16
    patch_len: int = 16
    stride: int = 8
    num_layers: int = 3
    expansion_factor: int = 2
    dropout: float = 0.1
    head_dropout: float = 0.1
    mode: Literal["common_channel", "mix_channel"] = "common_channel"
    gated_attn: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 16),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        num_layers=params.get("num_layers", 3),
        expansion_factor=params.get("expansion_factor", 2),
        dropout=params.get("dropout", 0.1),
        head_dropout=params.get("head_dropout", 0.1),
        mode=params.get("mode", "common_channel"),
        gated_attn=params.get("gated_attn", True),
    )


SPEC = ModelSpec(
    name="PatchTSMixer",
    module="tsflab.models.patchtsmixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/PatchTSMixer.toml",
    model_card="src/tsflab/models/patchtsmixer/README.md",
    smoke_config="configs/runs/smoke_patchtsmixer.toml",
    capabilities=frozenset({'time-series'}),
    components=('flatten_forecast_head', 'revin', 'softmax_gate'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
