"""Runtime specification for DPWMixer."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dpwmixer.model import Model

"""Validated parameters for DPWMixer."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    down_sampling_layers: int = 2


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 128),
        dropout=params.get("dropout", 0.1),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        down_sampling_layers=params.get("down_sampling_layers", 2),
    )


SPEC = ModelSpec(
    name="DPWMixer",
    module="tsflab.models.dpwmixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/DPWMixer.toml",
    model_card="src/tsflab/models/dpwmixer/README.md",
    smoke_config="configs/runs/smoke_dpwmixer.toml",
    capabilities=frozenset({'time-series'}),
    components=('revin', 'wavelet'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
