"""Runtime specification for AWEMixer."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.awemixer.model import Model

"""Validated parameters for AWEMixer."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    dropout: float = 0.2
    num_scales: int = 3
    wavelet_level: int = 3
    wavelet: str = "db4"
    num_fusion_layers: int = 1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 128),
        dropout=params.get("dropout", 0.2),
        num_scales=params.get("num_scales", 3),
        wavelet_level=params.get("wavelet_level", 3),
        wavelet=params.get("wavelet", "db4"),
        num_fusion_layers=params.get("num_fusion_layers", 1),
    )


SPEC = ModelSpec(
    name="AWEMixer",
    module="tsflab.models.awemixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/AWEMixer.toml",
    model_card="src/tsflab/models/awemixer/README.md",
    smoke_config="configs/runs/smoke_awemixer.toml",
    capabilities=frozenset({'time-series'}),
    components=('revin', 'wavelet'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
