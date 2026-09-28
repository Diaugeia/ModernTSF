"""Runtime specification for Sensorformer."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.sensorformer.model import Model

"""Validated parameters for Sensorformer."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    layers: int = 2
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        n_heads=params.get("n_heads", 4),
        d_ff=params.get("d_ff", 128),
        layers=params.get("layers", 2),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        dropout=params.get("dropout", 0.1),
    )


SPEC = ModelSpec(
    name="Sensorformer",
    module="moderntsf.models.sensorformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/Sensorformer.toml",
    model_card="src/moderntsf/models/sensorformer/README.md",
    smoke_config="configs/runs/smoke_sensorformer.toml",
    capabilities=frozenset({'time-series'}),
    components=('embed', 'flatten_forecast_head', 'global_patch_compression_attention'),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
