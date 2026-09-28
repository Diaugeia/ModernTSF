"""Runtime specification for Dualformer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.dualformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    c_out: int = Field(gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=3, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: str = "gelu"
    factor: float = Field(default=1.0, gt=0.0)
    alpha: float = Field(default=1.0, gt=0.0, le=1.0)
    num_harmonics: int = Field(default=3, gt=0)
    dc_bins: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.enc_in != self.c_out:
            raise ValueError("enc_in and c_out must match")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        c_out=params["c_out"],
        d_model=params.get("d_model", 512),
        n_heads=params.get("n_heads", 8),
        e_layers=params.get("e_layers", 3),
        d_ff=params.get("d_ff", 2048),
        dropout=params.get("dropout", 0.1),
        activation=params.get("activation", "gelu"),
        factor=params.get("factor", 1.0),
        alpha=params.get("alpha", 1.0),
        num_harmonics=params.get("num_harmonics", 3),
        dc_bins=params.get("dc_bins", 3),
    )


SPEC = ModelSpec(
    name="Dualformer",
    module="moderntsf.models.dualformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/Dualformer.toml",
    model_card="src/moderntsf/models/dualformer/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
    components=(
        "forecast_embedding",
        "frequency_band_sampler",
        "harmonic_energy_gate",
        "revin",
        "self_attention_family",
        "transformer_encdec",
    ),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
