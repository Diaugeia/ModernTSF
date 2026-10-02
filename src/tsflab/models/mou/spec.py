"""Runtime specification for MoU."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mou.model import Model

"""Validated parameters for MoU."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 16
    d_ff: int = 256
    e_layers: int = 1
    num_experts: int = 4
    top_k: int = 2
    d_state: int = 21
    ffn_expand: int = 2
    dropout: float = 0.1
    mamba_dropout: float = 0.1
    ffn_dropout: float = 0.2
    block_dropout: float = 0.2
    conv_dropout: float = 0.3
    attn_dropout: float = 0.0
    tf_dropout: float = 0.2
    head_dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        d_model=params.get("d_model", 128),
        n_heads=params.get("n_heads", 16),
        d_ff=params.get("d_ff", 256),
        e_layers=params.get("e_layers", 1),
        num_experts=params.get("num_experts", 4),
        top_k=params.get("top_k", 2),
        d_state=params.get("d_state", 21),
        ffn_expand=params.get("ffn_expand", 2),
        dropout=params.get("dropout", 0.1),
        mamba_dropout=params.get("mamba_dropout", 0.1),
        ffn_dropout=params.get("ffn_dropout", 0.2),
        block_dropout=params.get("block_dropout", 0.2),
        conv_dropout=params.get("conv_dropout", 0.3),
        attn_dropout=params.get("attn_dropout", 0.0),
        tf_dropout=params.get("tf_dropout", 0.2),
        head_dropout=params.get("head_dropout", 0.0),
        use_revin=params.get("use_revin", True),
    )


SPEC = ModelSpec(
    name="MoU",
    module="tsflab.models.mou",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/MoU.toml",
    model_card="src/tsflab/models/mou/README.md",
    smoke_config="configs/runs/smoke_mou.toml",
    capabilities=frozenset({'time-series'}),
    components=('revin', 'mamba', 'positional_encoding', 'flatten_forecast_head'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
