"""Model specification for RAGC."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.ragc.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated RAGC parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the
    dataset, so they are not declared here. ``enc_in`` (= number of nodes
    ``N``) is required; the remaining defaults are kept modest for a fast
    spatiotemporal smoke run.
    """

    enc_in: int
    input_dim: int = 3
    spatial_dim: int = 16
    embed_dim: int = 16
    temp_dim_tid: int = 8
    temp_dim_diw: int = 8
    num_time_in_day: int = 24
    num_day_in_week: int = 7
    num_layer: int = 2
    order: int = 1
    sse_p: float = 0.1
    use_sse: bool = True


def build_model(cfg, params):
    """Construct RAGC from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params.get("input_dim", 3),
        spatial_dim=params.get("spatial_dim", 16),
        embed_dim=params.get("embed_dim", 16),
        temp_dim_tid=params.get("temp_dim_tid", 8),
        temp_dim_diw=params.get("temp_dim_diw", 8),
        num_time_in_day=params.get("num_time_in_day", 24),
        num_day_in_week=params.get("num_day_in_week", 7),
        num_layer=params.get("num_layer", 2),
        order=params.get("order", 1),
        sse_p=params.get("sse_p", 0.1),
        use_sse=params.get("use_sse", True),
    )


SPEC = ModelSpec(
    name="RAGC",
    module="tsflab.models.ragc",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/RAGC.toml",
    model_card="src/tsflab/models/ragc/README.md",
    smoke_config=None,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks", "regularized_adaptive_graph_conv"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
