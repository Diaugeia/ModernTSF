"""Model specification for VisiFold."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.visifold.model import Model

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Validated VisiFold parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset and are not
    declared here. Defaults are modest for fast smoke runs.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    input_dim: int = 3
    steps_per_day: int = 24
    input_embedding_dim: int = 8
    tod_embedding_dim: int = 4
    dow_embedding_dim: int = 4
    spatial_embedding_dim: int = 8
    feed_forward_dim: int = 16
    num_heads: int = 2
    num_layers: int = 1
    mask_ratio: float = 0.2
    subgraph_size: int = 4
    dropout: float = 0.0


def build_model(cfg, params):
    """Construct VisiFold from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params.get("input_dim", 3),
        steps_per_day=params.get("steps_per_day", 24),
        input_embedding_dim=params.get("input_embedding_dim", 8),
        tod_embedding_dim=params.get("tod_embedding_dim", 4),
        dow_embedding_dim=params.get("dow_embedding_dim", 4),
        spatial_embedding_dim=params.get("spatial_embedding_dim", 8),
        feed_forward_dim=params.get("feed_forward_dim", 16),
        num_heads=params.get("num_heads", 2),
        num_layers=params.get("num_layers", 1),
        mask_ratio=params.get("mask_ratio", 0.2),
        subgraph_size=params.get("subgraph_size", 4),
        dropout=params.get("dropout", 0.0),
    )


SPEC = ModelSpec(
    name="VisiFold",
    module="tsflab.models.visifold",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/VisiFold.toml",
    model_card="src/tsflab/models/visifold/README.md",
    smoke_config=None,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks", "node_visibility"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
