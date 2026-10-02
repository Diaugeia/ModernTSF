"""Model specification for STPGNN."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.stpgnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STPGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    must not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    required spatial dimension; the remaining fields are kept small so the smoke
    run stays fast.
    """

    enc_in: int
    dropout: float = 0.1
    topk: int = 4
    residual_channels: int = 16
    end_channels: int = 64
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    dims: int = 16


def build_model(cfg, params):
    """Construct STPGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), dropout=params.get('dropout', 0.1), topk=params.get('topk', 4), residual_channels=params.get('residual_channels', 16), end_channels=params.get('end_channels', 64), kernel_size=params.get('kernel_size', 2), blocks=params.get('blocks', 2), layers=params.get('layers', 2), dims=params.get('dims', 16))
    )


SPEC = ModelSpec(
    name='STPGNN',
    module='tsflab.models.stpgnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/STPGNN.toml',
    model_card='src/tsflab/models/stpgnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=(),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
