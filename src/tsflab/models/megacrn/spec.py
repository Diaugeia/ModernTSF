"""Model specification for MegaCRN."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.megacrn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MegaCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are *injected* by the runner from the dataset
    (see ``src/tsflab/benchmark/runner/run_one.py``) and need not be declared in TOML.
    """

    enc_in: int  # number of spatial nodes N (required)
    input_dim: int = 3
    rnn_units: int = 32
    num_layers: int = 1
    cheb_k: int = 3
    mem_num: int = 8
    mem_dim: int = 16


def build_model(cfg, params):
    """Construct MegaCRN from a validated run configuration."""
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), rnn_units=params.get('rnn_units', 32), num_layers=params.get('num_layers', 1), cheb_k=params.get('cheb_k', 3), mem_num=params.get('mem_num', 8), mem_dim=params.get('mem_dim', 16))


SPEC = ModelSpec(
    name='MegaCRN',
    module='tsflab.models.megacrn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MegaCRN.toml',
    model_card='src/tsflab/models/megacrn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
