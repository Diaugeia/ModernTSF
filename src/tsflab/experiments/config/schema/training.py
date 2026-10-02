"""Optimization, checkpointing, curriculum, and optional training-trick schemas."""

from pydantic import BaseModel, ConfigDict, Field


class TrainOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Adam"
    lr: float = 0.001
    weight_decay: float = 0.0001
    lradj: str = "constant"
    params: dict = Field(default_factory=dict)


class TrainCheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = "best"
    save_k: int = 3


class CurriculumConfig(BaseModel):
    """Curriculum-learning trick (BasicTS horizon scheme).

    Disabled by default. When ``enabled`` is True the loss is computed on a
    growing forecast horizon: full horizon during ``warmup_epochs``, then
    ``min((level + 1) * cl_epochs, pred_len)`` where ``level`` advances every
    ``step_size`` epochs after warmup.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    warmup_epochs: int = 0
    step_size: int = 1
    cl_epochs: int = 1


class TrainTricksConfig(BaseModel):
    """Optional, opt-in training tricks / pluggable callbacks.

    Every field defaults to disabled. When the whole section is omitted (the
    default), no callbacks are built and the train loop is byte-identical to the
    pre-callback behavior.
    """

    model_config = ConfigDict(extra="forbid")

    grad_clip_norm: float | None = None
    grad_clip_norm_type: float = 2.0
    grad_accum_steps: int = 1
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int
    batch_size: int = 32
    loss: str = "mse"
    loss_params: dict = Field(default_factory=dict)
    patience: int = 3
    optimizer: TrainOptimizerConfig = TrainOptimizerConfig()
    checkpoint: TrainCheckpointConfig = TrainCheckpointConfig()
    tricks: TrainTricksConfig = Field(default_factory=TrainTricksConfig)
