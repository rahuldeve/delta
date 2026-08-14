from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal


class SplitType(StrEnum):
    RANDOM = auto()
    SCAFFOLD = auto()
    BUTINA = auto()


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 10
    n_splits: int = 5
    use_feats: bool = False
    split_type: SplitType = SplitType.BUTINA
    random_seed: int = 42


@dataclass
class LogConfig:
    """Instrumentation depth for `train.cli`. Ignored when wandb is disabled."""

    # Gradient L2 norms, per parameter tensor plus a total. 0 disables them.
    grad_norm_every_n_steps: int = 1
    # Parameter/gradient histograms via wandb.watch. 0 disables them.
    watch_log_freq: int = 50
    watch_log: Literal["gradients", "parameters", "all"] = "all"


@dataclass
class WandbDisabled:
    pass


@dataclass
class WandbEnabled:
    project_name: str
    tags: list[str] = field(default_factory=list)
    model_name_suffix: str | None = None


WandbConfig = WandbDisabled | WandbEnabled
