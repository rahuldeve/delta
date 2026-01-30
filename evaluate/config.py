from dataclasses import dataclass, field
from enum import StrEnum, auto


class SplitType(StrEnum):
    RANDOM = auto()
    SCAFFOLD = auto()


@dataclass
class TrainConfig:
    batch_size: int = 8
    max_epochs: int = 30
    early_stopping_patience: int = 10
    n_splits: int = 2
    split_type: SplitType = SplitType.RANDOM
    random_seed: int = 42


@dataclass
class WandbDisabled:
    pass


@dataclass
class WandbEnabled:
    project_name: str = "evaluate_tba"
    tags: list[str] = field(default_factory=list)


WandbConfig = WandbDisabled | WandbEnabled
