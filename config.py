from dataclasses import dataclass, field
from enum import StrEnum, auto


class SplitType(StrEnum):
    RANDOM = auto()
    SCAFFOLD = auto()
    BUTINA = auto()


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 20
    n_splits: int = 5
    split_type: SplitType = SplitType.BUTINA
    random_seed: int = 42


@dataclass
class WandbDisabled:
    pass


@dataclass
class WandbEnabled:
    project_name: str = "evaluate_tba"
    tags: list[str] = field(default_factory=list)


WandbConfig = WandbDisabled | WandbEnabled
