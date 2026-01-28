from dataclasses import dataclass
from enum import StrEnum, auto

RANDOM_SEED = 42


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
