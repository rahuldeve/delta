from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class GT:
    th: float


@dataclass
class LT:
    th: float


type DSThreshold = GT | LT


class SupportedDatasets(Enum):
    SINGLE_TARGET_TBA = auto()
    DUAL_TARGET_TBA = auto()
    GSK_HEPG2 = auto()
    PK = auto()
    DB_MALARIA = auto()
    DB_HEPG2 = auto()
