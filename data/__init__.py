from dataclasses import dataclass


@dataclass
class GT:
    th: float


@dataclass
class LT:
    th: float


type DSThreshold = GT | LT
