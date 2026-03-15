from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, Self, TypeVar

import numpy as np
from pandas import DataFrame

from config import TrainConfig


class PreparedDatasetSplit(NamedTuple):
    train_split: Any
    val_split: Any
    test_split: Any
    extras: dict[str, Any] = dict()


class ModelConfig(ABC):
    pass


ModelConfigType = TypeVar("ModelConfigType", bound=ModelConfig)


class RefModel(ABC, Generic[ModelConfigType]):
    @staticmethod
    @abstractmethod
    def prepare_splits(
        *, train_df: DataFrame, val_df: DataFrame, test_df: DataFrame
    ) -> PreparedDatasetSplit:
        pass

    @classmethod
    @abstractmethod
    def build(cls, *, model_config: ModelConfigType, **kwargs) -> Self:
        pass

    @abstractmethod
    def train_func(
        self,
        *,
        train_split: Any,
        val_split: Any,
        train_config: TrainConfig,
        **kwargs,
    ) -> Self:
        pass

    @abstractmethod
    def tune_binary_classification_threshold(
        self, *, val_split: Any, val_labels: Any, train_config: TrainConfig, **kwargs
    ) -> float:
        pass

    @abstractmethod
    def predict_func(
        self, *, test_split: Any, binary_classification_threshold: float, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        pass
