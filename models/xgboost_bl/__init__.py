from typing import Self

import numpy as np
import xgboost as xgb
from ghostml import optimize_threshold_from_predictions
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import StandardScaler

from config import TrainConfig
from misc import set_seeds
from models.abc import PreparedDatasetSplit, RefModel
from models.config import XGBoostConfig
from models.xgboost_bl.utils import CorrelationThreshold


class XGBoostRef(RefModel[XGBoostConfig]):
    def __init__(
        self, model: xgb.XGBClassifier, feature_pipeline: BaseEstimator
    ) -> None:
        self.model = model
        self.feature_pipeline = feature_pipeline

    @staticmethod
    def prepare_splits(*, train_df, val_df, test_df):
        return PreparedDatasetSplit(
            train_split=train_df, val_split=val_df, test_split=test_df
        )

    @classmethod
    def build(
        cls,
        *,
        model_config: XGBoostConfig,
        **kwargs,
    ) -> "XGBoostRef":
        feature_pipeline = make_pipeline(
            FunctionTransformer(
                lambda df: df.drop(["inchi", "smiles", "mol"], axis=1, errors="ignore")
            ),
            CorrelationThreshold(0.95),
            VarianceThreshold(0.1),
            StandardScaler(),
        ).set_output(transform="pandas")

        model = xgb.XGBClassifier(
            early_stopping_rounds=200, random_state=model_config.random_state, verbosity=0
        )
        return XGBoostRef(model, feature_pipeline)

    def train_func(
        self,
        *,
        train_split: DataFrame,
        val_split: DataFrame,
        train_config: TrainConfig,
        **kwargs,
    ) -> Self:
        set_seeds(train_config.random_seed)

        feat_cols = [c for c in train_split.columns if c.startswith("feat")]
        X_train = train_split.loc[:, feat_cols]
        y_train = train_split["bin_target"]

        X_val = val_split.loc[:, feat_cols]
        y_val = val_split["bin_target"]

        X_train_trans = self.feature_pipeline.fit_transform(X_train, y_train)
        X_val_trans = self.feature_pipeline.transform(X_val)

        self.model = self.model.fit(
            X_train_trans,
            y_train,
            eval_set=[(X_val_trans, y_val)],
            verbose=50
        )

        return self

    def tune_binary_classification_threshold(
        self,
        *,
        val_split: DataFrame,
        val_labels,
        train_config: TrainConfig,
        **kwargs,
    ) -> float:
        feat_cols = [c for c in val_split.columns if c.startswith("feat")]
        X_val = val_split.loc[:, feat_cols]
        y_val = val_split["bin_target"]
        X_val_trans = self.feature_pipeline.transform(X_val)

        val_pred_probs = self.model.predict_proba(X_val_trans)[:, 1]
        thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
        optimal_threshold = optimize_threshold_from_predictions(
            y_val,
            val_pred_probs,
            thresholds,
            ThOpt_metrics="Kappa",
            random_seed=train_config.random_seed,
        )

        return optimal_threshold

    def predict_func(
        self,
        *,
        test_split: DataFrame,
        binary_classification_threshold: float,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        feat_cols = [c for c in test_split.columns if c.startswith("feat")]
        X_test = test_split.loc[:, feat_cols]
        X_test_trans = self.feature_pipeline.transform(X_test)

        test_pred_probs = self.model.predict_proba(X_test_trans)[:, 1]
        test_preds = (test_pred_probs >= binary_classification_threshold).astype(float)
        return test_pred_probs, test_preds
