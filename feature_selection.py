import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, validate_data
from stratified_continious_split import ContinuousStratifiedKFold


class CrossValidatedFeatureSelector(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    def __init__(self, estimator, n_splits=5) -> None:
        self.estimator = estimator
        self.n_splits = n_splits
        self.cv = ContinuousStratifiedKFold(n_splits=self.n_splits)

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.mask_

    def fit(self, X, y):
        X = validate_data(
            self, 
            X=X,
            dtype=float,
            ensure_all_finite="allow-nan"
        )  # type: ignore
        X, y = check_X_y(X, y)

        feat_importances = []
        for train_idxs, _ in self.cv.split(X, y):
            X_train, y_train = X[train_idxs], y[train_idxs]
            self.estimator.fit(X_train, y_train)
            feat_importances.append(self.estimator.feature_importances_)

        self.mean_feature_importance_ = np.vstack(feat_importances).mean(axis=0)
        self.mask_ = ~np.isclose(self.mean_feature_importance_, 0.0)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class CorrelationThreshold(SelectorMixin, BaseEstimator):
    def __init__(self, threshold=0.95) -> None:
        self.threshold = threshold

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # True for preserved columns, False for dropped columns
        self.mask_ = np.array(
            [not any(upper[column] > self.threshold) for column in upper.columns]
        )

        X = validate_data(  # type: ignore
            self,
            X=X,
            accept_sparse=("csr", "csc"),
            dtype=float,
            ensure_all_finite="allow-nan",
        )

        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.mask_
