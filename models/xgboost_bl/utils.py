import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import (
    _check_feature_names, # type: ignore
    _check_n_features, # type: ignore
    check_is_fitted,
    validate_data, # type: ignore
)


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


class MorganFP(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    def __init__(self, rdkit_mol_col_name, n_bits=4096) -> None:
        self.n_bits = n_bits
        self.rdkit_mol_col_name = rdkit_mol_col_name

    @property
    def _n_features_out(self):
        return self.n_bits

    @property
    def n_features_(self):
        return self.n_features_in_  # type: ignore

    def smiles_to_morgan_fingerprint(self, mol):
        fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.n_bits)
        morgan_fp = fg.GetFingerprint(mol)
        fp_array = np.zeros((1,))
        ConvertToNumpyArray(morgan_fp, fp_array)
        return fp_array

    def fit(self, X, y=None):
        _check_n_features(self, X, reset=True)  # type: ignore
        _check_feature_names(self, X, reset=True)  # type: ignore

        # check if smiles column is present
        assert self.rdkit_mol_col_name in X.columns

        # X = self._validate_data(
        #     X, y
        #     # dtype=[np.float64, np.float32, np.str_],
        #     # ensure_2d=True
        # )
        return self

    def transform(self, X, y=None):
        X = X[self.rdkit_mol_col_name].map(
            lambda x: self.smiles_to_morgan_fingerprint(x)
        )
        X = np.vstack(X)
        return X
