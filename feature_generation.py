import numpy as np
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect  # type: ignore
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.validation import _check_n_features, _check_feature_names

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
        morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=int(self.n_bits))
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

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
