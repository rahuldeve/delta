import random
from functools import wraps
from typing import Callable, Concatenate, ParamSpec

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from joblib import Parallel, delayed
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.rdBase import BlockLogs
from sklearn import metrics

from commons.confidence import delong_confidence_intervals


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize(smiles):
    with BlockLogs():
        # follows the steps in
        # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ
        mol = Chem.MolFromSmiles(smiles)  # type: ignore

        if mol is None:
            return None

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = (
            rdMolStandardize.Uncharger()
        )  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        # te = rdMolStandardize.TautomerEnumerator() # idem
        # taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        return uncharged_parent_clean_mol


def generate_features(df):
    with BlockLogs():
        df = pd.concat(
            [
                df.reset_index(drop=True),
                pd.DataFrame.from_records(df["mol"].map(CalcMolDescriptors).tolist()),
            ],
            axis=1,
        )

    return df


Param = ParamSpec("Param")
BaseFunc = Callable[Concatenate[pd.DataFrame, Param], pd.DataFrame]
WrappedFunc = Callable[Concatenate[pd.DataFrame, Param], pd.DataFrame]


def parallelize(n_jobs: int = 4) -> Callable[[BaseFunc], WrappedFunc]:
    def _inner(f: BaseFunc) -> WrappedFunc:
        @wraps(f)
        def parallel_func(
            df: pd.DataFrame, *args: Param.args, **kwargs: Param.kwargs
        ) -> pd.DataFrame:
            num_chunks = len(df) // n_jobs
            tail_size = len(df) % num_chunks
            if tail_size == 0:
                chunks = np.vsplit(df, num_chunks)
            else:
                head = df.iloc[:-tail_size, :]
                tail = df.iloc[-tail_size:, :]
                chunks = np.vsplit(head, num_chunks) + [tail]

            jobs = Parallel(n_jobs=n_jobs)(
                delayed(f)(chunk, *args, **kwargs) for chunk in chunks
            )
            return pd.concat(jobs).reset_index(drop=True)  # type: ignore

        return parallel_func

    return _inner


@parallelize(16)
def standardize_df(df):
    df["inchi"] = df["smiles"].map(standardize)
    return df


@parallelize(16)
def add_features(df):
    with BlockLogs():
        df["mol"] = df["inchi"].map(Chem.MolFromInchi)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame.from_records(df["mol"].map(CalcMolDescriptors).tolist()),
        ],
        axis=1,
    )

    return df


def get_scaffold(smi) -> str:
    """
    Generate the Bemis-Murcko scaffold for a given molecule.

    :param smi: A SMILES string or an RDKit molecule object representing the
                molecule for which to generate the scaffold.
    :return: A SMILES string representing the Bemis-Murcko scaffold of the input
             molecule. If the scaffold cannot be generated, the input SMILES
             string is returned.
    """
    scaffold = MurckoScaffoldSmiles(smi)
    if len(scaffold) == 0:
        scaffold = smi
    return scaffold


def calc_scores(y_pred_prob, y_pred, y_true):
    metrics_dict = dict()
    metrics_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    metrics_dict["balanced_accuracy"] = metrics.balanced_accuracy_score(y_true, y_pred)
    metrics_dict["f1"] = metrics.f1_score(y_true, y_pred)
    metrics_dict["precision"] = metrics.precision_score(y_true, y_pred)
    metrics_dict["recall"] = metrics.recall_score(y_true, y_pred)
    metrics_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_pred_prob)
    metrics_dict["average_precision"] = metrics.average_precision_score(
        y_true, y_pred_prob
    )

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    metrics_dict["specificity"] = tn / (tn + fp)
    metrics_dict["sensitivity"] = tp / (tp + fn)
    auc, (lb, ub) = delong_confidence_intervals(y_true, y_pred_prob)
    metrics_dict["test_delong_auc"] = auc
    metrics_dict["lb"] = lb
    metrics_dict["ub"] = ub

    return metrics_dict
