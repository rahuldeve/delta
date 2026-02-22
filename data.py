import random

import numpy as np
import pandas as pd
import ray
import rdkit.Chem as Chem
import torch
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.rdBase import BlockLogs
from dataclasses import dataclass


@dataclass
class GT:
    th: float


@dataclass
class LT:
    th: float


type DSThreshold = GT | LT


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(smiles):
    with BlockLogs():
        params = Chem.SmilesParserParams()  # type: ignore
        params.removeHs = False
        mol = Chem.MolFromSmiles(smiles, params)  # type: ignore
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        return uncharged_parent_clean_mol


def mol_to_inchi(mol):
    with BlockLogs():
        return Chem.MolToInchi(mol)


def get_scaffold(mol) -> str:
    smi = Chem.MolToSmiles(mol)  # type: ignore
    scaffold: str = MurckoScaffoldSmiles(smi)  # type: ignore
    if len(scaffold) == 0:
        scaffold = smi
    return scaffold


def getMolDescriptors(mol):
    """calculate the full list of descriptors for a molecule missingVal is used if the descriptor cannot be calculated"""
    res = {}
    for nm, fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        with BlockLogs():
            try:
                val = fn(mol)
            except Exception:
                val = None

        res[nm] = val
    return res


def preprocess_row(row):
    smi = row["smiles"]
    mol = standardize(smi)

    if mol is None:
        return dict(inchi=None, scaffold=None) | {
            name: None for name, _ in Descriptors.descList
        }

    return dict(
        inchi=mol_to_inchi(mol), scaffold=get_scaffold(mol)
    ) | getMolDescriptors(mol)


def preprocess_ray(df):
    assert "smiles" in set(df.columns)

    df = (
        ray.data.from_pandas(df, override_num_blocks=len(df) // 64)
        .map(lambda row: row | preprocess_row(row))
        # filter any rows that have None. This includes any mols that have NaN descriptor values
        .filter(lambda row: not any(pd.isna(v) for v in row.values()))
        .to_pandas()
    )

    df = df.groupby(["inchi"]).filter(lambda x: len(x) == 1).reset_index(drop=True)
    clusters, _ = pd.factorize(df["scaffold"])
    df["cluster"] = pd.Series(clusters)
    df["mol"] = df["inchi"].map(Chem.MolFromInchi)
    df = df.drop(["smiles", "inchi", "scaffold"], axis=1)
    return df


def load_single_target_tba():
    df = pd.read_excel("../datasets/GSK_TBA_AH_JSFedit070425.xlsx")
    df = df.loc[:, ["smiles", "parent_remaining_24", "metabolite_detected"]]
    df = preprocess_ray(df)
    df["cont_target"] = df["parent_remaining_24"].round(1) / 100
    df["bin_target"] = df["cont_target"] > 0.5
    return df, GT(0.50)


def load_dual_target_tba():
    df = pd.read_excel("../datasets/GSK_TBA_AH_JSFedit070425.xlsx")
    df = df.loc[:, ["smiles", "parent_remaining_24", "metabolite_detected"]]
    df = preprocess_ray(df)
    df["cont_target"] = (
        df["parent_remaining_24"].round(1) / 100 + df["metabolite_detected"]
    )
    df["bin_target"] = df["cont_target"] > 1.5
    return df, GT(1.5)


def load_gsk_hepg2():
    df = pd.read_csv("../datasets/GSK_HepG2.csv")
    df = df.loc[:, ["SMILES", "% inhibition of HepG2 cell line: PCT_INHIB_HEPG2 (%)"]]
    df.columns = ["smiles", "per_inhibition"]

    df = preprocess_ray(df)
    df["cont_target"] = df["per_inhibition"] / 100
    df["bin_target"] = df["cont_target"] > 0.5
    return df, GT(0.5)


def load_derbyshire_malaria():
    df = pd.read_csv("../datasets/Derbyshire_malaria.csv")
    cols = ["Compound SMILES", "Parasite % Control Avg"]
    df = df.loc[:, cols]
    df.columns = ["smiles", "parasite_remain_per"]

    df = preprocess_ray(df)
    df["cont_target"] = df["parasite_remain_per"] / 100
    df["bin_target"] = df["cont_target"] < 0.15
    return df, LT(0.15)

def load_derbyshire_hepg2():
    df = pd.read_csv("../datasets/Derbyshire_malaria.csv")
    cols = ["Compound SMILES", "Liver % Control Avg"]
    df = df.loc[:, cols]
    df.columns = ["smiles", "liver_remain_per"]

    df = preprocess_ray(df)
    df["cont_target"] = df["liver_remain_per"] / 100
    df["bin_target"] = df["cont_target"] < 0.5
    return df, LT(0.5)


def load_pk():
    df = pd.read_csv("../datasets/PK.csv")
    df = df.loc[:, ['mol', 'AUC']]
    df.columns = ["smiles", 'auc']
    df = df.dropna().reset_index(drop=True)

    df = preprocess_ray(df)
    df["cont_target"] = df["auc"]
    df["bin_target"] = df["cont_target"] > 1000
    return df, GT(1000)
