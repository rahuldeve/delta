import random

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.rdBase import BlockLogs

RANDOM_SEED = 42


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(smiles):
    with BlockLogs():
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smiles, params)  # type: ignore
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        return uncharged_parent_clean_mol


def generate_features(df):
    with BlockLogs():
        feats = pd.DataFrame.from_records(df["mol"].map(CalcMolDescriptors).tolist())
        feats.columns = [f"feat_{f}" for f in feats.columns]
        df = pd.concat(
            [
                df.reset_index(drop=True),
                feats,
            ],
            axis=1,
        )

    return df


def mol_to_inchi(mol):
    with BlockLogs():
        return Chem.MolToInchi(mol)


def get_scaffold(mol) -> str:
    smi = Chem.MolToSmiles(mol)
    scaffold = MurckoScaffoldSmiles(smi)
    if len(scaffold) == 0:
        scaffold = smi
    return scaffold
