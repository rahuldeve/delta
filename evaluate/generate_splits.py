import pickle

import pandas as pd
import ray
import rdkit.Chem as Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.rdBase import BlockLogs
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ray.init(ignore_reinit_error=True)


def standardize(smiles):
    with BlockLogs():
        # follows the steps in
        # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smiles, params)  # type: ignore

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
        return pickle.dumps(uncharged_parent_clean_mol)


def mol_to_inchi(mol):
    with BlockLogs():
        return Chem.MolToInchi(mol)


def generate_features(mol):
    with BlockLogs():
        return {f"feat_{k}": v for k, v in CalcMolDescriptors(mol).items()}


def get_scaffold(mol) -> str:
    smi = Chem.MolToSmiles(mol)
    scaffold = MurckoScaffoldSmiles(smi)
    if len(scaffold) == 0:
        scaffold = smi
    return scaffold


def preprocess(path):
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]
    df.columns = ["smiles", "per_inhibition"]

    # standardize and convert to inchi
    df["mol"] = df["smiles"].map(standardize)
    df = df.dropna(subset=["mol"])
    df["inchi"] = df["mol"].map(mol_to_inchi)
    df = df.groupby(["inchi"]).filter(lambda x: len(x) == 1).reset_index(drop=True)

    df["is_cytotoxic"] = df["per_inhibition"] > 50.0

    df = generate_features(df)

    clusters, _ = pd.factorize(
        df["mol"]
        .map(Chem.MolToSmiles)  # type: ignore
        .map(get_scaffold)
    )
    df["cluster"] = pd.Series(clusters)
    df = df.drop(["smiles", "inchi"], axis=1)
    return df


def repeated_5x5_cv(df):
    df = df.copy()
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    for outer_idx in range(5):
        for inner_idx, (train_idxs, val_test_idxs) in enumerate(
            cv.split(df, groups=df["cluster"])
        ):
            df_train = df.loc[train_idxs].reset_index(drop=True)
            df_val_test = df.loc[val_test_idxs].reset_index(drop=True)

            val_idxs, test_idxs = next(
                GroupShuffleSplit(n_splits=1, test_size=0.5).split(
                    df_val_test, groups=df_val_test["cluster"]
                )
            )

            df_val = df_val_test.loc[val_idxs].reset_index(drop=True)
            df_test = df_val_test.loc[test_idxs].reset_index(drop=True)
            yield f"{outer_idx}x{inner_idx}", (df_train, df_val, df_test)


df = pd.read_csv("../GSK_HepG2.csv")
df = df.iloc[:, 1:]
df.columns = ["smiles", "per_inhibition"]

df = (
    (
        ray.data.from_pandas(df.reset_index(), override_num_blocks=len(df) // 64)
        .map(lambda row: row | {"mol_ser": standardize(row["smiles"])})
        .filter(lambda row: row["mol_ser"] is not None)
        .map(lambda row: row | {"inchi": mol_to_inchi(pickle.loads(row["mol_ser"]))})
        .map(lambda row: row | generate_features(pickle.loads(row["mol_ser"])))
        .map(lambda row: row | {"scaffold": get_scaffold(pickle.loads(row["mol_ser"]))})
    )
    .materialize()
    .to_pandas()
)

df = df.groupby(["inchi"]).filter(lambda x: len(x) == 1).reset_index(drop=True)

clusters, _ = pd.factorize(df["scaffold"])
df["cluster"] = pd.Series(clusters)
df = df.drop(["smiles", "inchi", "scaffold"], axis=1)


splits_generator = repeated_5x5_cv(df)
for split_idx, (df_train, df_val, df_test) in splits_generator:
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    pd.concat([df_train, df_val, df_test]).to_parquet(
        f"./generated_splits/split_{split_idx}.parquet"
    )
