import pandas as pd
import rdkit.Chem as Chem
from rdkit.rdBase import BlockLogs
from sklearn.model_selection import GroupShuffleSplit
from commons.utils import get_scaffold, standardize


def mol_to_inchi(mol):
    with BlockLogs():
        return Chem.MolToInchi(mol)


def load_and_split_gsk_dataset(path, RANDOM_SEED):
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]
    df.columns = ["smiles", "per_inhibition"]

    # standardize and convert to inchi
    df["mol"] = df["smiles"].map(standardize)
    df = df.dropna(subset=["mol"])
    df["inchi"] = df["mol"].map(mol_to_inchi)
    df = df.groupby(["inchi"]).filter(lambda x: len(x) == 1).reset_index(drop=True)

    clusters, _ = pd.factorize(
        df["mol"]
        .map(Chem.MolToSmiles)  # type: ignore
        .map(get_scaffold)
    )
    clusters = pd.Series(clusters)

    df = df.drop(["smiles", "inchi"], axis=1)

    splitter = GroupShuffleSplit(n_splits=1, random_state=RANDOM_SEED)
    train_idxs, val_test_idxs = next(splitter.split(df, groups=clusters))
    df_train = df.loc[train_idxs].reset_index(drop=True)
    df_val_test = df.loc[val_test_idxs].reset_index(drop=True)
    clusters_val_test = clusters.iloc[val_test_idxs].reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, random_state=RANDOM_SEED, test_size=0.5)
    val_idxs, test_idxs = next(splitter.split(df_val_test, groups=clusters_val_test))
    df_val = df_val_test.loc[val_idxs].reset_index(drop=True)
    df_test = df_val_test.loc[test_idxs].reset_index(drop=True)

    return df_train, df_val, df_test
