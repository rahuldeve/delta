import numpy as np
import pandas as pd
import ray
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.ML.Cluster import Butina
from rdkit.rdBase import BlockLogs


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


# ----------- Clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
def taylor_butina_clustering(
    fp_list: list[DataStructs.ExplicitBitVect], cutoff: float = 0.65
) -> list[int]:
    """Cluster a set of fingerprints using the RDKit Taylor-Butina implementation

    :param fp_list: a list of fingerprints
    :param cutoff: distance cutoff (1 - Tanimoto similarity)
    :return: a list of cluster ids
    """
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    cluster_res = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = np.zeros(nfps, dtype=int)
    for cluster_num, cluster in enumerate(cluster_res):
        for member in cluster:
            cluster_id_list[member] = cluster_num
    return cluster_id_list.tolist()


def get_butina_clusters(mol_list, cutoff: float = 0.65) -> list[int]:
    """
    Cluster a list of SMILES strings using the Butina clustering algorithm.

    :param cutoff: The cutoff value to use for clustering
    :return: List of cluster labels corresponding to each SMILES string in the input list.
    """
    fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_list = [fg.GetFingerprint(x) for x in mol_list]
    return taylor_butina_clustering(fp_list, cutoff=cutoff)


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
    df["mol"] = df["inchi"].map(Chem.MolFromInchi)

    clusters, _ = pd.factorize(df["scaffold"])
    df["scaffold_cluster"] = pd.Series(clusters)
    df["butina_cluster"] = get_butina_clusters(df["mol"])

    df = df.drop(["smiles", "inchi", "scaffold"], axis=1)
    return df
